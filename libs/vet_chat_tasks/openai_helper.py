# libs/vet_chat_tasks/openai_helper.py
from __future__ import annotations

import os
from typing import Generator, Iterable, Optional, Dict, Any, Union

from openai import OpenAI, Stream  # Stream is for typing the streaming context

# logging (inherits your JsonFormatter via vet-chat-worker.*)
import logging
log = logging.getLogger("vet-chat-worker.openai")


def _is_gpt5(model: str | None) -> bool:
    return str(model or "").lower().startswith("gpt-5")


def _is_gpt4_family(model: str | None) -> bool:
    m = str(model or "").lower()
    return m.startswith("gpt-4") or m.startswith("gpt-4o")


def _float_from_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _capabilities_for_model(model: str) -> Dict[str, bool]:
    """
    Centralized capability switchboard.
    Keep this conservative and adjust as OpenAI updates docs.
    """
    m = (model or "").lower()

    if _is_gpt5(m):
        # Reasoning family (no classic sampling controls)
        return {
            "supports_temperature": False,
            "supports_reasoning_effort": True,
            "supports_verbosity": True,   # per launch notes; some surfaces may not yet accept it
        }

    # Default: GPT-4.* models (classic sampling), no reasoning/verbosity
    if _is_gpt4_family(m):
        return {
            "supports_temperature": True,
            "supports_reasoning_effort": False,
            "supports_verbosity": False,
        }

    # Fallback (non-5 models) — assume classic sampling support, no reasoning/verbosity
    return {
        "supports_temperature": True,
        "supports_reasoning_effort": False,
        "supports_verbosity": False,
    }


def _normalize_annotation(ann: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize different SDK shapes for output-text annotations into a simple dict.
    We care about URL/file citations and their spans.
    """
    try:
        atype = getattr(ann, "type", None)
        # Some SDKs wrap the payload; unwrap common shapes
        src = getattr(ann, "url_citation", None) or getattr(ann, "file_citation", None) or ann

        start = getattr(src, "start_index", None)
        end   = getattr(src, "end_index", None)
        url   = getattr(src, "url", None)
        title = getattr(src, "title", None)
        file_id  = getattr(src, "file_id", None) or getattr(ann, "file_id", None)
        filename = getattr(src, "filename", None) or getattr(ann, "filename", None)

        if atype in {"url_citation", "file_citation", "container_file_citation"}:
            return {
                "type": atype,
                "start": start,
                "end": end,
                "url": url,
                "title": title,
                "file_id": file_id,
                "filename": filename,
            }
    except Exception:
        pass
    return None


class OpenAIChatHelper:
    """
    Unified streaming helper using the Responses API for ALL models.
    - Filters unsupported params per model family (e.g., drops `temperature` for GPT-5).
    - Supports server-side cancellation via responses.cancel(response_id).
    - Yields either:
        * str  -> text deltas
        * dict -> inline citation events: {"event":"citation","start":int,"end":int,"url":str,"title":str|None}
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        print(f"El modelo que se usa es {self.model}")
        self.last_response_id: Optional[str] = None
        self.current_response_id: Optional[str] = None  # tracks in-flight stream

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            organization=os.environ.get("OPENAI_ORG_ID") or None,
        )

        # Defaults (env-tunable)
        self.default_temperature = _float_from_env("OPENAI_TEMPERATURE", 0.7)
        self.default_reasoning = os.getenv("OPENAI_REASONING_EFFORT", "medium")  # low|medium|high|minimal*
        self.default_verbosity = os.getenv("OPENAI_VERBOSITY")  # low|medium|high (GPT-5 only)
        self.default_max_tokens = os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
        if self.default_max_tokens is not None:
            try:
                self.default_max_tokens = int(self.default_max_tokens)
            except Exception:
                self.default_max_tokens = None

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def stream_text(
        self,
        *,
        user_text: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,  # GPT-5 only
        verbosity: Optional[str] = None,         # GPT-5 only
        previous_response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store: bool = True,
    ) -> Generator[Union[str, Dict[str, Any]], None, None]:
        model = self.model
        caps = _capabilities_for_model(model)

        # Coalesce defaults, then drop if unsupported
        effective_temperature: Optional[float] = None
        if caps["supports_temperature"]:
            try:
                effective_temperature = (
                    float(temperature) if temperature is not None else float(self.default_temperature)
                )
            except Exception:
                effective_temperature = float(self.default_temperature)

        effective_reasoning = (reasoning_effort or self.default_reasoning) if caps["supports_reasoning_effort"] else None
        effective_verbosity = (verbosity or self.default_verbosity) if caps["supports_verbosity"] else None

        if max_output_tokens is None:
            max_output_tokens = self.default_max_tokens

        yield from self._stream_with_responses_api(
            model=model,
            user_text=user_text,
            system_prompt=system_prompt,
            temperature=effective_temperature,          # None if unsupported
            max_output_tokens=max_output_tokens,
            reasoning_effort=effective_reasoning,       # None if unsupported
            verbosity=effective_verbosity,              # None if unsupported
            previous_response_id=previous_response_id,
            metadata=metadata,
            store=store,
        )

    def retrieve_response(self, response_id: str):
        return self.client.responses.retrieve(response_id)

    # Best-effort server-side cancel
    def cancel_current(self) -> bool:
        rid = self.current_response_id
        if not rid:
            return False
        try:
            self.client.responses.cancel(rid)
            return True
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────
    def _stream_with_responses_api(
        self,
        *,
        model: str,
        user_text: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        reasoning_effort: Optional[str],
        verbosity: Optional[str],
        previous_response_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        store: bool,
    ) -> Iterable[Union[str, Dict[str, Any]]]:

        kwargs: Dict[str, Any] = {
            "model": model,
            "input": user_text,
            "store": bool(store),
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt

        # Only include keys the current model family supports
        caps = _capabilities_for_model(model)
        if (temperature is not None) and caps["supports_temperature"]:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        if reasoning_effort and caps["supports_reasoning_effort"]:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if verbosity and caps["supports_verbosity"]:
            # The verbosity control is top-level in Responses per GPT-5 launch notes.
            kwargs["verbosity"] = str(verbosity)

        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if metadata:
            kwargs["metadata"] = metadata

        # --- Web search toggle (env-driven) ---
        enable_web = os.getenv("VET_CHAT_ENABLE_WEB_SEARCH", "1") == "1"
        if enable_web:
            web_tool: dict = {"type": "web_search"}
            # Optional: constrain sources to specific sites (comma-separated list)
            sites = os.getenv("VET_CHAT_WEB_SITES")  # e.g., "who.int,cdc.gov"
            if sites:
                web_tool["sites"] = [s.strip() for s in sites.split(",") if s.strip()]
            # Optional: approximate user location for more relevant results
            # VET_CHAT_WEB_LOCATION="EC,Guayaquil,Guayas"  -> country, city, region
            loc = os.getenv("VET_CHAT_WEB_LOCATION")
            if loc:
                parts = [p.strip() for p in loc.split(",")]
                web_tool["user_location"] = {
                    "type": "approximate",
                    "approximate": {
                        "country": (parts[0] if len(parts) > 0 else None),
                        "city":    (parts[1] if len(parts) > 1 else None),
                        "region":  (parts[2] if len(parts) > 2 else None),
                    },
                }
            kwargs["tools"] = [web_tool]

            # Optional: force the model to actually use web search for this request
            if os.getenv("VET_CHAT_FORCE_WEB_SEARCH", "0") == "1":
                kwargs["tool_choice"] = {"type": "web_search"}  # or just "auto" (default)

        # Log AFTER tools are attached so you can see them
        log.info(
            "openai_request_built",
            extra={
                "requested_model": model,
                "kwargs_model": kwargs.get("model"),
                "supports_temperature": caps["supports_temperature"],
                "supports_reasoning_effort": caps["supports_reasoning_effort"],
                "supports_verbosity": caps["supports_verbosity"],
                "has_temperature": "temperature" in kwargs,
                "has_reasoning": "reasoning" in kwargs,
                "has_verbosity": "verbosity" in kwargs,
                "max_output_tokens": kwargs.get("max_output_tokens"),
                "has_tools": "tools" in kwargs,
                "tool_choice": kwargs.get("tool_choice"),
                "tools": kwargs.get("tools"),
            },
        )

        # Start streaming
        self.current_response_id = None
        with self.client.responses.stream(**kwargs) as stream:  # type: Stream
            for event in stream:
                et = getattr(event, "type", None)

                if et == "response.created":
                    rid = None
                    try:
                        rid = getattr(getattr(event, "response", None), "id", None)
                    except Exception:
                        pass
                    rid = rid or getattr(event, "id", None)
                    if rid:
                        self.current_response_id = rid

                # Token deltas (text)
                elif et == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta

                # Inline citation annotations arriving during the stream
                # Known event name: "response.output_text.annotation.added"
                elif et and et.startswith("response.output_text.annotation"):
                    ann = getattr(event, "annotation", None)
                    norm = _normalize_annotation(ann) if ann is not None else None
                    if norm:
                        payload = {
                            "event": "citation",
                            "start": norm.get("start"),
                            "end": norm.get("end"),
                            "url": norm.get("url"),
                            "title": norm.get("title"),
                            "source_type": norm.get("type"),
                            "file_id": norm.get("file_id"),
                            "filename": norm.get("filename"),
                        }
                        # Stream the citation immediately so FE can place it next to the text
                        yield payload

                elif et == "response.error":
                    msg = getattr(event, "error", None)
                    raise RuntimeError(f"OpenAI stream error: {msg}")

            # Finish & capture final id
            final = stream.get_final_response()

            try:
                log.info(
                    "openai_response_final",
                    extra={
                        "response_id": getattr(final, "id", None),
                        "final_model": getattr(final, "model", None),
                    },
                )
            except Exception:
                pass

            try:
                self.last_response_id = getattr(final, "id", None)
            except Exception:
                self.last_response_id = None

        self.current_response_id = None
