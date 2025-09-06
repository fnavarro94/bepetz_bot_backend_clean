# libs/vet_chat_tasks/openai_helper.py
from __future__ import annotations

import os
from typing import Generator, Iterable, Optional, Dict, Any

from openai import OpenAI, Stream  # Stream is for typing of the streaming context

def _is_gpt5(model: str) -> bool:
    return str(model or "").lower().startswith("gpt-5")

def _float_from_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


class OpenAIChatHelper:
    """
    Unified streaming helper using the Responses API for ALL models.
    Now supports server-side cancellation via responses.cancel(response_id).
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.last_response_id: Optional[str] = None
        self.current_response_id: Optional[str] = None  # <── NEW

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            organization=os.environ.get("OPENAI_ORG_ID") or None,
        )

        self.default_temperature = _float_from_env("OPENAI_TEMPERATURE", 0.7)
        self.default_reasoning = os.getenv("OPENAI_REASONING_EFFORT", "medium")
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
        reasoning_effort: Optional[str] = None,  # only for GPT-5.*
        previous_response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store: bool = True,
    ) -> Generator[str, None, None]:
        model = self.model
        temperature = self._coalesce_float(temperature, self.default_temperature)
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_max_tokens

        yield from self._stream_with_responses_api(
            model=model,
            user_text=user_text,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=(reasoning_effort or self.default_reasoning) if _is_gpt5(model) else None,
            previous_response_id=previous_response_id,
            metadata=metadata,
            store=store,
        )

    def retrieve_response(self, response_id: str):
        return self.client.responses.retrieve(response_id)

    # NEW: best-effort server-side cancel
    def cancel_current(self) -> bool:
        """
        Attempts to cancel the currently-streaming response on OpenAI's side.
        Returns True if a cancel call was issued; False otherwise.
        """
        rid = self.current_response_id
        if not rid:
            return False
        try:
            # OpenAI Responses API supports cancellation of an in-flight response id
            self.client.responses.cancel(rid)
            return True
        except Exception:
            # Even if this fails, closing the stream (context exit) already stops generation.
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
        previous_response_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        store: bool,
    ) -> Iterable[str]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": user_text,
            "store": bool(store),
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if metadata:
            kwargs["metadata"] = metadata

        self.current_response_id = None  # reset before starting a new stream
        with self.client.responses.stream(**kwargs) as stream:  # type: Stream
            for event in stream:
                et = getattr(event, "type", None)

                # Capture the response id as early as possible
                if et == "response.created":
                    # Some SDK builds expose event.response.id; fall back to event.id if present.
                    rid = None
                    try:
                        rid = getattr(getattr(event, "response", None), "id", None)
                    except Exception:
                        pass
                    rid = rid or getattr(event, "id", None)
                    if rid:
                        self.current_response_id = rid

                elif et == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta

                elif et == "response.error":
                    msg = getattr(event, "error", None)
                    raise RuntimeError(f"OpenAI stream error: {msg}")

                # (You can handle tool events here if you add them later.)

            # Finish & capture final id; clear current stream id
            final = stream.get_final_response()
            try:
                self.last_response_id = getattr(final, "id", None)
            except Exception:
                self.last_response_id = None

        # We're out of the context: no longer streaming
        self.current_response_id = None

    @staticmethod
    def _coalesce_float(value: Optional[float], default: float) -> float:
        try:
            return float(value) if value is not None else float(default)
        except Exception:
            return float(default)
