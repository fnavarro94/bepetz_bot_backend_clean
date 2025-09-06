# libs/vet_chat_tasks/openai_helper.py
from __future__ import annotations

import os
from typing import Generator, Iterable, Optional, Dict, Any

from openai import OpenAI, Stream  # Stream is for typing of the streaming context

"""
Env vars

OPENAI_API_KEY           – required
OPENAI_BASE_URL          – optional (self-hosted proxy etc.)
OPENAI_ORG_ID            – optional
OPENAI_MODEL             – model name, e.g. "gpt-4o-mini", "gpt-4o", "gpt-5.1-mini"
OPENAI_REASONING_EFFORT  – for GPT-5 only ("low" | "medium" | "high"), default "medium"
OPENAI_TEMPERATURE       – default 0.7
OPENAI_MAX_OUTPUT_TOKENS – for Responses API (all models)
"""

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

    - Uses Responses API with streaming (`responses.stream`).
    - Sends `instructions` (system/developer message), `input` (user text).
    - Optionally chains context via `previous_response_id`.
    - Captures `last_response_id` after streaming to enable continuity.
    - Only sends the `reasoning` param for GPT-5.* models.

    Usage:
        helper = OpenAIChatHelper()
        for chunk in helper.stream_text(
            user_text="Hello there",
            system_prompt="You are helpful.",
            previous_response_id="<prior-response-id-if-any>",
            metadata={"consultation_id": "abc123"},
            store=True,
        ):
            publish(chunk)  # e.g., to Redis/relay

        # After the stream finishes:
        last_id = helper.last_response_id
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.last_response_id: Optional[str] = None

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            organization=os.environ.get("OPENAI_ORG_ID") or None,
        )

        # Defaults
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
        system_prompt: Optional[str] = None,   # sent as 'instructions'
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,  # only for GPT-5.*
        previous_response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store: bool = True,
    ) -> Generator[str, None, None]:
        """
        Stream text deltas (strings) as they arrive from the Responses API.
        """
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
        """
        Retrieve a previously stored response by ID.
        Note: You must have called with store=True when creating it.
        """
        return self.client.responses.retrieve(response_id)

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
        """
        Responses API with streaming.
        - `instructions`: system/developer guidance
        - `input`: user text
        - `previous_response_id`: chains the conversation
        - `store`: persist server-side so you can retrieve by ID later
        - `metadata`: arbitrary dict echoed back in retrieval/list contexts
        """
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

        # Stream and yield deltas
        with self.client.responses.stream(**kwargs) as stream:  # type: Stream
            for event in stream:
                # We care about:
                #  - "response.output_text.delta": incremental text
                #  - "response.error": raise
                et = getattr(event, "type", None)
                if et == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
                elif et == "response.error":
                    msg = getattr(event, "error", None)
                    raise RuntimeError(f"OpenAI stream error: {msg}")
                # Optional: handle other events if you add tools, etc.

            # Ensure the stream finishes and capture the response id
            final = stream.get_final_response()
            try:
                self.last_response_id = getattr(final, "id", None)
            except Exception:
                self.last_response_id = None

    @staticmethod
    def _coalesce_float(value: Optional[float], default: float) -> float:
        try:
            return float(value) if value is not None else float(default)
        except Exception:
            return float(default)
