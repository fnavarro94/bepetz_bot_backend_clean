# libs/vet_chat_tasks/openai_helper.py
from __future__ import annotations

import os
from typing import Generator, Iterable, Optional

from openai import OpenAI, Stream

"""
Env vars

OPENAI_API_KEY        – required
OPENAI_BASE_URL       – optional (self-hosted proxy etc.)
OPENAI_ORG_ID         – optional
OPENAI_MODEL          – model name, e.g. "gpt-5.1-mini" or "gpt-4o-mini"
OPENAI_REASONING_EFFORT – for GPT-5 only ("low" | "medium" | "high"), default "medium"
OPENAI_TEMPERATURE    – default 0.7
OPENAI_MAX_OUTPUT_TOKENS – for GPT-5 (Responses) or GPT-4 (Chat) appropriately
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
    Unified streaming helper:

    - For GPT-5.* → Responses API with streaming, supports `reasoning`
    - For GPT-4/4o* → Chat Completions API with streaming

    Usage:
        helper = OpenAIChatHelper()
        for chunk in helper.stream_text(user_text="Hello", system_prompt="You are helpful"):
            publish(chunk)  # e.g., to Redis/relay
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Yields text deltas as they arrive.
        """
        model = self.model
        temperature = self._coalesce_float(temperature, self.default_temperature)
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_max_tokens

        if _is_gpt5(model):
            yield from self._stream_with_responses_api(
                model=model,
                user_text=user_text,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort or self.default_reasoning,
            )
        else:
            yield from self._stream_with_chat_completions(
                model=model,
                user_text=user_text,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_output_tokens,  # chat-completions uses "max_tokens"
            )

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
    ) -> Iterable[str]:
        """
        GPT-5: Responses API (supports `reasoning`).
        - We pass `input` as a plain string and optional `system`.
        """
        kwargs = {
            "model": model,
            "input": user_text,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        # Stream and yield deltas
        with self.client.responses.stream(**kwargs) as stream:  # type: Stream
            for event in stream:
                # The event types we care about for text deltas:
                #  - "response.output_text.delta" (preferred)
                #  - "response.error" (raise)
                et = getattr(event, "type", None)
                if et == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
                elif et == "response.error":
                    msg = getattr(event, "error", None)
                    raise RuntimeError(f"OpenAI stream error: {msg}")
                # You could also watch "response.completed" if you need to hook EOS.
            # Ensure the stream finishes cleanly (raises if server aborted)
            _ = stream.get_final_response()

    def _stream_with_chat_completions(
        self,
        *,
        model: str,
        user_text: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Iterable[str]:
        """
        GPT-4/4o: Chat Completions API (no `reasoning` field).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            try:
                part = chunk.choices[0].delta
                if part and part.content:
                    yield part.content
            except Exception:
                # Defensive: some chunks may be control frames
                continue

    @staticmethod
    def _coalesce_float(value: Optional[float], default: float) -> float:
        try:
            return float(value) if value is not None else float(default)
        except Exception:
            return float(default)
