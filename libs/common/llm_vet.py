# common/llm_vet.py
import os
import json
import logging
from typing import Any, Dict, Optional, List

import asyncio, contextlib
from typing import Callable, Awaitable

import contextlib
# imports para leer datos de base de datos medica
from typing import Any, Dict, Optional
from common.vet_db_context import (
    build_session_snapshot_from_consultation,
    fetch_definitive_diagnosis,
    fetch_prescribed_medications,
)

import asyncio
from typing import Callable, Awaitable
import typing

logger = logging.getLogger("vet-llm")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = os.getenv("VET_LLM_MODEL", "gpt-5-mini")
# NEW: DB config for consultation → session_snapshot
PG_DSN = os.getenv("PG_DSN")  # e.g., "postgresql://user:pass@localhost:6543/bepetzdb"

# GPT-5-only knobs (optional)
REASONING_EFFORT = os.getenv("VET_LLM_REASONING_EFFORT", "minimal")  # minimal|low|medium|high
VERBOSITY = os.getenv("VET_LLM_VERBOSITY", "low")                    # low|medium|high
MAX_COMPLETION_TOKENS = int(os.getenv("VET_LLM_MAX_COMPLETION_TOKENS", "0") or 0)

# Non–GPT-5 knobs (ignored by GPT-5)
TEMP = os.getenv("VET_LLM_TEMPERATURE")
TOP_P = os.getenv("VET_LLM_TOP_P")

# ── Lazy OpenAI client (safe to import in API container) ──────────────────────
_AsyncOpenAI = None
_client = None
_openai_import_err: Optional[BaseException] = None


class LLMUserCancelled(Exception):
    """Raised when a user cancel occurs while a model response is in progress."""
    pass

def _ensure_openai_import():
    """Import SDK and exception types on demand."""
    global _AsyncOpenAI, _openai_import_err
    if _AsyncOpenAI is not None:
        return
    try:
        from openai import AsyncOpenAI  # pip install openai>=1.72.0
        # import exceptions for granular handling
        # Note: Some environments may not have all of these—fallbacks below.
        from openai import (  # type: ignore
            APIConnectionError, APITimeoutError, RateLimitError,
            AuthenticationError, BadRequestError, APIError
        )
        # expose to module namespace for type checking/except blocks
        globals().update({
            "APIConnectionError": APIConnectionError,
            "APITimeoutError": APITimeoutError,
            "RateLimitError": RateLimitError,
            "AuthenticationError": AuthenticationError,
            "BadRequestError": BadRequestError,
            "APIError": APIError,
        })
        _AsyncOpenAI = AsyncOpenAI
    except Exception as e:
        _openai_import_err = e
        _AsyncOpenAI = None

def _get_client():
    """Build a reusable AsyncOpenAI client with timeouts and retries."""
    global _client
    _ensure_openai_import()
    if _AsyncOpenAI is None:
        raise ImportError(
            "OpenAI SDK import failed. Install/upgrade with: pip install 'openai>=1.72.0'"
        ) from _openai_import_err
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    _client = _AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        timeout=60.0,     # total request timeout (seconds)
        max_retries=5,    # SDK-level retries with exponential backoff
    )
    return _client

def _is_gpt5(model: str) -> bool:
    return model.lower().startswith("gpt-5")

# ── JSON Schemas (your UI shapes) ─────────────────────────────────────────────
DIAGNOSTICS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "probability": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "probability", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

ADDITIONAL_EXAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "indications": {"type": "string"},
                    # New required field:
                    "priority": {"type": "string", "enum": ["alta", "media", "baja"]},
                },
                "required": ["name", "indications", "priority"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

PRESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "active_principle": {"type": "string"},
                    "dose": {"type": "number"},
                    "dose_unit": {"type": "string"},
                    "presentation": {"type": "string"},
                    "frequency": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "quantity_unit": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": [
                    "name","active_principle","dose","dose_unit",
                    "presentation","frequency","quantity","quantity_unit","notes"
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

COMPLEMENTARIES_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["name", "quantity", "notes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

# ── Core structured-output caller (Chat Completions) ──────────────────────────
async def _structured_call(
    section_name: str,
    schema: Dict[str, Any],
    system_prompt: str,
    user_context: Dict[str, Any],
    temperature: Optional[float] = None,  # ignored for GPT-5
) -> Dict[str, Any]:
    """
    Calls the model with a strict JSON Schema and returns parsed dict.

    - GPT-5: omit temperature/top_p; use reasoning_effort & verbosity (+ optional max_completion_tokens)
    - Non-GPT-5: allow temperature/top_p if provided
    - Raises RuntimeError with a human-readable cause on network/auth/HTTP issues
    """
    client = _get_client()

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": json.dumps(user_context, ensure_ascii=False)},
    ]

    kwargs: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"vet_{section_name}",
                "schema": schema,
                "strict": True,
            },
        },
    }

    if _is_gpt5(MODEL):
        # GPT-5: do not send temperature/top_p
        if REASONING_EFFORT:
            kwargs["reasoning_effort"] = REASONING_EFFORT
        if VERBOSITY:
            kwargs["verbosity"] = VERBOSITY
        if MAX_COMPLETION_TOKENS > 0:
            kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    else:
        # Non-GPT-5 models: standard sampling controls
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if TOP_P:
            kwargs["top_p"] = float(TOP_P)

    try:
        resp = await client.chat.completions.create(**kwargs)
    except NameError:
        # In case exception classes weren't imported for some reason:
        raise RuntimeError("OpenAI SDK is unavailable in this runtime")
    except AuthenticationError as e:  # 401/invalid key
        raise RuntimeError(
            "Authentication error with OpenAI: invalid or missing OPENAI_API_KEY. "
            "Verify the key is set in the vet-worker service environment."
        ) from e
    except (APITimeoutError, APIConnectionError) as e:
        # Network/DNS/connect/read timeout
        raise RuntimeError(f"Network/timeout while calling OpenAI: {e.__class__.__name__}: {e}") from e
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI rate limit exceeded: {e}") from e
    except BadRequestError as e:
        # 400 with helpful message (e.g., bad param)
        raise RuntimeError(f"Bad request to OpenAI: {e}") from e
    except APIError as e:
        # Generic non-2xx API error
        raise RuntimeError(f"OpenAI API error: {e}") from e
    except Exception as e:
        # Catch-all
        raise RuntimeError(f"Unexpected error calling OpenAI: {e}") from e

    msg = resp.choices[0].message
    # Some reasoning models may set `refusal` if content is refused
    refusal = getattr(msg, "refusal", None)
    if refusal:
        raise RuntimeError(f"Model refusal for {section_name}: {refusal}")

    try:
        return json.loads(msg.content)
    except Exception as e:
        logger.exception("Failed to parse JSON for %s: %s", section_name, e)
        raise
# common/llm_vet.py

async def _structured_call_cancelable_responses(
    *,
    section_name: str,
    schema: dict,
    system_prompt: str,
    user_context: dict,
    cancel_event: asyncio.Event,
    on_response_id: typing.Optional[typing.Callable[[str], typing.Awaitable[None]]] = None,
) -> dict:
    """
    Responses API (background mode) + strict JSON Schema.
    Optimistic cancel: we never block UX on network when cancelling.
    """
    client = _get_client()

    req = {
        "model": MODEL,
        "instructions": system_prompt.strip(),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(user_context, ensure_ascii=False)}
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": f"vet_{section_name}",
                "schema": schema,
                "strict": True,
            }
        },
        "background": True,
    }

    # GPT-5 knobs in Responses
    if _is_gpt5(MODEL) and REASONING_EFFORT:
        req["reasoning"] = {"effort": REASONING_EFFORT}
    if MAX_COMPLETION_TOKENS > 0:
        req["max_output_tokens"] = MAX_COMPLETION_TOKENS

    job = await client.responses.create(**req)
    rid = job.id

    try:
        if on_response_id:
            try:
                await on_response_id(rid)
            except Exception:
                logger.warning("on_response_id callback failed (rid=%s)", rid)

        # helper to cancel without blocking UX
        async def _bg_cancel():
            with contextlib.suppress(Exception):
                try:
                    await asyncio.wait_for(client.responses.cancel(rid), timeout=1.0)
                except Exception:
                    # swallow everything—this is best-effort
                    pass

        # Early cancel right after job creation
        if cancel_event.is_set():
            asyncio.create_task(_bg_cancel())
            raise LLMUserCancelled()

        # Poll until completion/failure or local cancel
        while True:
            if cancel_event.is_set():
                asyncio.create_task(_bg_cancel())
                raise LLMUserCancelled()

            state = await client.responses.retrieve(rid)
            status = getattr(state, "status", "") or ""

            if status == "completed":
                parsed = getattr(state, "output_parsed", None)
                if parsed is not None:
                    return parsed
                return json.loads(state.output_text)

            if status in ("failed", "cancelled", "rejected", "errored"):
                raise RuntimeError(f"OpenAI response did not complete (status={status})")

            await asyncio.sleep(0.25)

    except asyncio.CancelledError:
        # Worker runtime shutdown—cancel server-side best-effort but don't block
        asyncio.create_task(_bg_cancel())
        raise



async def gen_diagnostics_for_consultation_cancelable(
    consultation_id: int | str,
    *,
    cancel_event: asyncio.Event,
    on_response_id: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    session = await build_session_snapshot_from_consultation(consultation_id)
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: lista de diagnósticos diferenciales relevantes para pequeños animales.
Restricciones:
- Español neutral (LatAm).
- `probability` ∈ [0,1].
- `rationale`: 1–2 frases concisas basadas en el contexto.
- No inventes datos ni agregues campos extra.
- Si no hay datos suficientes, devuelve un item que indique "Información insuficiente".
"""
    user_context = {
        "task": "differential_diagnoses",
        "session_snapshot": session,
        "hints": [
            "Si hay signos GI (vómitos/diarrea, dolor abdominal, fiebre leve), considera GE aguda, pancreatitis, indiscreción alimentaria, etc."
        ],
    }
    return await _structured_call_cancelable_responses(
        section_name="diagnostics",
        schema=DIAGNOSTICS_SCHEMA,
        system_prompt=system_prompt,
        user_context=user_context,
        cancel_event=cancel_event,
        on_response_id=on_response_id,
    )


async def gen_additional_exams_for_consultation_cancelable(
    consultation_id: int | str,
    *,
    cancel_event: asyncio.Event,
    on_response_id: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    session = await build_session_snapshot_from_consultation(consultation_id)
    definitive = await fetch_definitive_diagnosis(consultation_id)
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: proponer exámenes complementarios razonables para el caso.
Cada item: `name`, `indications`, `priority` ∈ {"alta","media","baja"}.
Si el contexto es insuficiente y NO hay diagnóstico definitivo, devuelve exactamente un item "Información insuficiente".
"""
    user_context = {
        "task": "additional_exams",
        "session_snapshot": session,
        "previous": {"definitive_diagnosis": definitive} if definitive and any(definitive.values()) else {},
        "constraints": [
            "Máx. 3–5 ítems; prioriza utilidad diagnóstica inicial y costo.",
            "Asigna `priority` de forma consistente con la presentación clínica."
        ],
    }
    return await _structured_call_cancelable_responses(
        section_name="additional_exams",
        schema=ADDITIONAL_EXAMS_SCHEMA,
        system_prompt=system_prompt,
        user_context=user_context,
        cancel_event=cancel_event,
        on_response_id=on_response_id,
    )


async def gen_prescription_for_consultation_cancelable(
    consultation_id: int | str,
    *,
    cancel_event: asyncio.Event,
    on_response_id: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    session = await build_session_snapshot_from_consultation(consultation_id)
    definitive = await fetch_definitive_diagnosis(consultation_id)
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: plan terapéutico / medicación. Incluye TODOS los campos del esquema por item.
Si el contexto es insuficiente y NO hay diagnóstico definitivo, devuelve exactamente un item "Información insuficiente".
"""
    user_context = {
        "task": "therapeutic_plan",
        "session_snapshot": session,
        "previous": {"definitive_diagnosis": definitive} if definitive and any(definitive.values()) else {},
        "format_expectations": {"dose_unit": "mg/kg", "quantity_unit": "días"},
    }
    return await _structured_call_cancelable_responses(
        section_name="prescription",
        schema=PRESCRIPTION_SCHEMA,
        system_prompt=system_prompt,
        user_context=user_context,
        cancel_event=cancel_event,
        on_response_id=on_response_id,
    )


async def gen_complementary_treatments_for_consultation_cancelable(
    consultation_id: int | str,
    *,
    cancel_event: asyncio.Event,
    on_response_id: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    session = await build_session_snapshot_from_consultation(consultation_id)
    definitive = await fetch_definitive_diagnosis(consultation_id)
    meds = await fetch_prescribed_medications(consultation_id)
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: tratamientos complementarios. Cada item: `name`, `quantity`, `notes`.
Si el contexto es insuficiente y NO hay dx definitivo ni medicación, devuelve exactamente un item "Información insuficiente".
"""
    user_context = {
        "task": "complementary_treatments",
        "session_snapshot": session,
        "previous": {
            **({"definitive_diagnosis": definitive} if definitive and any(definitive.values()) else {}),
            **({"prescribed_medications": meds} if meds else {}),
        },
    }
    return await _structured_call_cancelable_responses(
        section_name="complementary_treatments",
        schema=COMPLEMENTARIES_SCHEMA,
        system_prompt=system_prompt,
        user_context=user_context,
        cancel_event=cancel_event,
        on_response_id=on_response_id,
    )


async def gen_diagnostics_for_consultation(consultation_id: int) -> Dict[str, Any]:
    session = await build_session_snapshot_from_consultation(consultation_id)
    return await gen_diagnostics_from_llm(session)
# ── Section-specific generators (Spanish) ─────────────────────────────────────
async def gen_diagnostics_from_llm(session: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: lista de diagnósticos diferenciales relevantes para pequeños animales.
Restricciones:
- Español neutral (LatAm).
- `probability` ∈ [0,1] (no es necesario que sumen 1).
- `rationale`: 1–2 frases concisas basadas en los signos del contexto.
- No inventes datos que no estén en el contexto ni agregues campos extra.
- sino hay datos suficientes pra hacer un diagnostico envia en el mismo formato de la respuesta contenido que indica que no hay diagnostico
"""
    user_context = {
        "task": "differential_diagnoses",
        "session_snapshot": session,
        "hints": [
            "Si hay signos GI (vómitos/diarrea, dolor abdominal, fiebre leve), considera GE aguda, pancreatitis, indiscreción alimentaria, etc."
        ],
    }
    return await _structured_call("diagnostics", DIAGNOSTICS_SCHEMA, system_prompt, user_context)



async def gen_additional_exams_for_consultation(consultation_id: int | str) -> Dict[str, Any]:
    session = await build_session_snapshot_from_consultation(consultation_id)
    definitive = await fetch_definitive_diagnosis(consultation_id)
    # Always call the LLM; abstention is handled in the prompt as a structured item.
    return await gen_additional_exams_from_llm(
        session,
        definitive_diagnosis=definitive if any((definitive or {}).values()) else None,
    )


async def gen_additional_exams_from_llm(
    session: Dict[str, Any],
    *,
    definitive_diagnosis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: proponer exámenes complementarios razonables para el caso.

Cada `item` **debe** incluir:
- `name` (p. ej., "Urianálisis")
- `indications` (justificación breve / protocolo)
- `priority` ∈ { "alta", "media", "baja" }

Usa TODA la información disponible:
- Contexto clínico (anamnesis, signos, constantes y examen físico).
- **Diagnóstico definitivo del/la veterinario/a** (si existe).

Reglas:
- No inventes datos fuera del contexto.
- Prioriza estudios que cambien conducta hoy o descarten urgencias.
- **Si el contexto es insuficiente y NO hay diagnóstico definitivo**, devuelve **exactamente un** item con:
  { "name": "Información insuficiente",
    "indications": "<explica brevemente qué falta>",
    "priority": "baja" }
No agregues campos fuera del esquema.
"""
    user_context = {
        "task": "additional_exams",
        "session_snapshot": session,
        "previous": {"definitive_diagnosis": definitive_diagnosis} if definitive_diagnosis else {},
        "constraints": [
            "Máx. 3–5 ítems; prioriza utilidad diagnóstica inicial y costo.",
            "Asigna `priority` de forma consistente con la presentación clínica."
        ],
    }
    return await _structured_call("additional_exams", ADDITIONAL_EXAMS_SCHEMA, system_prompt, user_context)




async def gen_prescription_for_consultation(consultation_id: int | str) -> Dict[str, Any]:
    session = await build_session_snapshot_from_consultation(consultation_id)
    definitive = await fetch_definitive_diagnosis(consultation_id)
    # Siempre invocamos al LLM; si falta contexto y no hay dx definitivo,
    # el propio modelo devuelve el item de "Información insuficiente".
    return await gen_prescription_from_llm(
        session,
        definitive_diagnosis=definitive if any((definitive or {}).values()) else None,
    )


async def gen_prescription_from_llm(
    session: Dict[str, Any],
    *,
    definitive_diagnosis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: plan terapéutico / medicación.

Para cada `item` incluye TODOS los campos del esquema:
- `name` (p. ej., "Metronidazol")
- `active_principle`
- `dose` (número, en mg/kg cuando aplique)
- `dose_unit` (p. ej., "mg/kg")
- `presentation` (p. ej., "Tabletas 250 mg")
- `frequency` (p. ej., "cada 12 h")
- `quantity` (entero)
- `quantity_unit` (p. ej., "días")
- `notes` (indicaciones/precauciones breves)

Usa TODA la información disponible:
- Contexto clínico (anamnesis, signos, constantes y examen físico).
- **Diagnóstico definitivo del/la veterinario/a** si está presente.

Reglas:
- No inventes datos fuera del contexto. Evita contraindicaciones obvias (especie/peso/estado).
- Ajusta dosis a rangos habituales; si falta un dato crítico (p. ej., peso), evita fármacos que lo requieran.
- **Si el contexto es insuficiente y NO hay diagnóstico definitivo**, devuelve **exactamente un** item con:
  {
    "name": "Información insuficiente",
    "active_principle": "N/A",
    "dose": 0,
    "dose_unit": "mg/kg",
    "presentation": "N/A",
    "frequency": "N/A",
    "quantity": 0,
    "quantity_unit": "días",
    "notes": "Contexto insuficiente para prescribir con seguridad; agrega signos, constantes y/o diagnóstico definitivo."
  }
No agregues campos fuera del esquema.
"""
    user_context = {
        "task": "therapeutic_plan",
        "session_snapshot": session,
        "previous": {"definitive_diagnosis": definitive_diagnosis} if definitive_diagnosis else {},
        "format_expectations": {"dose_unit": "mg/kg", "quantity_unit": "días"},
    }
    return await _structured_call("prescription", PRESCRIPTION_SCHEMA, system_prompt, user_context)




async def gen_complementary_treatments_for_consultation(consultation_id: int | str) -> Dict[str, Any]:
    session = await build_session_snapshot_from_consultation(consultation_id)
    definitive = await fetch_definitive_diagnosis(consultation_id)
    meds = await fetch_prescribed_medications(consultation_id)

    return await gen_complementary_treatments_from_llm(
        session,
        definitive_diagnosis=definitive if any((definitive or {}).values()) else None,
        prescribed_medications=meds if meds else None,
    )


async def gen_complementary_treatments_from_llm(
    session: Dict[str, Any],
    *,
    definitive_diagnosis: Optional[Dict[str, Any]] = None,
    prescribed_medications: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: tratamientos complementarios (p. ej., fluidoterapia, dieta, analgésicos no-farmacológicos, probióticos).

Cada `item`:
- `name` (intervención)
- `quantity` (rango/tiempo en texto)
- `notes` (instrucciones breves)

Usa TODA la información disponible:
- Contexto clínico (anamnesis, signos, constantes y examen físico).
- **Diagnóstico definitivo** del/la veterinario/a (si existe).
- **Medicaciones prescritas** en esta consulta (si existen) para evitar duplicidad o interacciones.

Reglas:
- No inventes datos fuera del contexto y mantén recomendaciones seguras para la especie.
- Prioriza medidas de soporte y cuidados en casa coherentes con el diagnóstico y con los fármacos indicados.
- **Abstención estructurada**: si el contexto es insuficiente y NO hay diagnóstico definitivo ni medicación prescrita,
  devuelve **exactamente un** item con:
  {
    "name": "Información insuficiente",
    "quantity": "N/A",
    "notes": "Contexto insuficiente para proponer tratamientos complementarios; agrega signos, constantes, diagnóstico definitivo y/o medicación."
  }
No agregues campos fuera del esquema.
"""
    user_context = {
        "task": "complementary_treatments",
        "session_snapshot": session,
        "previous": {
            **({"definitive_diagnosis": definitive_diagnosis} if definitive_diagnosis else {}),
            **({"prescribed_medications": prescribed_medications} if prescribed_medications else {}),
        },
    }
    return await _structured_call("complementary_treatments", COMPLEMENTARIES_SCHEMA, system_prompt, user_context)
