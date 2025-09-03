# common/llm_vet.py
import os
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("vet-llm")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = os.getenv("VET_LLM_MODEL", "gpt-5-mini")

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
"""
    user_context = {
        "task": "differential_diagnoses",
        "session_snapshot": session,
        "hints": [
            "Si hay signos GI (vómitos/diarrea, dolor abdominal, fiebre leve), considera GE aguda, pancreatitis, indiscreción alimentaria, etc."
        ],
    }
    return await _structured_call("diagnostics", DIAGNOSTICS_SCHEMA, system_prompt, user_context)

async def gen_additional_exams_from_llm(session: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: proponer exámenes complementarios razonables para el caso.
Cada `item` **debe** incluir:
- `name`: nombre del examen (p. ej., "Urianálisis").
- `indications`: breve justificación o protocolo.
- `priority`: **una** etiqueta de triage ∈ { "alta", "media", "baja" }:
    - "alta": cambia conductas inmediatas o descarta urgencias (obstrucción, sepsis, cuerpo extraño).
    - "media": orienta diagnóstico en el mismo día, pero no crítico inmediato.
    - "baja": diferible o de seguimiento si la evolución es favorable.
No agregues campos fuera del esquema.
"""
    user_context = {
        "task": "additional_exams",
        "session_snapshot": session,
        "constraints": [
            "Máx. 3–5 ítems; prioriza utilidad diagnóstica inicial y costo.",
            "Asigna `priority` de forma consistente con la presentación clínica."
        ],
    }
    return await _structured_call("additional_exams", ADDITIONAL_EXAMS_SCHEMA, system_prompt, user_context)

async def gen_prescription_from_llm(session: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: plan terapéutico / medicación.
Para cada `item` incluye TODOS los campos del esquema con dosis en mg/kg cuando aplique.
Notas:
- Usa fármacos comunes para pequeños animales.
- Ajusta dosis a rangos habituales y evita contraindicaciones evidentes por el contexto.
- Español; sin campos extra.
"""
    user_context = {
        "task": "therapeutic_plan",
        "session_snapshot": session,
        "format_expectations": {"dose_unit": "mg/kg", "quantity_unit": "días"},
    }
    return await _structured_call("prescription", PRESCRIPTION_SCHEMA, system_prompt, user_context)

async def gen_complementary_treatments_from_llm(session: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = """
Eres un asistente clínico veterinario. Devuelve **solo** el JSON pedido.
Objetivo: tratamientos complementarios (fluidoterapia, dieta, probióticos, etc.).
Cada `item`:
- `name` (intervención)
- `quantity` (rango/tiempo en texto)
- `notes` (instrucciones breves)
Sin campos extra.
"""
    user_context = {"task": "complementary_treatments", "session_snapshot": session}
    return await _structured_call("complementary_treatments", COMPLEMENTARIES_SCHEMA, system_prompt, user_context)
