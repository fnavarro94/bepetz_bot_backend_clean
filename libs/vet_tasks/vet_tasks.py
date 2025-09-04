# ==============================================================================
# File: tasks/vet_tasks.py  –  Vet workflow worker (independent from chat worker)
# Purpose:
#   • Exposes 4 Taskiq tasks:
#       1) run_diagnostics_task(session_id)
#       2) run_additional_exams_task(session_id)
#       3) run_prescription_task(session_id)
#       4) run_complementary_treatments_task(session_id)
#   • DOES NOT read session input data from Firestore (uses dummy JSON).
#   • Persists outputs to Firestore; reruns overwrite previous outputs.
#   • Optional Redis status fanout on channel: f"vet:{session_id}"
# ==============================================================================

import os
import json
import logging
from datetime import datetime, timezone

import redis.asyncio as aioredis
from google.cloud import firestore

from common.broker import vet_broker
from common.llm_vet import (
    
    gen_diagnostics_from_llm,
    gen_additional_exams_from_llm,
    gen_prescription_from_llm,
    gen_complementary_treatments_from_llm,)

from common.llm_vet import (
    gen_diagnostics_for_consultation, 
    gen_additional_exams_for_consultation, 
    gen_prescription_for_consultation,
    gen_complementary_treatments_for_consultation
)


# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("vet-worker")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][vet-worker] %(message)s"))
logger.handlers = [_handler]
logger.propagate = False

# ── Firestore (outputs only) ──────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB   = os.getenv("CHATS_FIRESTORE_DB")
db = firestore.AsyncClient(project=GCP_PROJECT_ID, database=FIRESTORE_DB)

# ── Redis (optional status streaming) ─────────────────────────────────────────
STREAM_REDIS_HOST = os.getenv("STREAM_REDIS_HOST", "")
STREAM_REDIS_PORT = int(os.getenv("STREAM_REDIS_PORT", "6379"))
STREAM_REDIS_SSL  = os.getenv("STREAM_REDIS_SSL", "false").lower() == "true"

redis_stream: aioredis.Redis | None = None
if STREAM_REDIS_HOST:
    redis_stream = aioredis.Redis(
        host=STREAM_REDIS_HOST,
        port=STREAM_REDIS_PORT,
        ssl=STREAM_REDIS_SSL,
        encoding="utf-8",
        decode_responses=True,
    )
    logger.info("Redis client ready for vet-worker (host=%s port=%s ssl=%s)",
                STREAM_REDIS_HOST, STREAM_REDIS_PORT, STREAM_REDIS_SSL)
else:
    logger.info("STREAM_REDIS_HOST not set → vet-worker Redis streaming disabled")

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

import signal, asyncio, logging
logger = logging.getLogger("vet-worker")

SHUTTING_DOWN = False
def _install_signal_logging():
    def _on(sig):
        global SHUTTING_DOWN
        SHUTTING_DOWN = True
        logger.error("Received signal %s → beginning graceful shutdown", sig)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, lambda s=sig: _on(s))
        except NotImplementedError:
            # Fallback (Windows / limited envs)
            signal.signal(sig, lambda *_: _on(sig))

_install_signal_logging()

def _vet_channel(session_id: str) -> str:
    return f"vet:{session_id}"

async def _publish_status(session_id: str, event: str, **fields) -> None:
    if not redis_stream:
        return
    payload = {
        "event": event,                # e.g., "status", "done", "error"
        "data": {**fields},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await redis_stream.publish(_vet_channel(session_id), json.dumps(payload))
    except Exception as e:
        logger.warning("Redis publish failed (session=%s, event=%s): %s", session_id, event, e)

# --- Dummy input data ----------------------------------------------------------
# Priority:
# 1) If VET_DUMMY_SESSIONS_JSON is set (JSON map of session_id->object), use it.
# 2) Else, generate a deterministic stub from session_id.
#
# Example env:
# export VET_DUMMY_SESSIONS_JSON='{
#   "sess-123": {"patient":{"species":"canine","breed":"Lab","age":6,"sex":"M","weight_kg":28},
#                "complaints":["vomiting","lethargy"], "vitals":{"temp_c":39.6,"hr_bpm":120,"rr_bpm":28},
#                "exam":{"findings":"abdominal discomfort","dehydration":2}, "history":{"diet":"kibble"} }
# }'

def _env_dummy_map() -> dict:
    try:
        data = {
                    "sess-gi": {
                        "patient": {"species":"canine","breed":"Beagle","age":5,"sex":"F","weight_kg":11.4},
                        "complaints": ["Vomiting","Diarrhea"],
                        "vitals": {"temp_c":39.7,"hr_bpm":118,"rr_bpm":30,"bcs":5,"mucous_membranes":"pink"},
                        "exam": {"findings":"Abdominal discomfort","dehydration":2},
                        "history": {"diet":"kibble","vaccines":"up-to-date"}
                    }
                }
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("Invalid VET_DUMMY_SESSIONS_JSON: %s", e)
    return {}

_ENV_DUMMY = _env_dummy_map()

def _dummy_session_by_id(session_id: str) -> dict:
    """Return a stable dummy payload for the given session id."""
    # 1) Env override
    if session_id in _ENV_DUMMY:
        return _ENV_DUMMY[session_id]

    # 2) Deterministic presets based on suffix to vary scenarios
    sid = session_id.lower()
    if sid.endswith("gi") or "gi" in sid:
        return {
            "patient": {"species": "canine", "breed": "Mixed", "age": 4, "sex": "F", "weight_kg": 18.2},
            "complaints": ["Vomiting", "Diarrhea"],
            "vitals": {"temp_c": 39.5, "hr_bpm": 110, "rr_bpm": 28, "bcs": 5, "mucous_membranes": "pink"},
            "exam": {"findings": "Mild abdominal discomfort on palpation", "pain_score": 2, "dehydration": 2, "notes": ""},
            "history": {"diet": "kibble + treats", "vaccines": "up-to-date", "deworming": "unknown", "meds": None, "allergies": None, "chronic": None},
            "labs": {},
        }
    if sid.endswith("fever") or "fever" in sid:
        return {
            "patient": {"species": "feline", "breed": "DSH", "age": 3, "sex": "M", "weight_kg": 4.5},
            "complaints": ["Lethargy", "Inappetence"],
            "vitals": {"temp_c": 40.1, "hr_bpm": 180, "rr_bpm": 40, "bcs": 4, "mucous_membranes": "pink"},
            "exam": {"findings": "Generalized discomfort", "pain_score": 1, "dehydration": 1, "notes": ""},
            "history": {"diet": "canned", "vaccines": "partial", "deworming": "recent", "meds": None, "allergies": None, "chronic": None},
            "labs": {},
        }
    # Default generic case
    return {
        "patient": {"species": "canine", "breed": "Labrador", "age": 6, "sex": "M", "weight_kg": 28.0},
        "complaints": ["Routine check", "Mild lethargy"],
        "vitals": {"temp_c": 38.6, "hr_bpm": 90, "rr_bpm": 22, "bcs": 6, "mucous_membranes": "pink"},
        "exam": {"findings": "No acute distress", "pain_score": 0, "dehydration": 0, "notes": ""},
        "history": {"diet": "kibble", "vaccines": "up-to-date", "deworming": "regular", "meds": None, "allergies": None, "chronic": None},
        "labs": {},
    }


def _build_dummy_medications(session: dict) -> list[dict]:
    """Return a static list of medications with the fields your UI expects."""
    return [
        {
            "name": "Maropitant",
            "active_principle": "Maropitant citrate",
            "dose": 1.0,                 # numeric dose
            "dose_unit": "mg/kg",        # unit shown next to 'Dosis'
            "presentation": "Solución inyectable 10 mg/ml",
            "frequency": "cada 24 h",
            "quantity": 3,               # number of days/packs/etc.
            "quantity_unit": "días",
            "notes": "Antiemético. Administrar con alimento para reducir náuseas."
        },
        {
            "name": "Metronidazol",
            "active_principle": "Metronidazol",
            "dose": 10.0,
            "dose_unit": "mg/kg",
            "presentation": "Tabletas 250 mg",
            "frequency": "cada 12 h",
            "quantity": 5,
            "quantity_unit": "días",
            "notes": "Si hay diarrea persistente. Suspender si aparece ataxia."
        },
        {
            "name": "Omeprazol",
            "active_principle": "Omeprazol",
            "dose": 1.0,
            "dose_unit": "mg/kg",
            "presentation": "Cápsulas 10 mg",
            "frequency": "cada 24 h",
            "quantity": 7,
            "quantity_unit": "días",
            "notes": "Proteger mucosa gástrica; dar por la mañana en ayunas."
        },
    ]


def _build_dummy_complementaries(session: dict) -> list[dict]:
    """Return a static list of complementary treatments."""
    return [
        {
            "name": "Fluidoterapia",
            "quantity": "20–40 ml/kg/día",
            "notes": "Cristaloides balanceados, ajustar según deshidratación."
        },
        {
            "name": "Dieta blanda",
            "quantity": "3–5 días",
            "notes": "Porciones pequeñas y frecuentes; arroz y pollo hervido."
        },
        {
            "name": "Pro/Prebióticos",
            "quantity": "según rótulo",
            "notes": "Iniciar tras 24–48 h de mejoría digestiva."
        },
    ]


def _build_dummy_differentials(session: dict) -> list[dict]:
    """Return ONLY the list of differentials the UI needs."""
    rationale = (
        "Porque presenta vómitos y diarrea postprandiales, poco apetito, "
        "decaimiento, leve deshidratación y sensibilidad abdominal."
    )
    return [
        {"name": "Gastroenteritis aguda",   "probability": 0.67, "rationale": rationale},
        {"name": "Pancreatitis aguda",     "probability": 0.35, "rationale": rationale},
        {"name": "Indiscreción alimentaria","probability": 0.24, "rationale": rationale},
    ]

async def _fetch_vet_session(session_id: str) -> dict:
    """
    Dummy fetch: returns generated/ENV-provided JSON.
    (Later you'll replace this with a Postgres read.)
    """
    return _dummy_session_by_id(session_id)

# --- Output persistence --------------------------------------------------------
async def _persist_output(session_id: str, kind: str, data: dict) -> None:
    """
    Persist under: vet_sessions/{session_id}/outputs/{kind}.
    Reruns **overwrite** previous data (merge=False).
    """
    out_ref = (
        db.collection("vet_sessions")
          .document(session_id)
          .collection("outputs")
          .document(kind)
    )
    await out_ref.set(
        {
            "result": data,
            "kind": kind,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
        merge=False,  # ← overwrite semantics
    )

# ------------------------------------------------------------------------------
# Diagnostics agent (placeholder logic)
# ------------------------------------------------------------------------------

def _draft_diagnostics_from_session(session: dict) -> dict:
    patient = session.get("patient", {})
    complaints = session.get("complaints", [])
    vitals = session.get("vitals", {})
    exam = session.get("exam", {})
    history = session.get("history", {})
    labs = session.get("labs", {})

    flags = []
    if vitals.get("temp_c") and vitals["temp_c"] > 39.2:
        flags.append("fever")
    if exam.get("dehydration", 0) >= 2:
        flags.append("possible_dehydration")
    if any("vomiting" == str(c).lower() for c in complaints):
        flags.append("gastrointestinal")

    primary = "Undetermined"
    ddx = []
    if "fever" in flags and "gastrointestinal" in flags:
        primary = "Acute gastroenteritis (tentative)"
        ddx = ["Dietary indiscretion", "Infectious enteritis", "Pancreatitis", "Foreign body"]
    elif "fever" in flags:
        primary = "Systemic inflammatory process (tentative)"
        ddx = ["Viral/bacterial infection", "Tick-borne disease", "Immune-mediated disease"]
    elif "gastrointestinal" in flags:
        primary = "Gastrointestinal upset (tentative)"
        ddx = ["Dietary indiscretion", "Parasitism", "Gastritis"]

    return {
        "summary": {
            "primary_tentative_diagnosis": primary,
            "key_flags": flags,
        },
        "differential_diagnoses": ddx,
        "supporting_evidence": {
            "complaints": complaints,
            "vitals": vitals,
            "exam_findings": exam.get("findings"),
            "history_notes": history,
            "labs_snapshot": labs,
        },
        "confidence": "low",
        "recommendations_next": [
            "Correlate with full physical exam and lab results.",
            "Consider targeted labs/imaging if signs persist or worsen.",
        ],
    }




#-------------------------------------------------------------------------------
# Cancell Helpers
#-------------------------------------------------------------------------------
# --- add near other imports ---
import asyncio, json

# --- helper: create a cancel Event wired to Redis Pub/Sub ---
async def _make_cancel_event(session_id: str, kind: str) -> asyncio.Event:
    ev = asyncio.Event()
    if not redis_stream:
        print(f"no redis stream found so sending cancel event")
        return ev
    print(f"redis_stream was found so running code")
    async def listener():
        pubsub = redis_stream.pubsub()
        await pubsub.subscribe(f"vet:{session_id}:control")
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                data = msg.get("data")
                try:
                    obj = json.loads(data) if isinstance(data, str) else None
                except Exception:
                    obj = None
                if isinstance(obj, dict) and obj.get("event") == "cancel":
                    k = (obj.get("data") or {}).get("kind")
                    if k in (kind, "*"):
                        ev.set()
                        break
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe(f"vet:{session_id}:control")
                await pubsub.close()
            except Exception:
                pass

    asyncio.create_task(listener())
    return ev

# --- helper: run a long awaitable unless cancel_event fires ---
import traceback
class UserCancelled(Exception):
    pass
from contextlib import suppress
import asyncio

async def _wait_for_cancel_or(coro, cancel_event: asyncio.Event, label: str = ""):
    t_main = asyncio.create_task(coro, name=f"main:{label}")
    t_cancel = asyncio.create_task(cancel_event.wait(), name=f"cancel:{label}")
    try:
        done, _ = await asyncio.wait({t_main, t_cancel}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # External runtime cancellation (e.g., worker shutdown)
        logger.error("Outer/runtime cancellation while waiting (label=%s). "
                     "t_main.done=%s t_main.cancelled=%s",
                     label, t_main.done(), t_main.cancelled(), exc_info=True)
        t_main.cancel()
        with suppress(asyncio.CancelledError):
            await t_main
        raise

    # User-triggered cancel path
    if t_cancel in done and cancel_event.is_set():
        t_main.cancel()
        with suppress(asyncio.CancelledError):
            await t_main
        raise UserCancelled()

    # Normal path: main finished first; cancel the cancel-waiter cleanly
    t_cancel.cancel()
    with suppress(asyncio.CancelledError):
        await t_cancel
    return await t_main
# --- helper: common cancelled pathway ---
async def _handle_cancelled(session_id: str, kind: str):
    await _publish_status(session_id, "status", phase=f"{kind}-cancelled")
    try:
        await _persist_output(session_id, kind, {"status": "cancelled"})
    except Exception:
        logger.warning("Persist cancel marker failed (session=%s kind=%s)", session_id, kind)



# ------------------------------------------------------------------------------
# Taskiq tasks (queueable from API)
# ------------------------------------------------------------------------------

from typing import Optional, List

@vet_broker.task
async def run_diagnostics_task(session_id: str) -> None:
    kind = "diagnostics"
    await _publish_status(session_id, "status", phase="diagnostics-started")
    cancel_event = await _make_cancel_event(session_id, kind)

    try:
        # Don’t force int(session_id) – allow both:
        consultation_id: Optional[int | str]
        try:
            consultation_id = int(session_id)
        except ValueError:
            consultation_id = session_id

        payload = await _wait_for_cancel_or(
            gen_diagnostics_for_consultation(consultation_id),
            cancel_event,
            label=f"{kind}:{session_id}",
        )

        await _publish_status(session_id, "diagnostics", **payload)
        await _persist_output(session_id, "diagnostics", payload)
        await _publish_status(session_id, "status", phase="diagnostics-finished")
        await _publish_status(session_id, "done", kind="diagnostics")

    except UserCancelled:
        await _handle_cancelled(session_id, kind)

    except asyncio.CancelledError as e:
        # Runtime cancellation — log with stack + shutdown bit
        logger.exception("Runtime cancelled task (user_cancel=%s, shutting_down=%s)",
                         cancel_event.is_set(), SHUTTING_DOWN)
        await _publish_status(
            session_id, "error",
            message=f"Runtime cancelled task (shutting_down={SHUTTING_DOWN})"
        )
        # Optional: re-raise to let the broker see it
        # raise

    except Exception as e:
        logger.exception("Diagnostics failed")
        items = _build_dummy_differentials({})
        await _publish_status(session_id, "diagnostics", items=items)
        await _persist_output(session_id, "diagnostics", {"items": items})
        await _publish_status(session_id, "error", message=str(e))

@vet_broker.task
async def run_additional_exams_task(session_id: str) -> None:
    """Queueable task: propose additional exams using DB context + vet's definitive diagnosis; persist result."""
    await _publish_status(session_id, "status", phase="additional-exams-started")
    try:
        consultation_id = session_id  # string is fine; SQL uses ::bigint casts

        # Call LLM with full context (always). Prompt handles "Información insuficiente" when needed.
        try:
            # requires: from common.llm_vet import gen_additional_exams_for_consultation
            payload = await gen_additional_exams_for_consultation(consultation_id)
        except Exception as e:
            logger.warning("LLM additional_exams failed; falling back to stub. reason=%s", e)
            payload = {
                "items": [
                    {
                        "name": "Información insuficiente",
                        "indications": "Fallo del servicio LLM o de red; respuesta de respaldo.",
                        "priority": "baja",
                    }
                ]
            }

        # Stream to relay as a named SSE event
        await _publish_status(session_id, "additional-exams", **payload)

        # Persist the same JSON for later fetch
        await _persist_output(session_id, "additional_exams", payload)

        await _publish_status(session_id, "status", phase="additional-exams-finished")
        await _publish_status(session_id, "done", kind="additional-exams")
        logger.info("Additional exams completed (session=%s)", session_id)

    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Additional exams failed (session=%s): %s", session_id, e)


@vet_broker.task
async def run_prescription_task(session_id: str) -> None:
    """Queueable task: emit medications list for the 'Plan terapéutico / Medicación' UI."""
    await _publish_status(session_id, "status", phase="prescription-started")
    try:
        consultation_id = session_id  # string OK; SQL casts use ::bigint in queries

        try:
            # requires: from common.llm_vet import gen_prescription_for_consultation
            payload = await gen_prescription_for_consultation(consultation_id)
        except Exception as e:
            logger.warning("LLM prescription failed; falling back to stub. reason=%s", e)
            payload = {
                "items": [
                    {
                        "name": "Información insuficiente",
                        "active_principle": "N/A",
                        "dose": 0,
                        "dose_unit": "mg/kg",
                        "presentation": "N/A",
                        "frequency": "N/A",
                        "quantity": 0,
                        "quantity_unit": "días",
                        "notes": "Fallo del servicio LLM o de red; respuesta de respaldo.",
                    }
                ]
            }

        # Stream to SSE that this panel listens to
        await _publish_status(session_id, "prescription", **payload)

        # Persist
        await _persist_output(session_id, "prescription", payload)

        await _publish_status(session_id, "status", phase="prescription-finished")
        await _publish_status(session_id, "done", kind="prescription")
        logger.info("Prescription completed (session=%s)", session_id)
    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Prescription failed (session=%s): %s", session_id, e)



@vet_broker.task
async def run_complementary_treatments_task(session_id: str) -> None:
    """Queueable task: emit complementary treatments list for the UI."""
    await _publish_status(session_id, "status", phase="complementary-treatments-started")
    try:
        consultation_id = session_id  # string OK; SQL uses ::bigint casts

        try:
            payload = await gen_complementary_treatments_for_consultation(consultation_id)
        except Exception as e:
            logger.warning("LLM complementary_treatments failed; falling back to stub. reason=%s", e)
            payload = {
                "items": [
                    {
                        "name": "Información insuficiente",
                        "quantity": "N/A",
                        "notes": "Fallo del servicio LLM o de red; respuesta de respaldo.",
                    }
                ]
            }

        # Named event must be 'complementary-treatments'
        await _publish_status(session_id, "complementary-treatments", **payload)

        # Persist
        await _persist_output(session_id, "complementary_treatments", payload)

        await _publish_status(session_id, "status", phase="complementary-treatments-finished")
        await _publish_status(session_id, "done", kind="complementary-treatments")
        logger.info("Complementary treatments completed (session=%s)", session_id)
    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Complementary treatments failed (session=%s): %s", session_id, e)