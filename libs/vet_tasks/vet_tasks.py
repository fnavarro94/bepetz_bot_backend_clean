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

from common.llm_vet import (
    LLMUserCancelled,
    gen_diagnostics_for_consultation_cancelable,
    gen_additional_exams_for_consultation_cancelable,
    gen_prescription_for_consultation_cancelable,
    gen_complementary_treatments_for_consultation_cancelable,
)
import contextlib 

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

# --- helper: create a cancel Event wired to Redis Pub/Sub (durable+ephemeral) ---
async def _make_cancel_event(session_id: str, kind: str) -> asyncio.Event:
    ev = asyncio.Event()
    if not redis_stream:
        return ev

    channel  = f"vet:{session_id}:control"
    flag_key = f"vet:{session_id}:{kind}:cancelled"

    pubsub = redis_stream.pubsub()
    await pubsub.subscribe(channel)  # 1) subscribe first (catch future publishes)

    # 2) then check the durable sticky flag (catch past cancels)
    try:
        if await redis_stream.get(flag_key):
            ev.set()
            # We won't need the listener; clean up right away
            with suppress(Exception):
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            return ev
    except Exception:
        # Non-fatal; fall through to live listening
        pass

    async def listener():
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                try:
                    data = json.loads(msg["data"])
                except Exception:
                    continue
                if data.get("event") == "cancel":
                    k = (data.get("data") or {}).get("kind")
                    # Keep it strict (no '*' wildcard unless you truly need it)
                    if k == kind:
                        ev.set()
                        break
        finally:
            with suppress(Exception):
                await pubsub.unsubscribe(channel)
                await pubsub.close()

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


# --- add this tiny helper (or inline the mapping if you prefer) ---
def _hyphen_kind(kind: str) -> str:
    if kind == "additional_exams":
        return "additional-exams"
    if kind == "complementary_treatments":
        return "complementary-treatments"
    return kind  # diagnostics/prescription already hyphen-free


# --- helper: common cancelled pathway ---
# tasks/vet_tasks.py
# --- UPDATED: common cancelled pathway ----------------------------------------
async def _handle_cancelled(session_id: str, kind: str):
    ek = _hyphen_kind(kind)  # diagnostics / additional-exams / prescription / complementary-treatments
    await _publish_status(session_id, "status", phase=f"{ek}-cancelled")
    await _publish_status(session_id, "cancelled", kind=ek)
    await _publish_status(session_id, "done", kind=ek, cancelled=True)

    async def _bg_persist_and_clear():
        with suppress(Exception):
            # NEW: durable run-state and output marker
            await _persist_run_state(
                session_id, kind,
                status="cancelled",
                phase=f"{ek}-cancelled",
                finished_at=firestore.SERVER_TIMESTAMP,
            )
            await _persist_output(session_id, kind, {"status": "cancelled"})
        if redis_stream:
            with suppress(Exception):
                await redis_stream.delete(f"vet:{session_id}:{kind}:cancelled")
    asyncio.create_task(_bg_persist_and_clear())



# --- NEW: persist run-state ----------------------------------------------------
async def _persist_run_state(session_id: str, kind: str, **fields) -> None:
    """
    Durable per-step state under:
      vet_sessions/{session_id}/runs/{kind}
    Merge-write with an automatic updated_at.
    """
    ref = (
        db.collection("vet_sessions")
          .document(session_id)
          .collection("runs")
          .document(kind)
    )
    payload = {**fields, "updated_at": firestore.SERVER_TIMESTAMP}
    await ref.set(payload, merge=True)


# ------------------------------------------------------------------------------
# Taskiq tasks (queueable from API)
# ------------------------------------------------------------------------------

from typing import Optional, List

@vet_broker.task
async def run_diagnostics_task(session_id: str) -> None:
    kind = "diagnostics"
    await _publish_status(session_id, "status", phase="diagnostics-started")

    # NEW: durable state → running
    await _persist_run_state(session_id, kind, status="running", phase="diagnostics-started")

    cancel_event = await _make_cancel_event(session_id, kind)

    try:
        # Accept both int and str for consultation_id
        try:
            consultation_id = int(session_id)
        except ValueError:
            consultation_id = session_id

        payload = await gen_diagnostics_for_consultation_cancelable(
            consultation_id,
            cancel_event=cancel_event,
        )

        await _publish_status(session_id, "diagnostics", **payload)
        await _persist_output(session_id, "diagnostics", payload)

        # NEW: durable state → done
        await _persist_run_state(
            session_id, kind,
            status="done",
            phase="diagnostics-finished",
            finished_at=firestore.SERVER_TIMESTAMP,
        )

        await _publish_status(session_id, "status", phase="diagnostics-finished")
        await _publish_status(session_id, "done", kind="diagnostics")

    except LLMUserCancelled:
        await _handle_cancelled(session_id, kind)

    except asyncio.CancelledError:
        logger.exception("Runtime cancelled task (user_cancel=%s, shutting_down=%s)",
                         cancel_event.is_set(), SHUTTING_DOWN)
        if cancel_event.is_set():
            await _handle_cancelled(session_id, kind)
        else:
            ek = _hyphen_kind(kind)
            await _publish_status(
                session_id, "error",
                message=f"Runtime cancelled task (shutting_down={SHUTTING_DOWN})"
            )
            # NEW: durable state → error
            await _persist_run_state(
                session_id, kind,
                status="error",
                phase=f"{ek}-runtime-cancel",
                error_message="Runtime cancelled task",
                finished_at=firestore.SERVER_TIMESTAMP,
            )
            await _publish_status(session_id, "done", kind=ek, error=True)
        return

    except Exception as e:
        logger.exception("Diagnostics failed")
        items = _build_dummy_differentials({})
        await _publish_status(session_id, "diagnostics", items=items)
        await _persist_output(session_id, "diagnostics", {"items": items})
        await _publish_status(session_id, "error", message=str(e))

        # NEW: durable state → error
        await _persist_run_state(
            session_id, kind,
            status="error",
            phase="diagnostics-error",
            error_message=str(e),
            finished_at=firestore.SERVER_TIMESTAMP,
        )

        await _publish_status(session_id, "done", kind=_hyphen_kind(kind), error=True)


@vet_broker.task
async def run_additional_exams_task(session_id: str) -> None:
    """Queueable task: propose additional exams using DB context + vet's definitive diagnosis; persist result."""
    kind = "additional_exams"
    await _publish_status(session_id, "status", phase="additional-exams-started")

    # NEW: durable state → running
    await _persist_run_state(session_id, kind, status="running", phase="additional-exams-started")

    cancel_event = await _make_cancel_event(session_id, kind)

    try:
        payload = await gen_additional_exams_for_consultation_cancelable(
            consultation_id=session_id,   # string OK
            cancel_event=cancel_event,
        )

        await _publish_status(session_id, "additional-exams", **payload)
        await _persist_output(session_id, "additional_exams", payload)

        # NEW: durable state → done
        await _persist_run_state(
            session_id, kind,
            status="done",
            phase="additional-exams-finished",
            finished_at=firestore.SERVER_TIMESTAMP,
        )

        await _publish_status(session_id, "status", phase="additional-exams-finished")
        await _publish_status(session_id, "done", kind="additional-exams")
        logger.info("Additional exams completed (session=%s)", session_id)

    except LLMUserCancelled:
        await _handle_cancelled(session_id, kind)

    except asyncio.CancelledError:
        logger.exception("Runtime cancelled task (user_cancel=%s, shutting_down=%s)",
                         cancel_event.is_set(), SHUTTING_DOWN)
        if cancel_event.is_set():
            await _handle_cancelled(session_id, kind)
        else:
            ek = _hyphen_kind(kind)
            await _publish_status(
                session_id, "error",
                message=f"Runtime cancelled task (shutting_down={SHUTTING_DOWN})"
            )
            # NEW: durable state → error
            await _persist_run_state(
                session_id, kind,
                status="error",
                phase=f"{ek}-runtime-cancel",
                error_message="Runtime cancelled task",
                finished_at=firestore.SERVER_TIMESTAMP,
            )
            await _publish_status(session_id, "done", kind=ek, error=True)
        return

    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Additional exams failed (session=%s): %s", session_id, e)

        # Fallback stub
        payload = {
            "items": [
                {
                    "name": "Información insuficiente",
                    "indications": "Fallo del servicio LLM o de red; respuesta de respaldo.",
                    "priority": "baja",
                }
            ]
        }
        await _publish_status(session_id, "additional-exams", **payload)
        await _persist_output(session_id, "additional_exams", payload)

        # NEW: durable state → error
        await _persist_run_state(
            session_id, kind,
            status="error",
            phase="additional-exams-error",
            error_message=str(e),
            finished_at=firestore.SERVER_TIMESTAMP,
        )

        await _publish_status(session_id, "done", kind=_hyphen_kind(kind), error=True)

@vet_broker.task
async def run_prescription_task(session_id: str) -> None:
    """Queueable task: emit medications list for the 'Plan terapéutico / Medicación' UI."""
    kind = "prescription"
    await _publish_status(session_id, "status", phase="prescription-started")

    # NEW: durable state → running
    await _persist_run_state(session_id, kind, status="running", phase="prescription-started")

    cancel_event = await _make_cancel_event(session_id, kind)

    try:
        payload = await gen_prescription_for_consultation_cancelable(
            consultation_id=session_id,   # string OK
            cancel_event=cancel_event,
        )

        await _publish_status(session_id, "prescription", **payload)
        await _persist_output(session_id, "prescription", payload)

        # NEW: durable state → done
        await _persist_run_state(
            session_id, kind,
            status="done",
            phase="prescription-finished",
            finished_at=firestore.SERVER_TIMESTAMP,
        )

        await _publish_status(session_id, "status", phase="prescription-finished")
        await _publish_status(session_id, "done", kind="prescription")
        logger.info("Prescription completed (session=%s)", session_id)

    except LLMUserCancelled:
        await _handle_cancelled(session_id, kind)

    except asyncio.CancelledError:
        logger.exception("Runtime cancelled task (user_cancel=%s, shutting_down=%s)",
                         cancel_event.is_set(), SHUTTING_DOWN)
        if cancel_event.is_set():
            await _handle_cancelled(session_id, kind)
        else:
            ek = _hyphen_kind(kind)
            await _publish_status(
                session_id, "error",
                message=f"Runtime cancelled task (shutting_down={SHUTTING_DOWN})"
            )
            # NEW: durable state → error
            await _persist_run_state(
                session_id, kind,
                status="error",
                phase=f"{ek}-runtime-cancel",
                error_message="Runtime cancelled task",
                finished_at=firestore.SERVER_TIMESTAMP,
            )
            await _publish_status(session_id, "done", kind=ek, error=True)
        return

    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Prescription failed (session=%s): %s", session_id, e)

        # Fallback stub
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
        await _publish_status(session_id, "prescription", **payload)
        await _persist_output(session_id, "prescription", payload)

        # NEW: durable state → error
        await _persist_run_state(
            session_id, kind,
            status="error",
            phase="prescription-error",
            error_message=str(e),
            finished_at=firestore.SERVER_TIMESTAMP,
        )

        await _publish_status(session_id, "done", kind=_hyphen_kind(kind), error=True)

@vet_broker.task
async def run_complementary_treatments_task(session_id: str) -> None:
    """Queueable task: emit complementary treatments list for the UI."""
    kind = "complementary_treatments"
    await _publish_status(session_id, "status", phase="complementary-treatments-started")

    # NEW: durable state → running
    await _persist_run_state(session_id, kind, status="running", phase="complementary-treatments-started")

    cancel_event = await _make_cancel_event(session_id, kind)

    try:
        payload = await gen_complementary_treatments_for_consultation_cancelable(
            consultation_id=session_id,   # string OK
            cancel_event=cancel_event,
        )

        await _publish_status(session_id, "complementary-treatments", **payload)
        await _persist_output(session_id, "complementary_treatments", payload)

        # NEW: durable state → done
        await _persist_run_state(
            session_id, kind,
            status="done",
            phase="complementary-treatments-finished",
            finished_at=firestore.SERVER_TIMESTAMP,
        )

        await _publish_status(session_id, "status", phase="complementary-treatments-finished")
        await _publish_status(session_id, "done", kind="complementary-treatments")
        logger.info("Complementary treatments completed (session=%s)", session_id)

    except LLMUserCancelled:
        await _handle_cancelled(session_id, kind)

    except asyncio.CancelledError:
        logger.exception("Runtime cancelled task (user_cancel=%s, shutting_down=%s)",
                         cancel_event.is_set(), SHUTTING_DOWN)
        if cancel_event.is_set():
            await _handle_cancelled(session_id, kind)
        else:
            ek = _hyphen_kind(kind)
            await _publish_status(
                session_id, "error",
                message=f"Runtime cancelled task (shutting_down={SHUTTING_DOWN})"
            )
            # NEW: durable state → error
            await _persist_run_state(
                session_id, kind,
                status="error",
                phase=f"{ek}-runtime-cancel",
                error_message="Runtime cancelled task",
                finished_at=firestore.SERVER_TIMESTAMP,
            )
            await _publish_status(session_id, "done", kind=ek, error=True)
        return

    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Complementary treatments failed (session=%s): %s", session_id, e)

        # Fallback stub
        payload = {
            "items": [
                {
                    "name": "Información insuficiente",
                    "quantity": "N/A",
                    "notes": "Fallo del servicio LLM o de red; respuesta de respaldo.",
                }
            ]
        }
        await _publish_status(session_id, "complementary-treatments", **payload)
        await _persist_output(session_id, "complementary_treatments", payload)

        # NEW: durable state → error
        await _persist_run_state(
            session_id, kind,
            status="error",
            phase="complementary-treatments-error",
            error_message=str(e),
            finished_at=firestore.SERVER_TIMESTAMP,
        )

        await _publish_status(session_id, "done", kind=_hyphen_kind(kind), error=True)
