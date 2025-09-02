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

# ------------------------------------------------------------------------------
# Taskiq tasks (queueable from API)
# ------------------------------------------------------------------------------

@vet_broker.task
async def run_diagnostics_task(session_id: str) -> None:
    print(f"running diagnostics task")
    await _publish_status(session_id, "status", phase="diagnostics_started")
    try:
        session = await _fetch_vet_session(session_id)
        if not session:
            await _publish_status(session_id, "error", message="No dummy data available for session")
            logger.warning("No dummy session data: %s", session_id)
            return

        result = _draft_diagnostics_from_session(session)
        await _persist_output(session_id, "diagnostics", result)

        await _publish_status(session_id, "status", phase="diagnostics_finished")
        await _publish_status(session_id, "done", kind="diagnostics")
        logger.info("Diagnostics completed (session=%s)", session_id)
    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Diagnostics failed (session=%s): %s", session_id, e)


@vet_broker.task
async def run_additional_exams_task(session_id: str) -> None:
    await _publish_status(session_id, "status", phase="additional_exams_started")
    try:
        session = await _fetch_vet_session(session_id)
        # TODO: implement real logic
        result = {
            "message": "Additional exams generator not implemented yet.",
            "inputs_preview": {"complaints": session.get("complaints", []), "vitals": session.get("vitals", {})},
        }
        await _persist_output(session_id, "additional_exams", result)
        await _publish_status(session_id, "status", phase="additional_exams_finished")
        await _publish_status(session_id, "done", kind="additional_exams")
    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Additional exams failed (session=%s): %s", session_id, e)


@vet_broker.task
async def run_prescription_task(session_id: str) -> None:
    await _publish_status(session_id, "status", phase="prescription_started")
    try:
        session = await _fetch_vet_session(session_id)
        # TODO: implement real logic
        result = {"message": "Prescription generator not implemented yet.", "patient": session.get("patient")}
        await _persist_output(session_id, "prescription", result)
        await _publish_status(session_id, "status", phase="prescription_finished")
        await _publish_status(session_id, "done", kind="prescription")
    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Prescription failed (session=%s): %s", session_id, e)


@vet_broker.task
async def run_complementary_treatments_task(session_id: str) -> None:
    await _publish_status(session_id, "status", phase="complementary_started")
    try:
        session = await _fetch_vet_session(session_id)
        # TODO: implement real logic
        result = {"message": "Complementary treatments generator not implemented yet.", "history": session.get("history")}
        await _persist_output(session_id, "complementary_treatments", result)
        await _publish_status(session_id, "status", phase="complementary_finished")
        await _publish_status(session_id, "done", kind="complementary_treatments")
    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Complementary treatments failed (session=%s): %s", session_id, e)
