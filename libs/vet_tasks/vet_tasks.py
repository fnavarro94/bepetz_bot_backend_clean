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

# ------------------------------------------------------------------------------
# Taskiq tasks (queueable from API)
# ------------------------------------------------------------------------------
@vet_broker.task
async def run_diagnostics_task(session_id: str) -> None:
    """Queueable task: emit a minimal list of differentials + persist it."""
    print("running diagnostics task")
    await _publish_status(session_id, "status", phase="diagnostics-started")
    try:
        # (Optional) fetch dummy session data – not used for the static payload,
        # but keep it here in case you want to branch later.
        session = await _fetch_vet_session(session_id)
        if not session:
            await _publish_status(session_id, "error", message="No dummy data available for session")
            logger.warning("No dummy session data: %s", session_id)
            return

        # Build ONLY what the UI needs: a list of {name, probability, rationale}
        rationale = (
            "Porque presenta vómitos y diarrea postprandiales, poco apetito, "
            "decaimiento, leve deshidratación y sensibilidad abdominal."
        )
        items = [
            {"name": "Gastroenteritis aguda",     "probability": 0.67, "rationale": rationale},
            {"name": "Pancreatitis aguda",       "probability": 0.35, "rationale": rationale},
            {"name": "Indiscreción alimentaria", "probability": 0.24, "rationale": rationale},
        ]
        payload = {"items": items}

        # Stream to relay as a named SSE event: event='diagnostics', data=payload
        await _publish_status(session_id, "diagnostics", **payload)

        # Persist the same JSON so the UI can fetch it later if desired
        await _persist_output(session_id, "diagnostics", payload)

        await _publish_status(session_id, "status", phase="diagnostics-finished")
        await _publish_status(session_id, "done", kind="diagnostics")
        logger.info("Diagnostics completed (session=%s)", session_id)
    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Diagnostics failed (session=%s): %s", session_id, e)

@vet_broker.task
async def run_additional_exams_task(session_id: str) -> None:
    """Queueable task: emit a minimal list of complementary exams + persist it."""
    await _publish_status(session_id, "status", phase="additional-exams-started")
    try:
        # Keep fetch in case you want to branch by session later
        session = await _fetch_vet_session(session_id)
        if not session:
            await _publish_status(session_id, "error", message="No dummy data available for session")
            logger.warning("No dummy session data: %s", session_id)
            return

        # Build ONLY what the UI needs for the form
        items = [
            {
                "name": "Urianálisis",
                "indications": "Examen físico, químico y microscópico de orina.",
            },
            {
                "name": "Radiografía abdominal",
                "indications": "Proyecciones latero-lateral y ventro-dorsal.",
            },
        ]
        payload = {"items": items}

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
        session = await _fetch_vet_session(session_id)
        if not session:
            await _publish_status(session_id, "error", message="No dummy data available for session")
            logger.warning("No dummy session data: %s", session_id)
            return

        items = _build_dummy_medications(session)
        payload = {"items": items}

        # Stream to SSE relay with the named event this panel listens to
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
        session = await _fetch_vet_session(session_id)
        if not session:
            await _publish_status(session_id, "error", message="No dummy data available for session")
            logger.warning("No dummy session data: %s", session_id)
            return

        items = _build_dummy_complementaries(session)
        payload = {"items": items}

        # Named event must be 'complementary_treatments'
        await _publish_status(session_id, "complementary-treatments", **payload)

        # Persist
        await _persist_output(session_id, "complementary_treatments", payload)

        await _publish_status(session_id, "status", phase="complementary-treatments-finished")
        await _publish_status(session_id, "done", kind="complementary-treatments")
        logger.info("Complementary treatments completed (session=%s)", session_id)
    except Exception as e:
        await _publish_status(session_id, "error", message=str(e))
        logger.exception("Complementary treatments failed (session=%s): %s", session_id, e)
