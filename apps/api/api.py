# api_underscores.py — Copy of API with underscore-standardized kinds

import os
from uuid import uuid4
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

# from dotenv import load_dotenv
# load_dotenv(override=True)

from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from redis.asyncio import Redis
from common.redis_conn import get_redis_connection

from tasks.tasks import process_message_task  # worker task (signature: user_id, message, attachments)

from google.cloud import storage
from google.cloud import firestore
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth


from typing import Optional, Dict, Any
import asyncio

from google.cloud.firestore_v1 import FieldFilter  # for equality filters

# ── New: Vet workflow tasks (Diagnostics / Exams / Prescription / Complementary) ──
from vet_chat_tasks.vet_chat_tasks import process_vet_chat_message_task
try:
    # when the new worker file is added (below), these imports will resolve
    from vet_tasks.vet_tasks import (
        run_diagnostics_task,
        run_additional_exams_task,
        run_prescription_task,
        run_complementary_treatments_task,
    )
except Exception:
    # Safe fallback so API can start even if worker file isn't deployed yet.
    run_diagnostics_task = None
    run_additional_exams_task = None
    run_prescription_task = None
    run_complementary_treatments_task = None

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

class UploadURLRequest(BaseModel):
    user_id: int
    file_name: str
    mime_type: str  # e.g. "image/png"

class UploadURLResponse(BaseModel):
    upload_url: str
    bucket: str
    object_path: str
    expires_at: datetime

class AttachmentMeta(BaseModel):
    bucket: str
    object_path: str
    mime_type: str
    file_name: str

class ChatRequest(BaseModel):
    user_id: int
    message: str
    attachments: List[AttachmentMeta] = Field(default_factory=list)

class ChatMessage(BaseModel):
    timestamp: datetime
    role: str
    content: str
    attachments: List[AttachmentMeta] = Field(default_factory=list)

class MessagesResponse(BaseModel):
    # Always the single continuum id for this user: f"u{user_id}:continuum"
    conversation_id: str
    messages: List[ChatMessage]

class QueueResponse(BaseModel):
    status: str
    task_id: str
    user_id: int
    # FE subscribes to this single continuum id
    conversation_id: str


class SessionMeta(BaseModel):
    session_id: str
    generation: int = 0
    status: str = "unknown"
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    is_active: bool = False

class SessionListResponse(BaseModel):
    user_id: int
    active_session_id: Optional[str]
    sessions: List[SessionMeta]

class SessionIdResponse(BaseModel):
    user_id: int
    session: Optional[SessionMeta] = None

class SessionViewResponse(BaseModel):
    user_id: int
    session: SessionMeta
    # From the timeline "summary" entry
    summary: Optional[str] = None
    notes: Optional[str] = None
    # Messages that belong to ONLY this session (hydrated from message_ref)
    messages: List[ChatMessage]

class VetQueueResponse(BaseModel):
    status: str
    task_id: str
    session_id: str
    kind: str


#---------------------------------
# Persistent state models
#---------------------------------

# ── NEW: Vet run-state models ─────────────────────────────────────────
from typing import Optional, Dict

class VetRunState(BaseModel):
    status: str = "idle"                 # idle|queued|running|cancel_requested|done|error|cancelled
    phase: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class VetStateResponse(BaseModel):
    session_id: str
    runs: Dict[str, VetRunState]         # by kind
    # last persisted output "updated_at" per kind (quick FE hint)
    outputs_updated_at: Dict[str, Optional[datetime]]


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Chatbot API",
    description="Receives user messages and queues them for processing by a worker.",
    version="1.2.0",
)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://bepetz-chatbot-ui-dev.web.app",
    # "https://bepetz-chatbot-ui-dev--felipe-preview.web.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
)

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB = os.getenv("CHATS_FIRESTORE_DB")
db = firestore.AsyncClient(project=GCP_PROJECT_ID, database=FIRESTORE_DB)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _continuum_id(user_id: int) -> str:
    return f"u{user_id}:continuum"

async def _get_active_session_id(user_id: int) -> Optional[str]:
    cont_ref = db.collection("continuums").document(str(user_id))
    snap = await cont_ref.get()
    d = snap.to_dict() or {}
    return d.get("active_session_id")

async def _list_session_docs_for_user(user_id: int):
    q = (
        db.collection("sessions")
        .where(filter=FieldFilter("user_id", "==", str(user_id)))
    )
    return [s async for s in q.stream()]

def _session_meta_from_snap(snap, active_id: Optional[str]) -> SessionMeta:
    d = snap.to_dict() or {}
    return SessionMeta(
        session_id=snap.id,
        generation=int(d.get("generation", 0)),
        status=d.get("status", "unknown"),
        start_at=d.get("start_at"),
        end_at=d.get("end_at"),
        is_active=(snap.id == active_id),
    )


def continuum_id_for_user(user_id: int) -> str:
    """Single canonical conversation id the FE subscribes to."""
    return f"u{user_id}:continuum"


def _generate_put_signed_url(blob, content_type: str, minutes: int = 15) -> str:
    """
    Generate a V4 signed URL for PUT uploads that works both locally (JSON key)
    and on Cloud Run (keyless via IAM Credentials API).
    """
    creds, _ = google.auth.default()

    # A) Local with service-account JSON -> creds can sign locally
    if isinstance(creds, service_account.Credentials):
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=minutes),
            method="PUT",
            content_type=content_type,
            credentials=creds,
        )

    # B) Cloud Run / token-only -> use IAM-based signing
    creds.refresh(Request())
    sa_email = getattr(creds, "service_account_email", None) or os.getenv("SA_EMAIL")
    if not sa_email:
        raise RuntimeError(
            "No service account email available for keyless signing. "
            "Set SA_EMAIL or run the service with a service account."
        )

    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="PUT",
        content_type=content_type,
        service_account_email=sa_email,
        access_token=creds.token,
    )



# ──────────────────────────────────────────────────────────────────────────────
# Persistent state helpers
# ──────────────────────────────────────────────────────────────────────────────

# ── NEW: Firestore refs & helpers for vet run states ─────────────────
_VET_KINDS = ["diagnostics", "additional_exams", "prescription", "complementary_treatments"]

def _vet_run_ref(session_id: str, kind: str):
    return (db.collection("vet_sessions")
             .document(session_id)
             .collection("runs")
             .document(kind))

def _vet_output_ref(session_id: str, kind: str):
    return (db.collection("vet_sessions")
             .document(session_id)
             .collection("outputs")
             .document(kind))

async def _set_vet_run_state(session_id: str, kind: str, **fields):
    """Merge-write run state; always bump updated_at."""
    payload = {**fields, "updated_at": firestore.SERVER_TIMESTAMP}
    await _vet_run_ref(session_id, kind).set(payload, merge=True)

async def _get_vet_run_state(session_id: str, kind: str) -> VetRunState:
    snap = await _vet_run_ref(session_id, kind).get()
    d = snap.to_dict() or {}
    # default to "idle" if nothing written yet
    return VetRunState(**({"status": "idle"} | d))

async def _get_vet_state_aggregate(session_id: str) -> VetStateResponse:
    runs: Dict[str, VetRunState] = {}
    outputs_updated_at: Dict[str, Optional[datetime]] = {}

    for k in _VET_KINDS:
        # run-state
        r = await _get_vet_run_state(session_id, k)
        runs[k] = r

        # output updated_at (helpful to know if UI can show last result)
        osnap = await _vet_output_ref(session_id, k).get()
        od = osnap.to_dict() or {}
        outputs_updated_at[k] = od.get("updated_at")

    return VetStateResponse(session_id=session_id, runs=runs, outputs_updated_at=outputs_updated_at)



# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Chatbot API is running."}


@app.post("/api/v1/message", status_code=status.HTTP_202_ACCEPTED, response_model=QueueResponse)
async def receive_message(request: ChatRequest, redis: Redis = Depends(get_redis_connection)):
    """
    Accept a user message for the user's single continuum. The worker manages
    sessions, rollover, and timeline persistence.
    """
    continuum_id = continuum_id_for_user(request.user_id)
    print(f"[API] Received message from user '{request.user_id}' for continuum '{continuum_id}'. Queuing task...")

    task = await process_message_task.kiq(
        user_id=request.user_id,
        message=request.message,
        attachments=[a.dict() for a in request.attachments],
    )

    print(f"[API] Task '{task.task_id}' for user '{request.user_id}' has been queued.")
    return QueueResponse(
        status="Message received and queued for processing.",
        task_id=task.task_id,
        user_id=request.user_id,
        conversation_id=continuum_id,
    )


@app.get("/api/v1/continuum/{user_id}/messages", response_model=MessagesResponse)
async def get_continuum_messages(user_id: int, limit: int = 200):
    """
    Return up to `limit` messages for the user's evergreen continuum,
    in ascending timestamp order. If the continuum hasn't been created yet,
    returns an empty list.
    """
    conv_id = continuum_id_for_user(user_id)
    conv_doc = db.collection("conversations").document(conv_id)
    msg_ref = conv_doc.collection("messages")

    # If the conversation doc doesn't exist yet, return empty
    convo_snap = await conv_doc.get()
    if not convo_snap.exists:
        return MessagesResponse(conversation_id=conv_id, messages=[])

    q = msg_ref.order_by("timestamp").limit(limit)
    docs = [doc async for doc in q.stream()]
    messages: List[ChatMessage] = []
    for snap in docs:
        m = snap.to_dict() or {}
        messages.append(ChatMessage(
            timestamp=m.get("timestamp"),
            role=m.get("role"),
            content=((m.get("content") or "").strip()
                      or ((m.get("asr") or {}).get("text") or "").strip()),
            attachments=m.get("attachments") or [],
        ))
    return MessagesResponse(conversation_id=conv_id, messages=messages)


@app.post("/api/v1/upload-url", response_model=UploadURLResponse)
async def get_upload_url(req: UploadURLRequest):
    """
    Pre-sign a GCS PUT URL for file uploads. Path is per-user (no conversation id).
    Filename extension is normalized to the provided MIME type so downstream
    workers can reason about formats reliably (e.g., .m4a for audio/mp4).
    """
    bucket_name = os.getenv("CHAT_UPLOAD_BUCKET")
    if not bucket_name:
        raise RuntimeError("CHAT_UPLOAD_BUCKET env var is required")

    # Normalize extension from MIME (ignore codec suffixes like ';codecs=opus')
    base_mime = req.mime_type.split(";", 1)[0].strip().lower()
    ext_map = {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/opus": "opus",
        "audio/mp4": "m4a",     # iOS/Safari voice notes
        "audio/aac": "m4a",     # treat AAC as m4a container downstream
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/flac": "flac",
    }

    # Pick extension: prefer MIME-derived; else keep/derive from provided file_name; else 'bin'
    requested_name = os.path.basename(req.file_name or "upload")  # defensive
    name_root, name_ext = os.path.splitext(requested_name)
    mime_ext = ext_map.get(base_mime)
    final_ext = mime_ext or (name_ext.lstrip(".") if name_ext else "") or "bin"
    safe_name = f"{name_root}.{final_ext}"

    object_path = f"{req.user_id}/incoming/{uuid4()}_{safe_name}"

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_path)

    # Signed URL enforces the exact Content-Type the client must PUT with
    url = _generate_put_signed_url(blob, content_type=req.mime_type, minutes=15)

    return UploadURLResponse(
        upload_url=url,
        bucket=bucket_name,
        object_path=object_path,
        expires_at=datetime.utcnow() + timedelta(minutes=15),
    )



@app.get("/api/v1/sessions/{user_id}/list", response_model=SessionListResponse)
async def list_sessions(user_id: int):
    """
    List all sessions for a user with minimal metadata and which one is active.
    """
    active_id = await _get_active_session_id(user_id)
    snaps = await _list_session_docs_for_user(user_id)

    # Sort in Python to avoid extra composite indexes:
    # first by generation asc (oldest -> newest)
    metas = sorted(
        (_session_meta_from_snap(s, active_id) for s in snaps),
        key=lambda m: m.generation,
    )

    return SessionListResponse(
        user_id=user_id,
        active_session_id=active_id,
        sessions=metas,
    )


@app.get("/api/v1/sessions/{user_id}/active", response_model=SessionIdResponse)
async def get_active_session(user_id: int):
    """
    Convenience endpoint to fetch metadata for the active session.
    """
    active_id = await _get_active_session_id(user_id)
    if not active_id:
        return SessionIdResponse(user_id=user_id, session=None)

    # Read its doc for meta
    sref = db.collection("sessions").document(active_id)
    ssnap = await sref.get()
    if not ssnap.exists:
        return SessionIdResponse(user_id=user_id, session=None)

    meta = _session_meta_from_snap(ssnap, active_id)
    return SessionIdResponse(user_id=user_id, session=meta)


@app.get("/api/v1/sessions/{user_id}/previous", response_model=SessionIdResponse)
async def get_previous_session(user_id: int):
    """
    Convenience endpoint to fetch the session *before* the active one by generation.
    """
    active_id = await _get_active_session_id(user_id)
    snaps = await _list_session_docs_for_user(user_id)
    metas = sorted(
        (_session_meta_from_snap(s, active_id) for s in snaps),
        key=lambda m: m.generation,
    )

    if not active_id or not metas:
        return SessionIdResponse(user_id=user_id, session=None)

    # Find active, then pick previous by generation
    idx = next((i for i, m in enumerate(metas) if m.session_id == active_id), None)
    if idx is None or idx == 0:
        return SessionIdResponse(user_id=user_id, session=None)

    return SessionIdResponse(user_id=user_id, session=metas[idx - 1])


@app.get("/api/v1/sessions/{user_id}/{session_id}/view", response_model=SessionViewResponse)
async def get_session_view(user_id: int, session_id: str):
    """
    Hydrated session 'view':
      - meta (status, generation, start/end)
      - the latest 'summary' entry (plus memory_delta.notes) from the timeline
      - only the messages that belong to this session, ordered by timeline ts
    """
    # Meta
    sref = db.collection("sessions").document(session_id)
    ssnap = await sref.get()
    if not ssnap.exists:
        raise RuntimeError("Session not found")

    active_id = await _get_active_session_id(user_id)
    meta = _session_meta_from_snap(ssnap, active_id)

    # Timeline (ordered by ts asc)
    t_ref = sref.collection("timeline")
    t_docs = [d async for d in t_ref.order_by("ts").stream()]

    # Collect message ids in timeline order and capture summary/notes
    message_ids: List[str] = []
    summary_text: Optional[str] = None
    notes_text: Optional[str] = None

    for d in t_docs:
        item = d.to_dict() or {}
        kind = item.get("kind")
        if kind == "message_ref":
            mid = item.get("message_id")
            if mid:
                message_ids.append(mid)
        elif kind == "summary":
            # Prefer the *last* summary we see in order (latest one)
            summary_text = (item.get("summary") or "") or summary_text
            md = item.get("memory_delta") or {}
            notes_text = (md.get("notes") or "") or notes_text

    # Hydrate message docs from the evergreen continuum
    conv_id = _continuum_id(user_id)
    mcoll = db.collection("conversations").document(conv_id).collection("messages")

    async def _fetch_msg(mid: str):
        snap = await mcoll.document(mid).get()
        d = snap.to_dict() or {}
        return ChatMessage(
            timestamp=d.get("timestamp"),
            role=d.get("role", "assistant"),
            content=((d.get("content") or "").strip() or ((d.get("asr") or {}).get("text") or "").strip()),
            attachments=d.get("attachments") or [],
        )

    # Preserve order from timeline
    msgs: List[ChatMessage] = []
    if message_ids:
        msgs = await asyncio.gather(*[_fetch_msg(mid) for mid in message_ids])

    return SessionViewResponse(
        user_id=user_id,
        session=meta,
        summary=summary_text,
        notes=notes_text,
        messages=msgs,
    )



# ──────────────────────────────────────────────────────────────────────────────
# Vet workflow queueing endpoints (independent from chat flow)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/vet/{session_id}/diagnostics/queue",
          response_model=VetQueueResponse,
          status_code=status.HTTP_202_ACCEPTED)
async def queue_vet_diagnostics(session_id: str):
    if not run_diagnostics_task:
        raise RuntimeError("Diagnostics worker not available. Did you deploy tasks/vet_tasks.py?")
    task = await run_diagnostics_task.kiq(session_id=session_id)

    # Persist durable run state
    await _set_vet_run_state(
        session_id, "diagnostics",
        status="queued",
        phase="diagnostics_queued",
        started_at=firestore.SERVER_TIMESTAMP,
    )

    return VetQueueResponse(
        status="queued",
        task_id=task.task_id,
        session_id=session_id,
        kind="diagnostics",
    )


@app.post("/api/v1/vet/{session_id}/additional_exams/queue",
          response_model=VetQueueResponse,
          status_code=status.HTTP_202_ACCEPTED)
async def queue_vet_additional_exams(session_id: str):
    if not run_additional_exams_task:
        raise RuntimeError("Additional_exams worker not available. Did you deploy tasks/vet_tasks.py?")
    task = await run_additional_exams_task.kiq(session_id=session_id)

    # Persist durable run state
    await _set_vet_run_state(
        session_id, "additional_exams",
        status="queued",
        phase="additional_exams_queued",
        started_at=firestore.SERVER_TIMESTAMP,
    )

    return VetQueueResponse(
        status="queued",
        task_id=task.task_id,
        session_id=session_id,
        kind="additional_exams",
    )

@app.post("/api/v1/vet/{session_id}/prescription/queue",
          response_model=VetQueueResponse,
          status_code=status.HTTP_202_ACCEPTED)
async def queue_vet_prescription(session_id: str):
    if not run_prescription_task:
        raise RuntimeError("Prescription worker not available. Did you deploy tasks/vet_tasks.py?")
    task = await run_prescription_task.kiq(session_id=session_id)

    # Persist durable run state
    await _set_vet_run_state(
        session_id, "prescription",
        status="queued",
        phase="prescription_queued",
        started_at=firestore.SERVER_TIMESTAMP,
    )


    return VetQueueResponse(
        status="queued",
        task_id=task.task_id,
        session_id=session_id,
        kind="prescription",
    )


@app.post("/api/v1/vet/{session_id}/complementary_treatments/queue",
          response_model=VetQueueResponse,
          status_code=status.HTTP_202_ACCEPTED)
async def queue_vet_complementary(session_id: str):
    if not run_complementary_treatments_task:
        raise RuntimeError("Complementary_treatments worker not available. Did you deploy tasks/vet_tasks.py?")
    task = await run_complementary_treatments_task.kiq(session_id=session_id)

    # Persist durable run state
    await _set_vet_run_state(
        session_id, "complementary_treatments",
        status="queued",
        phase="complementary_treatments_queued",
        started_at=firestore.SERVER_TIMESTAMP,
    )


    return VetQueueResponse(
        status="queued",
        task_id=task.task_id,
        session_id=session_id,
        kind="complementary_treatments",
    )




### Cancel code
# --- imports (top of file, near others) ---
import json
import redis.asyncio as aioredis
from enum import Enum
from pydantic import BaseModel

# --- streaming/control Redis (same env you use in relay/worker) ---
STREAM_REDIS_HOST = os.getenv("STREAM_REDIS_HOST", "localhost")
STREAM_REDIS_PORT = int(os.getenv("STREAM_REDIS_PORT", "6379"))
STREAM_REDIS_SSL  = os.getenv("STREAM_REDIS_SSL", "false").lower() == "true"

redis_stream = aioredis.Redis(
    host=STREAM_REDIS_HOST,
    port=STREAM_REDIS_PORT,
    ssl=STREAM_REDIS_SSL,
    encoding="utf-8",
    decode_responses=True,
)

# --- small models ---
class VetKind(str, Enum):
    diagnostics = "diagnostics"
    additional_exams = "additional_exams"
    prescription = "prescription"
    complementary_treatments = "complementary_treatments"

class VetCancelResponse(BaseModel):
    status: str
    session_id: str
    kind: str


# --- cooperative cancel endpoint (underscored kinds; hyphens normalized) ---
@app.post("/api/v1/vet/{session_id}/{kind}/cancel",
          response_model=VetCancelResponse,
          status_code=status.HTTP_202_ACCEPTED)
async def cancel_vet_step(session_id: str, kind: str):
    """
    Accepts kind keys using underscores (preferred). If a hyphenated slug is
    provided, it is normalized to underscores internally and in responses.
    """
    # hyphen → underscore normalization (idempotent for underscore input)
    kind_key = _norm_kind(kind)

    # Optional: update persisted run-state so FE reflects cancel intent immediately
    cur = await _get_vet_run_state(session_id, kind_key)
    if (cur.status or "").lower() == "queued":
        # If it was just queued (not started), mark as cancelled right away
        await _set_vet_run_state(
            session_id, kind_key,
            status="cancelled",
            phase=f"{kind_key}_cancelled",
            finished_at=firestore.SERVER_TIMESTAMP,
        )
    else:
        # Otherwise mark that a cancel was requested; worker will flip to cancelled
        await _set_vet_run_state(
            session_id, kind_key,
            status="cancel_requested",
            phase=f"{kind_key}_cancel_requested",
        )

    # Send durable+ephemeral cancel signal to workers
    control_channel = f"vet:{session_id}:control"
    flag_key = f"vet:{session_id}:{kind_key}:cancelled"      # underscore key for sticky bit

    payload = {"event": "cancel", "data": {"kind": kind_key}}  # worker expects underscore key
    stamp = {"ts": datetime.utcnow().isoformat() + "Z"}

    pipe = redis_stream.pipeline()
    pipe.set(flag_key, json.dumps(stamp), ex=3600)             # sticky 1h
    pipe.publish(control_channel, json.dumps(payload))         # instant notify
    await pipe.execute()

    # Optional: nudge UIs on the general channel (underscore phase for consistency)
    await redis_stream.publish(
        f"vet:{session_id}",
        json.dumps({"event": "status", "data": {"phase": f"{kind_key}_cancel_requested"}})
    )

    # Return the normalized kind key (underscore)
    return VetCancelResponse(status="cancel_requested", session_id=session_id, kind=kind_key)




# ── Helpers for path-kind normalization (accept hyphens in URLs) ──
def _norm_kind(kind: str) -> str:
    return str(kind).replace("-", "_")

# ── NEW: aggregate state for a session ──
@app.get("/api/v1/vet/{session_id}/state", response_model=VetStateResponse)
async def get_vet_state(session_id: str):
    """
    Returns run-state for all vet kinds and a quick 'outputs_updated_at' map.
    """
    return await _get_vet_state_aggregate(session_id)

# ── OPTIONAL: per-kind run-state ──
@app.get("/api/v1/vet/{session_id}/runs/{kind}", response_model=VetRunState)
async def get_vet_run_state(session_id: str, kind: str):
    k = _norm_kind(kind)
    return await _get_vet_run_state(session_id, k)

# ── OPTIONAL: fetch last persisted output for a kind ──
from pydantic import BaseModel, Field

class VetOutputResponse(BaseModel):
    kind: str
    updated_at: Optional[datetime] = None
    result: Dict[str, Any] = Field(default_factory=dict)

@app.get("/api/v1/vet/{session_id}/outputs/{kind}", response_model=VetOutputResponse)
async def get_vet_output(session_id: str, kind: str):
    k = _norm_kind(kind)
    snap = await _vet_output_ref(session_id, k).get()
    d = snap.to_dict() or {}
    return VetOutputResponse(kind=k, updated_at=d.get("updated_at"), result=d.get("result") or {})




#--------------------------------------------------
# Vet Chat Components
#--------------------------------------------------

class VetChatMessageRequest(BaseModel):
    message: str
    attachments: List[AttachmentMeta] = Field(default_factory=list)

class VetChatQueueResponse(BaseModel):
    status: str
    task_id: str
    consultation_id: str


# ──────────────────────────────────────────────────────────────────────────────
# Vet-chat message endpoint (simple: consultation_id + message)
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/api/v1/vet_chat/{consultation_id}/message",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=VetChatQueueResponse,
)
async def queue_vet_chat_message(consultation_id: str, req: VetChatMessageRequest):
    """
    Minimal vet-chat flow:
    - Accepts consultation_id and a message (+ optional attachments).
    - Queues a worker job that:
        * looks up/creates the OpenAI session for this consultation,
        * sends the message,
        * streams tokens to your FE (implementation detail in vet_chat_tasks),
        * persists mapping consultation_id → openai_session_id for reuse.
    """
    if not process_vet_chat_message_task:
        raise RuntimeError("Vet-chat worker not available. Did you deploy vet_chat_tasks/vet_chat_tasks.py?")

    task = await process_vet_chat_message_task.kiq(
        consultation_id=consultation_id,
        message=req.message,
        attachments=[a.dict() for a in req.attachments],
    )

    return VetChatQueueResponse(
        status="queued",
        task_id=task.task_id,
        consultation_id=consultation_id,
    )



# ──────────────────────────────────────────────────────────────────────────────
# Vet Chat – history response models (place near your other Pydantic models)
# ──────────────────────────────────────────────────────────────────────────────

class VetChatHistoryMessage(BaseModel):
    id: str
    created_at: Optional[datetime] = None
    role: str
    content: str
    attachments: List[AttachmentMeta] = Field(default_factory=list)
    # Optional metadata your worker stores on assistant turns
    model: Optional[str] = None
    app_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    response_id: Optional[str] = None
    complete: Optional[bool] = None
    chunks: Optional[int] = None
    system_instructions: Optional[str] = None
    reply_to: Optional[str] = None

class VetChatHistoryResponse(BaseModel):
    consultation_id: str
    last_response_id: Optional[str] = None
    app_id: Optional[str] = None
    app: Optional[Dict[str, Any]] = None
    messages: List[VetChatHistoryMessage] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Vet Chat – turn state model
# ──────────────────────────────────────────────────────────────────────────────
class VetChatTurnStateResponse(BaseModel):
    consultation_id: str
    turn_id: Optional[str] = None
    turn_status: Optional[str] = None            # "started" | "error" | "done"
    turn_started_at: Optional[datetime] = None
    turn_completed_at: Optional[datetime] = None
    turn_updated_at: Optional[datetime] = None
    turn_error_message: Optional[str] = None
    turn_user_message_id: Optional[str] = None
    turn_assistant_message_id: Optional[str] = None
    turn_last_response_id: Optional[str] = None
    turn_model: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Vet Chat – fetch historical conversation (hard-coded collection name)
# ──────────────────────────────────────────────────────────────────────────────
@app.get(
    "/api/v1/vet_chat/{consultation_id}/history",
    response_model=VetChatHistoryResponse
)
async def get_vet_chat_history(
    consultation_id: str,
    limit: int = 250,
    after: Optional[datetime] = None,   # return messages strictly after this timestamp
    order: str = "asc",                 # "asc" (default) or "desc" by created_at
):
    """
    Retrieve the persisted conversation for a vet consultation:
      - Reads from: vet_chat_consultations/{consultation_id}/messages
      - Orders by 'created_at' (asc by default).
      - Supports 'after' cursor (timestamp) and 'limit' for pagination.
      - Also returns the consultation doc's last_response_id and app metadata.
    """
    # Clamp limit (defensive)
    limit = max(1, min(limit, 1000))

    # Consultation doc (holds app + chain anchor)
    consult_ref = db.collection("vet_chat_consultations").document(consultation_id)
    consult_snap = await consult_ref.get()

    if not consult_snap.exists:
        # Conversation hasn't started yet
        return VetChatHistoryResponse(
            consultation_id=consultation_id,
            last_response_id=None,
            app_id=None,
            app=None,
            messages=[],
        )

    consult_doc = consult_snap.to_dict() or {}
    app_id = consult_doc.get("openai_app_id")
    app = consult_doc.get("openai_app")
    last_response_id = consult_doc.get("openai_last_response_id")

    # Build the query against the messages subcollection
    mref = consult_ref.collection("messages")

    # Order direction
    direction = firestore.Query.ASCENDING
    if str(order).lower() == "desc":
        direction = firestore.Query.DESCENDING

    q = mref.order_by("created_at", direction=direction)

    # Optional 'after' cursor (strictly greater than the provided timestamp)
    if after:
        q = q.where(filter=FieldFilter("created_at", ">", after))

    q = q.limit(limit)

    docs = [d async for d in q.stream()]

    # Map Firestore docs to response model
    out_msgs: List[VetChatHistoryMessage] = []
    for d in docs:
        m = d.to_dict() or {}
        out_msgs.append(
            VetChatHistoryMessage(
                id=d.id,
                created_at=m.get("created_at"),
                role=m.get("role", "assistant"),
                content=m.get("content", ""),
                attachments=m.get("attachments") or [],
                model=m.get("model"),
                app_id=m.get("app_id"),
                previous_response_id=m.get("previous_response_id"),
                response_id=m.get("response_id"),
                complete=m.get("complete"),
                chunks=m.get("chunks"),
                system_instructions=m.get("system_instructions"),
                reply_to=m.get("reply_to"),
            )
        )

    return VetChatHistoryResponse(
        consultation_id=consultation_id,
        last_response_id=last_response_id,
        app_id=app_id,
        app=app,
        messages=out_msgs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Vet Chat – fetch current turn state (hard-coded collection name)
# ──────────────────────────────────────────────────────────────────────────────
@app.get(
    "/api/v1/vet_chat/{consultation_id}/turn_state",
    response_model=VetChatTurnStateResponse,
)
async def get_vet_chat_turn_state(consultation_id: str):
    """
    Returns the persisted turn state written by the worker so the FE can know
    whether a turn is 'started', 'error', or 'done' even if SSE disconnected.
    Reads from: vet_chat_consultations/{consultation_id}.
    """
    ref = db.collection("vet_chat_consultations").document(consultation_id)
    snap = await ref.get()

    if not snap.exists:
        # No conversation yet → empty state
        return VetChatTurnStateResponse(consultation_id=consultation_id)

    d = snap.to_dict() or {}
    return VetChatTurnStateResponse(
        consultation_id=consultation_id,
        turn_id=d.get("turn_id"),
        turn_status=d.get("turn_status"),
        turn_started_at=d.get("turn_started_at"),
        turn_completed_at=d.get("turn_completed_at"),
        turn_updated_at=d.get("turn_updated_at"),
        turn_error_message=d.get("turn_error_message"),
        turn_user_message_id=d.get("turn_user_message_id"),
        turn_assistant_message_id=d.get("turn_assistant_message_id"),
        turn_last_response_id=d.get("turn_last_response_id"),
        turn_model=d.get("turn_model"),
    )



# --- NEW: Vet Chat cancel response model (place near other Pydantic models) ---
class VetChatCancelResponse(BaseModel):
    status: str
    consultation_id: str
    turn_id: Optional[str] = None


# --- NEW: Vet Chat cooperative cancel endpoint ---
@app.post("/api/v1/vet_chat/{consultation_id}/cancel",
          response_model=VetChatCancelResponse,
          status_code=status.HTTP_202_ACCEPTED)
async def cancel_vet_chat_turn(consultation_id: str):
    """
    Cancels the *current* streaming turn for a given consultation.

    Behavior:
      - Looks up the consultation doc to get the current turn_id (if any).
      - Marks turn_status='cancel_requested' immediately.
      - Sets a turn-scoped sticky cancel flag in Redis with 1h TTL.
      - Publishes a 'cancel' control event so the worker can stop mid-stream.
      - Publishes a 'cancel_requested' status event on the main stream channel.
    """
    # Look up current turn_id (if no doc/turn -> graceful 'requested' anyway)
    ref = db.collection(os.getenv("VET_CHAT_COLLECTION", "vet_chat_consultations")).document(consultation_id)
    snap = await ref.get()
    turn_id = None
    if snap.exists:
        d = snap.to_dict() or {}
        turn_id = d.get("turn_id")

    # Update persisted UI state → cancel requested
    await ref.set(
        {"turn_status": "cancel_requested", "turn_updated_at": firestore.SERVER_TIMESTAMP},
        merge=True,
    )

    # Redis control channel + sticky key (scoped to turn if available)
    control_channel = f"vet_chat:{consultation_id}:control"
    if turn_id:
        flag_key = f"vet_chat:{consultation_id}:cancelled:{turn_id}"
        payload  = {"event": "cancel", "data": {"turn_id": turn_id}}
    else:
        # Fallback: generic cancel of "current" (worker will accept if its turn matches or treat as generic)
        flag_key = f"vet_chat:{consultation_id}:cancelled:any"
        payload  = {"event": "cancel", "data": {"turn_id": None}}

    stamp = {"ts": datetime.utcnow().isoformat() + "Z"}

    pipe = redis_stream.pipeline()
    pipe.set(flag_key, json.dumps(stamp), ex=3600)     # sticky 1h
    pipe.publish(control_channel, json.dumps(payload)) # instant notify
    await pipe.execute()

    # Optional: nudge UIs on the main SSE channel
    await redis_stream.publish(
        f"vet_chat:{consultation_id}",
        json.dumps({"event": "status", "data": {"phase": "cancel_requested"}})
    )

    return VetChatCancelResponse(status="cancel_requested", consultation_id=consultation_id, turn_id=turn_id)
