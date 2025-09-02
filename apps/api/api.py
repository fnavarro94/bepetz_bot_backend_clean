# api.py — Single-continuum API

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
    kind: str  # "diagnostics" | "additional_exams" | "prescription" | "complementary_treatments"



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
            content=m.get("content", ""),
            attachments=m.get("attachments") or [],
        ))
    return MessagesResponse(conversation_id=conv_id, messages=messages)


@app.post("/api/v1/upload-url", response_model=UploadURLResponse)
async def get_upload_url(req: UploadURLRequest):
    """
    Pre-sign a GCS PUT URL for file uploads. Path is per-user (no conversation id).
    """
    bucket_name = os.getenv("CHAT_UPLOAD_BUCKET")
    if not bucket_name:
        raise RuntimeError("CHAT_UPLOAD_BUCKET env var is required")

    object_path = f"{req.user_id}/incoming/{uuid4()}_{req.file_name}"
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_path)

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
            content=d.get("content", ""),
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

@app.post("/api/v1/vet/{session_id}/diagnostics/queue", response_model=VetQueueResponse, status_code=status.HTTP_202_ACCEPTED)
async def queue_vet_diagnostics(session_id: str):
    if not run_diagnostics_task:
        raise RuntimeError("Diagnostics worker not available. Did you deploy tasks/vet_tasks.py?")
    task = await run_diagnostics_task.kiq(session_id=session_id)
    return VetQueueResponse(status="queued", task_id=task.task_id, session_id=session_id, kind="diagnostics")


@app.post("/api/v1/vet/{session_id}/additional-exams/queue", response_model=VetQueueResponse, status_code=status.HTTP_202_ACCEPTED)
async def queue_vet_additional_exams(session_id: str):
    if not run_additional_exams_task:
        raise RuntimeError("Additional-exams worker not available. Did you deploy tasks/vet_tasks.py?")
    task = await run_additional_exams_task.kiq(session_id=session_id)
    return VetQueueResponse(status="queued", task_id=task.task_id, session_id=session_id, kind="additional_exams")


@app.post("/api/v1/vet/{session_id}/prescription/queue", response_model=VetQueueResponse, status_code=status.HTTP_202_ACCEPTED)
async def queue_vet_prescription(session_id: str):
    if not run_prescription_task:
        raise RuntimeError("Prescription worker not available. Did you deploy tasks/vet_tasks.py?")
    task = await run_prescription_task.kiq(session_id=session_id)
    return VetQueueResponse(status="queued", task_id=task.task_id, session_id=session_id, kind="prescription")


@app.post("/api/v1/vet/{session_id}/complementary-treatments/queue", response_model=VetQueueResponse, status_code=status.HTTP_202_ACCEPTED)
async def queue_vet_complementary(session_id: str):
    if not run_complementary_treatments_task:
        raise RuntimeError("Complementary-treatments worker not available. Did you deploy tasks/vet_tasks.py?")
    task = await run_complementary_treatments_task.kiq(session_id=session_id)
    return VetQueueResponse(status="queued", task_id=task.task_id, session_id=session_id, kind="complementary_treatments")
