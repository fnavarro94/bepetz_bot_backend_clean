# api.py — Single-continuum API

import os
from uuid import uuid4
from datetime import datetime, timedelta
from typing import List

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
