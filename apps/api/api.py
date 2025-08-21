
import os
from dotenv import load_dotenv
#load_dotenv(override=True)
from fastapi import FastAPI, Depends, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from fastapi.middleware.cors import CORSMiddleware
# Internal imports
from common.redis_conn import get_redis_connection
from tasks.tasks import process_message_task

from google.cloud import storage
from uuid import uuid4
from datetime import timedelta, datetime

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# top of file (api service)
import re
from google.cloud import firestore

from typing import List, Optional


class UploadURLRequest(BaseModel):
    user_id: int
    conversation_id: str
    file_name: str
    mime_type: str             # e.g. "image/png"

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
    conversation_id: str
    message: str
    attachments: list[AttachmentMeta] = []      # NEW



class ConversationSummary(BaseModel):
    id: str
    subject: str
    created_at: datetime
    last_message_at: datetime

class ConversationsResponse(BaseModel):
    conversations: List[ConversationSummary]

class ChatMessage(BaseModel):
    timestamp: datetime
    role: str
    content: str
    attachments: List[AttachmentMeta] = Field(default_factory=list)

class MessagesResponse(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]




class QueueResponse(BaseModel):
    """Defines the structure for the API's response."""
    status: str
    task_id: str
    # This now correctly matches the integer type for user_id
    user_id: int
    # Added conversation_id to the response model
    conversation_id: str

# --- FastAPI Application ---
app = FastAPI(
    title="Chatbot API",
    description="Receives user messages and queues them for processing by a worker.",
    version="1.1.0" # Version bump to reflect changes
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB   = os.getenv("CHATS_FIRESTORE_DB")

print("Probando ci cd api 2")
db = firestore.AsyncClient(project=GCP_PROJECT_ID, database=FIRESTORE_DB)

def make_conversation_key(user_id: int, conversation_id: str) -> str:
    # normalize/sanitize to keep Redis/Firestore keys clean
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", (conversation_id or "").strip()) or "default"
    return f"u{user_id}:{safe}"

print(f"This is base url {os.getenv('BASE_URL')}")

# --- helper: works with both JSON-key and keyless (Cloud Run) ---
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
            credentials=creds,  # explicitly use the key on disk
        )

    # B) Cloud Run / token-only -> use IAM-based signing
    creds.refresh(Request())
    # Prefer ADC-attached SA email; allow override via env for local tweaks
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
        access_token=creds.token,  # triggers IAM Credentials API signing
    )

@app.post(
    "/api/v1/message",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=QueueResponse,
)
async def receive_message(request: ChatRequest, redis: Redis = Depends(get_redis_connection)):
    print(f"API: Received message from user '{request.user_id}' in conversation '{request.conversation_id}'. Queuing task...")

    # NEW: build globally-unique conversation key
    conversation_key = make_conversation_key(request.user_id, request.conversation_id)

    print(f"este es el payload asend messages {request}")
    task = await process_message_task.kiq(
        user_id=request.user_id,
        conversation_id=conversation_key,   # ← send the key to the worker
        message=request.message,
        attachments=[a.dict() for a in request.attachments],
    )

    print(f"API: Task '{task.task_id}' for user '{request.user_id}' has been queued.")

    # Return the standardized conversation_id (the key) so the FE can subscribe to it
    return QueueResponse(
        status="Message received and queued for processing.",
        task_id=task.task_id,
        user_id=request.user_id,
        conversation_id=conversation_key,   # ← IMPORTANT: return the key
    )


@app.get("/")
async def root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Chatbot API is running."}



@app.post("/api/v1/upload-url", response_model=UploadURLResponse)
async def get_upload_url(req: UploadURLRequest):
    bucket_name = os.getenv("CHAT_UPLOAD_BUCKET")
    object_path = f"{req.user_id}/{req.conversation_id}/{uuid4()}_{req.file_name}"

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_path)

    url = _generate_put_signed_url(blob, content_type=req.mime_type, minutes=15)

    return UploadURLResponse(
        upload_url=url,
        bucket=bucket_name,
        object_path=object_path,
        expires_at=datetime.utcnow() + timedelta(minutes=15),
    )



@app.get("/api/v1/conversations", response_model=ConversationsResponse)
async def list_conversations(user_id: int):
    """
    Return all conversation docs for this user, most-recent first.
    We sort in Python to avoid composite-index requirements.
    """
    conv_ref = db.collection("conversations")
    q = conv_ref.where("user_id", "==", str(user_id))
    docs = [doc async for doc in q.stream()]

    items: list[ConversationSummary] = []
    for snap in docs:
        data = snap.to_dict() or {}
        created_at = data.get("created_at")
        last_at    = data.get("last_message_at") or created_at
        items.append(ConversationSummary(
            id=snap.id,
            subject=data.get("subject", "Conversation"),
            created_at=created_at,
            last_message_at=last_at,
        ))

    items.sort(key=lambda i: i.last_message_at or i.created_at, reverse=True)
    return ConversationsResponse(conversations=items)



@app.get("/api/v1/conversations/{conversation_id}/messages", response_model=MessagesResponse)
async def get_conversation_messages(conversation_id: str, limit: int = 200):
    """
    Return up to `limit` messages in ascending timestamp order.
    """
    conv_doc = db.collection("conversations").document(conversation_id)
    msg_ref  = conv_doc.collection("messages")
    q = msg_ref.order_by("timestamp").limit(limit)

    docs = [doc async for doc in q.stream()]
    messages: list[ChatMessage] = []
    for snap in docs:
        m = snap.to_dict() or {}
        messages.append(ChatMessage(
            timestamp=m.get("timestamp"),
            role=m.get("role"),
            content=m.get("content", ""),
            attachments=m.get("attachments") or [],
        ))

    return MessagesResponse(conversation_id=conversation_id, messages=messages)
