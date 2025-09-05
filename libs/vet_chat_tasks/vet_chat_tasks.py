import os
import sys
import json
import time
import logging
from typing import Optional, Dict, Any

import asyncio
import redis.asyncio as aioredis
from google.cloud import firestore

from common.broker import vet_chat_broker
from vet_chat_tasks.openai_helper import OpenAIChatHelper

# ──────────────────────────────────────────────────────────────────────────────
# Logging (structured JSON)
# ──────────────────────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "ts": record.created,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        skip = {
            "args","asctime","created","exc_info","exc_text","filename","funcName",
            "levelname","levelno","lineno","module","msecs","message","msg","name",
            "pathname","process","processName","relativeCreated","stack_info",
            "thread","threadName"
        }
        ctx = {k: v for k, v in record.__dict__.items() if k not in skip}
        if ctx:
            base["ctx"] = ctx
        return json.dumps(base, default=str)

logger = logging.getLogger("vet-chat-worker")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
logger.handlers = [_handler]
logger.propagate = False

# ──────────────────────────────────────────────────────────────────────────────
# Firestore (metadata only)
# ──────────────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB   = os.getenv("CHATS_FIRESTORE_DB")
db = firestore.AsyncClient(project=GCP_PROJECT_ID, database=FIRESTORE_DB)

CONSULTATIONS_COLL = os.getenv("VET_CHAT_COLLECTION", "vet_chat_consultations")

def _consult_ref(consultation_id: str):
    return db.collection(CONSULTATIONS_COLL).document(consultation_id)

async def _ensure_thread_doc(consultation_id: str) -> None:
    ref = _consult_ref(consultation_id)
    snap = await ref.get()
    if not snap.exists:
        await ref.set({
            "consultation_id": consultation_id,
            "openai_thread_id": None,
            "openai_last_response_id": None,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

async def _touch_model(consultation_id: str, *, model: Optional[str]) -> None:
    await _consult_ref(consultation_id).set(
        {"model": model, "updated_at": firestore.SERVER_TIMESTAMP},
        merge=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Redis Pub/Sub (to relay)
# ──────────────────────────────────────────────────────────────────────────────
STREAM_REDIS_HOST = os.getenv("STREAM_REDIS_HOST", "localhost")
STREAM_REDIS_PORT = int(os.getenv("STREAM_REDIS_PORT", "6379"))
STREAM_REDIS_SSL  = os.getenv("STREAM_REDIS_SSL", "false").lower() == "true"

redis_stream: aioredis.Redis | None = aioredis.Redis(
    host=STREAM_REDIS_HOST,
    port=STREAM_REDIS_PORT,
    ssl=STREAM_REDIS_SSL,
    encoding="utf-8",
    decode_responses=True,
)

def _channel(consultation_id: str) -> str:
    return f"vet_chat:{consultation_id}"

def _ready_key_for_channel(channel_name: str) -> str:
    # matches relay's _ready_key convention: f"sse:ready:{channel}"
    return f"sse:ready:{channel_name}"

async def _publish_status(consultation_id: str, phase: str, **extra) -> None:
    if not redis_stream:
        return
    ev = {"event": "status", "data": {"phase": phase, **extra}}
    await redis_stream.publish(_channel(consultation_id), json.dumps(ev))

# Prefer strict in-order delivery: await the publish in the loop.
async def _publish_delta(consultation_id: str, text: str) -> None:
    if not redis_stream or not text:
        return
    await redis_stream.publish(_channel(consultation_id), text)

# ──────────────────────────────────────────────────────────────────────────────
# Config knobs
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL            = os.getenv("VET_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_INSTRUCTIONS     = os.getenv("VET_CHAT_INSTRUCTIONS")
DEFAULT_REASONING_EFFORT = os.getenv("VET_CHAT_REASONING_EFFORT")
DEFAULT_MAX_OUTPUT       = os.getenv("VET_CHAT_MAX_OUTPUT_TOKENS")
DEFAULT_MAX_OUTPUT       = int(DEFAULT_MAX_OUTPUT) if (DEFAULT_MAX_OUTPUT and DEFAULT_MAX_OUTPUT.isdigit()) else None

# Wait briefly for FE to attach to relay (optional but avoids lost-first-chunk).
READY_WAIT_MS = int(os.getenv("VET_CHAT_READY_WAIT_MS", "1200"))
# Log stream heartbeats (throttle interval in seconds)
STREAM_LOG_HEARTBEAT_S = float(os.getenv("VET_CHAT_LOG_HEARTBEAT_S", "0.5"))
# Toggle per-chunk logging (guarded by heartbeat to avoid spam)
LOG_STREAM = os.getenv("VET_CHAT_LOG_STREAM", "1") == "1"

async def _wait_until_ready(channel_name: str, timeout_ms: int = READY_WAIT_MS) -> None:
    if not redis_stream or timeout_ms <= 0:
        return
    key = _ready_key_for_channel(channel_name)
    deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000.0)
    # quick initial check
    try:
        if await redis_stream.get(key):
            return
    except Exception:
        return
    # poll briefly
    while asyncio.get_event_loop().time() < deadline:
        try:
            if await redis_stream.get(key):
                return
        except Exception:
            break
        await asyncio.sleep(0.05)

# ──────────────────────────────────────────────────────────────────────────────
# TaskIQ task
# ──────────────────────────────────────────────────────────────────────────────
@vet_chat_broker.task
async def process_vet_chat_message_task(
    consultation_id: str,
    message: str,
    attachments: list[dict] | None = None,   # reserved
    *,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    reasoning_effort: Optional[str] = None,  # GPT-5 only
    max_output_tokens: Optional[int] = None,
) -> None:
    """
    1) Ensure a metadata doc for this consultation.
    2) Wait briefly for the relay/FE 'ready' latch (optional).
    3) Stream text deltas from OpenAI and publish each chunk to Redis (SSE).
    4) Always send an explicit end-of-stream sentinel and a final 'done' status.
    """
    await _ensure_thread_doc(consultation_id)
    await _publish_status(consultation_id, phase="started")

    model             = model or DEFAULT_MODEL
    system_prompt     = instructions if instructions is not None else DEFAULT_INSTRUCTIONS
    reasoning_effort  = reasoning_effort or DEFAULT_REASONING_EFFORT
    max_output_tokens = max_output_tokens if max_output_tokens is not None else DEFAULT_MAX_OUTPUT

    # touch metadata
    await _touch_model(consultation_id, model=model)

    channel = _channel(consultation_id)

    try:
        await _publish_status(consultation_id, phase="accepted", model=model)

        # Optional: let the relay mark 'ready' after FE connects
        await _wait_until_ready(channel)

        helper = OpenAIChatHelper(model=model)

        token_count = 0
        last_log_t  = time.monotonic()

        for chunk in helper.stream_text(
            user_text=message,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,   # ignored by 4o/4o-mini; used by GPT-5.*
        ):
            if not chunk:
                continue

            token_count += 1
            await _publish_delta(consultation_id, chunk)

            # Lightweight streaming heartbeat logs
            if LOG_STREAM:
                now = time.monotonic()
                if token_count == 1 or (now - last_log_t) >= STREAM_LOG_HEARTBEAT_S:
                    logger.info(
                        "vet_chat_stream_delta",
                        extra={
                            "consultation_id": consultation_id,
                            "tokens": token_count,
                            "delta_len": len(chunk),
                            "delta_snippet": chunk[:80],
                        },
                    )
                    last_log_t = now

        await _publish_status(consultation_id, phase="completed")
        logger.info("vet_chat_stream_completed", extra={"consultation_id": consultation_id, "tokens": token_count})

    except Exception as e:
        await _publish_status(consultation_id, phase="error", message=str(e))
        logger.error("vet_chat_worker_exception", extra={"consultation_id": consultation_id, "err": str(e)})

    finally:
        # Explicit EOS so relay emits `event: done` and the FE flips out of streaming
        if redis_stream:
            try:
                await redis_stream.publish(channel, "[END-OF-STREAM]")
            except Exception as pub_err:
                logger.error("vet_chat_eos_publish_failed", extra={"consultation_id": consultation_id, "err": str(pub_err)})

        # Final status (belt & suspenders)
        await _publish_status(consultation_id, phase="done")
        logger.info("vet_chat_stream_done", extra={"consultation_id": consultation_id})
