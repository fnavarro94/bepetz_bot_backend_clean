# ==============================================================================
# File: tasks.py (stream‑enabled worker with non-blocking publish)
# Purpose: Consume user messages, run ADK agent with SSE streaming, and publish
#           each token chunk to a Google Cloud Pub/Sub topic.
# ==============================================================================
import os
os.environ["GRPC_DNS_RESOLVER"] = "native" 
import logging
import time
import mimetypes
import uuid
from datetime import datetime, timezone

from common.broker import broker
from common.db_queries import get_all_pet_details_by_user_id

# --- Google Cloud --------------------------------------------------------------
from google.cloud import pubsub_v1
from google.cloud import firestore

import redis.asyncio as aioredis     # ← NEW
import asyncio                       # ← NEW

from common.adk_client import ensure_session, run_agent_stream, create_session
import httpx

from google.cloud import storage
from google.genai import types   # ADK uses Part/Blob underneath

from dotenv import load_dotenv
from google.cloud.firestore_v1 import FieldFilter
load_dotenv(override=True)


# ------------------------------------------------------------------------------
#  Logging
# ------------------------------------------------------------------------------
# --- Structured JSON logging so `extra=` is visible ---
import sys, json

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "ts": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # include all custom attributes added via `extra=...`
        skip = {
            "args","asctime","created","exc_info","exc_text","filename","funcName",
            "levelname","levelno","lineno","module","msecs","message","msg","name",
            "pathname","process","processName","relativeCreated","stack_info","thread","threadName"
        }
        ctx = {k: v for k, v in record.__dict__.items() if k not in skip}
        if ctx:
            base["ctx"] = ctx
        return json.dumps(base, default=str)

logger = logging.getLogger("chat-worker")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
logger.handlers = [_handler]
logger.propagate = False



# ------------------------------------------------------------------------------
#  Firestore & Pub/Sub clients
# ------------------------------------------------------------------------------

# GCS client is thread-safe – create once
gcs = storage.Client()
publisher = pubsub_v1.PublisherClient(
    publisher_options=pubsub_v1.types.PublisherOptions(enable_message_ordering=True)
)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB = os.getenv("CHATS_FIRESTORE_DB")
print(f"Using Firestore DB: {FIRESTORE_DB}")
TOPIC_ID       = "chatbot-stream-topic"
TOPIC_PATH     = publisher.topic_path(GCP_PROJECT_ID, TOPIC_ID)


print(f"Confirming this is the new version. running with  GPC_PROJECT_ID {GCP_PROJECT_ID} and  FIRESTORE_DB = {FIRESTORE_DB}")
db = firestore.AsyncClient(project=GCP_PROJECT_ID, database=FIRESTORE_DB)

# ── NEW: Redis client for parallel streaming ─────────────────────────
STREAM_REDIS_HOST = os.getenv("STREAM_REDIS_HOST")
STREAM_REDIS_PORT = int(os.getenv("STREAM_REDIS_PORT", "6379"))
STREAM_REDIS_SSL  = os.getenv("STREAM_REDIS_SSL", "false").lower() == "true"

SOFT_LIMIT = int(os.getenv("SESSION_SOFT_TOKENS", "2500"))
HARD_LIMIT = int(os.getenv("SESSION_HARD_TOKENS", "3000"))
DUMMY_DELAY = int(os.getenv("DUMMY_SUMMARY_DELAY_SECONDS", "3"))

print(f"Running with SOFT_LIMIT {SOFT_LIMIT} and hard limmit {HARD_LIMIT} ultima version. with dummy delay {DUMMY_DELAY}")

def continuum_id(uid: int) -> str:
    return f"u{uid}:continuum"






LOG_PAYLOADS = os.getenv("LOG_PAYLOADS", "1") == "1"  # set to 0 to avoid logging text

def _snippet(text: str, n: int = 300) -> str:
    if not text: return ""
    text = text.replace("\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")

async def log_event_fs(user_id: int, event: str, data: dict | None = None):
    """Durable breadcrumb trail per user."""
    await db.collection("continuums").document(str(user_id)) \
        .collection("events").add({
            "event": event,
            "data": data or {},
            "ts": firestore.SERVER_TIMESTAMP,
        })


async def _fetch_pets(user_id: int) -> dict:
    try:
        return await get_all_pet_details_by_user_id(user_id) or {}
    except Exception as e:
        logger.warning("pet_fetch_failed", extra={"user_id": user_id, "err": str(e)})
        return {}
# Helper to log usage from Firestore control-plane
async def log_session_usage(user_id: int, session_id: str, note: str = ""):
    cont_ref = db.collection("continuums").document(str(user_id))
    snap = await cont_ref.get()
    d = snap.to_dict() or {}
    logger.info(
        "session_usage",
        extra={
            "user_id": user_id,
            "note": note,
            "session_id": session_id,                 # ADK session you’re using
            "status": d.get("status"),
            "generation": d.get("generation"),
            "usage_tokens": d.get("usage_tokens", 0),
            "soft_limit_tokens": d.get("soft_limit_tokens", SOFT_LIMIT),
            "hard_limit_tokens": d.get("hard_limit_tokens", HARD_LIMIT),
        },
    )
async def ensure_active_session_or_restore(user_id: int) -> str:
    """Durable source of truth in Firestore; recreate ADK session if missing."""
    cont_ref = db.collection("continuums").document(str(user_id))
    conv_ref = db.collection("conversations").document(continuum_id(user_id))

    snap = await cont_ref.get()
    data = snap.to_dict() or {}
    sid = data.get("active_session_id")

    if not sid:
        # First time for this user: seed metadata (e.g., pets) and create session
         # First time for this user: fetch pets from DB ONLY
        init_state = {
            "info_mascotas": await _fetch_pets(user_id),
            "generation": 0,
        }
        sid = (await create_session(str(user_id), state=init_state))["id"]
        # Do NOT pass state here to avoid accidental overwrite of info_mascotas
        await ensure_session(sid, str(user_id))

        await cont_ref.set({
            "user_id": str(user_id),
            "active_session_id": sid,
            "status": "active",
            "generation": 0,
            "usage_tokens": 0,
            "soft_limit_tokens": SOFT_LIMIT,
            "hard_limit_tokens": HARD_LIMIT,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        await conv_ref.set({
            "user_id": str(user_id),
            "agent_session_id": sid,
            "subject": "Continuum",
            "created_at": firestore.SERVER_TIMESTAMP,
            "last_message_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        logger.info("session_created", extra={
        "user_id": user_id,
        "session_id": sid,
        "reason": "init"
         })
        
        await log_event_fs(user_id, "session_created", {"session_id": sid, "reason": "init"})
        return sid

    # Session id exists in control plane; make sure ADK recognizes it.
    try:
        await ensure_session(sid, str(user_id))
        return sid
    except Exception:
        # Recreate from carry_over (best-effort)
        carry_over = data.get("carry_over") or {}
        state_seed = {"generation": data.get("generation", 0)}
        if carry_over:
            state_seed["carry_over"] = carry_over

        # ↓ NEW: prefer cached pets from control-plane, fallback to DB
        pets = await _fetch_pets(user_id)
        state_seed["info_mascotas"] = pets

        new_sid = (await create_session(str(user_id), state=state_seed))["id"]
        await ensure_session(new_sid, str(user_id))
        logger.warning("session_recreated", extra={
                "user_id": user_id, "old_session_id": sid, "new_session_id": new_sid, "reason": "adk_missing"
            })
        
        await log_event_fs(user_id, "session_created", {
            "session_id": new_sid, "reason": "recreate_from_carry_over", "old_session_id": sid
        })
        await cont_ref.update({
            "active_session_id": new_sid,
            "status": "active",
            "usage_tokens": 0,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        await conv_ref.set({"agent_session_id": new_sid, "last_message_at": firestore.SERVER_TIMESTAMP}, merge=True)
        
        
        return new_sid


async def maybe_start_summarizer(user_id: int, add_tokens: int) -> None:
    cont_ref = db.collection("continuums").document(str(user_id))

    @firestore.async_transactional
    async def txn(tx):
        snap = await cont_ref.get(transaction=tx)
        d = snap.to_dict() or {}
        usage = int(d.get("usage_tokens", 0)) + int(add_tokens)
        tx.update(cont_ref, {"usage_tokens": usage, "updated_at": firestore.SERVER_TIMESTAMP})

        if d.get("status", "active") == "active" and usage >= int(d.get("soft_limit_tokens", SOFT_LIMIT)):
            tx.update(cont_ref, {
                "status": "summarizing",
                "summarize_started_at": firestore.SERVER_TIMESTAMP,
                "summarize_ready_at": None,
                "summarize_session_id": d.get("active_session_id"),
            })

    await txn(db.transaction())

    cur = (await cont_ref.get()).to_dict() or {}
    if cur.get("status") == "summarizing" and cur.get("summarize_ready_at") is None:
        await log_event_fs(user_id, "summarize_started", {
            "active_session_id": cur.get("active_session_id"),
            "summarize_session_id": cur.get("summarize_session_id"),
            "usage_tokens": cur.get("usage_tokens"),
            "soft_limit_tokens": cur.get("soft_limit_tokens"),
        })
        logger.info("summarize_started", extra={
            "user_id": user_id,
            "active_session_id": cur.get("active_session_id"),
            "summarize_session_id": cur.get("summarize_session_id"),
            "usage_tokens": cur.get("usage_tokens"),
            "soft_limit_tokens": cur.get("soft_limit_tokens"),
        })
        await summarizer_task.kiq(user_id=user_id, generation=int(cur.get("generation", 0)))


async def publish_status(ordering_key: str, event: str, **fields) -> None:
    """
    Send a named control event to Redis (awaited) so it arrives before chunks.
    event: 'status' | 'done' | 'error' (your choice)
    """
    if not redis_stream:
        return
    payload = {
        "event": event,
        "data": {**fields},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    # Await to guarantee ordering relative to upcoming chunk tasks
    await redis_stream.publish(ordering_key, json.dumps(payload))


# Only create the client if a host is configured — keeps code harmless
redis_stream: aioredis.Redis | None = None
if STREAM_REDIS_HOST:
    redis_stream = aioredis.Redis(
        host=STREAM_REDIS_HOST,
        port=STREAM_REDIS_PORT,
        ssl=STREAM_REDIS_SSL,
        encoding="utf-8",
        decode_responses=True,
    )
    logger.info("[Init] Redis-stream client ready (host=%s)", STREAM_REDIS_HOST)
else:
    logger.info("[Init] STREAM_REDIS_HOST not set → Redis streaming disabled")


# ------------------------------------------------------------------------------
#  Taskiq task
# ------------------------------------------------------------------------------
@broker.task
async def process_message_task(
    user_id: int,
    message: str,
    attachments: list[dict] | None = None,
) -> None:
    
    
   

    conv_id = continuum_id(user_id)
    convo_doc_ref = db.collection("conversations").document(conv_id)
    msg_ref = convo_doc_ref.collection("messages")
    cont_ref = db.collection("continuums").document(str(user_id))

       # 🔔 immediately tell the UI we’re thinking
    


    # 1) Ensure control-plane + ADK session
    session_id = await ensure_active_session_or_restore(user_id)
    await log_session_usage(user_id, session_id, note="before_stream")

    cont = (await cont_ref.get()).to_dict() or {}
    status = cont.get("status", "active")
    generation = int(cont.get("generation", 0))
    is_post_trigger = status in ("summarizing", "ready_to_rollover")

    # 3) Persist the user message to evergreen timeline immediately
    user_msg_id = str(uuid.uuid4())
    user_msg_doc = msg_ref.document(user_msg_id)

    await publish_status(
        conv_id, "status",
        phase="started",                 # or "thinking"
        user_id=str(user_id),
        user_message_id=user_msg_id,     # you’ll use this same id when you persist the message
    )

    await asyncio.sleep(5)

    await user_msg_doc.set({
        "timestamp": firestore.SERVER_TIMESTAMP,
        "role": "user",
        "content": message,
        "attachments": attachments or [],
        "type": "message",
        "generation": generation,
        "session_id": session_id,            # <<< ADD
        "post_trigger": is_post_trigger,
    })

    await convo_doc_ref.set({"last_message_at": firestore.SERVER_TIMESTAMP}, merge=True)

    # 4) If this message arrived after summarizer started, log it
    if is_post_trigger:
        payload = {"message_id": user_msg_id}   # ← use the explicit id
        if LOG_PAYLOADS:
            payload["content_snippet"] = _snippet(message)
        await log_event_fs(user_id, "post_trigger_enqueued", payload)
        logger.info("post_trigger_enqueued", extra={
            "user_id": user_id,
            "session_id": cont.get("active_session_id"),
            "status": status,
            "generation": generation,
            "message_id": user_msg_id,
            **({"content_snippet": _snippet(message)} if LOG_PAYLOADS else {})
        })

   

    sess_ref = db.collection("sessions").document(session_id)
    sess_snap = await sess_ref.get()

    if not sess_snap.exists:
        # Create once; sets start_at only on first write
        await sess_ref.create({
            "user_id": str(user_id),
            "status": "active",
            "start_at": firestore.SERVER_TIMESTAMP,
            "generation": generation,
        })
    else:
        # Update without touching start_at
        await sess_ref.set({
            "user_id": str(user_id),
            "status": "active",
            "generation": generation,
        }, merge=True)
    

    # 4) If summarizer finished → rollover BEFORE sending message to engine
    # 4) If summarizer finished → rollover BEFORE sending message to engine

    # after ensuring the session header doc (the .set on "sessions/{session_id}")
    cont = (await cont_ref.get()).to_dict() or {}
    status = cont.get("status", "active")
    generation = int(cont.get("generation", 0))

    if status == "ready_to_rollover":
        
        started_at = cont.get("summarize_started_at")
        if not started_at:
            logger.warning("ready_to_rollover without summarize_started_at; defaulting to epoch",
                        extra={"user_id": user_id, "session_id": session_id})
            started_at = datetime.fromtimestamp(0, tz=timezone.utc)
        

        tail_query = (
            msg_ref
            .where(filter=FieldFilter("timestamp", ">", started_at))
            .order_by("timestamp")
        )
        tail_docs = [d async for d in tail_query.stream()]

        tail, tail_ids = [], []
        for d in tail_docs:
            if d.id == user_msg_id:
                continue
            m = d.to_dict() or {}
            if m.get("type") == "message":
                tail_ids.append(d.id)
                tail.append({
                    "role": m["role"],
                    "content": m["content"],
                    "attachments": m.get("attachments", []),
                })                   


        tail_count = len(tail)
        carry_over = cont.get("carry_over") or {}
        # Log a preview so you can audit in logs easily
        tail_preview = []
        if LOG_PAYLOADS:
            for item in tail[:3]:
                tail_preview.append({"role": item["role"], "content_snippet": _snippet(item.get("content",""))})

        await log_event_fs(user_id, "rollover_prepared", {
            "old_session_id": session_id,
            "tail_count": tail_count,
            **({"tail_preview": tail_preview} if LOG_PAYLOADS else {}),
            **({"summary_snippet": _snippet(carry_over.get("summary","")),
                "notes_snippet": _snippet(((carry_over.get("memory_delta") or {}).get("notes","")))}
            if LOG_PAYLOADS else {})
        })
        logger.info("rollover_prepared", extra={
            "user_id": user_id,
            "old_session_id": session_id,
            "tail_count": tail_count,
            **({"tail_preview": tail_preview} if LOG_PAYLOADS else {}),
            **({"summary_snippet": _snippet(carry_over.get("summary","")),
                "notes_snippet": _snippet(((carry_over.get("memory_delta") or {}).get("notes","")))}
            if LOG_PAYLOADS else {})
        })

        # Create & ensure new ADK session seeded with summary + tail
        # ↓↓↓ ADD THESE TWO LINES (fresh fetch + optional cache) ↓↓↓
        pets = await _fetch_pets(user_id)

        # 🔒 Sanitize everything to make it JSON-safe (converts Firestore Timestamp / DatetimeWithNanoseconds → ISO strings)
        carry_over_safe = json_sanitize(carry_over)
        tail_safe       = json_sanitize(tail)
        pets_safe       = json_sanitize(pets)
        seed_state = {
            "carry_over": carry_over_safe,
            "generation": generation + 1,
            "tail": tail_safe,
            "info_mascotas": pets_safe,
        }

        new_sid = (await create_session(str(user_id), state=seed_state))["id"]
        await ensure_session(new_sid, str(user_id))

        # Optional: persist per-session docs for console inspection
        await db.collection("sessions").document(session_id).set({
            "user_id": str(user_id),
            "status": "closed",
            "end_at": firestore.SERVER_TIMESTAMP,
            "generation": generation,
        }, merge=True)

                # ⇢ timeline marker on OLD session
        await db.collection("sessions").document(session_id).collection("timeline").add({
            "ts": firestore.SERVER_TIMESTAMP,
            "kind": "session_closed",
            "generation": generation,           # ← old generation
        })

        await db.collection("sessions").document(new_sid).set({
            "user_id": str(user_id),
            "status": "active",
            "start_at": firestore.SERVER_TIMESTAMP,
            "generation": generation + 1,
            "seed": {
                "summary": carry_over.get("summary"),
                "memory_delta": carry_over.get("memory_delta"),
                "metadata": carry_over.get("metadata"),
                "tail_count": tail_count,
                "tail_ids": tail_ids,                      # <<< ADD (ordered)
                **({"tail_preview": tail_preview} if LOG_PAYLOADS else {})
            }
        }, merge=True)

        # ─────────────── INSERT THIS BLOCK HERE ───────────────
        batch = db.batch()
        sess_tl = db.collection("sessions").document(new_sid).collection("timeline")

        # session_opened
        batch.set(
            sess_tl.document(),
            {
                "ts": firestore.SERVER_TIMESTAMP,
                "kind": "session_opened",
                "generation": generation + 1,
                "seed": {
                    "has_summary": bool(carry_over.get("summary")),
                    "carry_over_count": tail_count,
                },
                "ord": 0,   # optional: preserves display order if ts ties
            },
        )

        # summary
        batch.set(
            sess_tl.document(),
            {
                "ts": firestore.SERVER_TIMESTAMP,
                "kind": "summary",
                "summary": carry_over.get("summary"),
                "memory_delta": carry_over.get("memory_delta"),
                "generation": generation + 1,
                "ord": 1,   # optional
            },
        )

        # carry_over_ref entries (preserve order)
        for i, mid in enumerate(tail_ids, start=2):
            batch.set(
                sess_tl.document(),
                {
                    "ts": firestore.SERVER_TIMESTAMP,
                    "kind": "carry_over_ref",
                    "message_id": mid,
                    "generation": generation + 1,
                    "ord": i,   # optional
                },
            )

        await batch.commit()

        # ───────────── END INSERTION ─────────────
       

        # Switch control-plane → new session
        await cont_ref.update({
            "active_session_id": new_sid,
            "status": "active",
            "generation": firestore.Increment(1),
            "usage_tokens": 0,
            "summarize_started_at": None,
            "summarize_ready_at": None,
            "summarize_session_id": None,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        await convo_doc_ref.set({"agent_session_id": new_sid}, merge=True)

        # UX marker and final rollover logs
        await msg_ref.add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "role": "system",
            "type": "marker",
            "marker": "rollover_complete",
            "generation": generation + 1,
        })
        await log_event_fs(user_id, "rollover_complete", {
            "old_session_id": session_id, "new_session_id": new_sid, "tail_count": tail_count
        })
        logger.info("rollover_complete", extra={
            "user_id": user_id,
            "old_session_id": session_id,
            "new_session_id": new_sid,
            "tail_count": tail_count
        })

        # Route THIS SAME user message to the NEW session
        session_id = new_sid
        generation += 1
    
    await user_msg_doc.update({                         # <<< ADD
    "session_id": session_id,                       # <<< ADD
    "generation": generation                        # <<< ADD
        })                                               # <<< ADD
    

    # ⇩ ADD THIS RIGHT HERE — timeline entry for the user message
    await db.collection("sessions").document(session_id).collection("timeline").add({
        "ts": firestore.SERVER_TIMESTAMP,
        "kind": "message_ref",
        "message_id": user_msg_id,
        "role": "user",
        "generation": generation,
    })

    # 5) Build parts from attachments (optional – keep your existing code)
    parts = []
    # (Load from GCS if you want inline_data; omitted here for brevity.)

    # 6) Stream to engine & publish chunks
    full_reply_parts: list[str] = []
    try:
        async for chunk, is_first in run_agent_stream(
            str(user_id), session_id, message, attachments=parts
        ):
            full_reply_parts.append(chunk)

            # Pub/Sub (ordering key = continuum id)
            await publish_non_blocking(
                publisher, TOPIC_PATH, ordering_key=conv_id,
                user_id=str(user_id), conversation_id=conv_id,
                chunk=chunk, message_id=str(uuid.uuid4())
            )
            # Redis (optional)
            publish_non_blocking_redis(conv_id, chunk)

    except Exception as exc:
        logger.error("Streaming/publish failure: %s", exc, exc_info=True)

    # 7) Persist assistant message
    full_reply = "".join(full_reply_parts)
    if full_reply:
        bot_msg_ref = msg_ref.document()  # auto-ID (or use uuid4 like the user message)
        await bot_msg_ref.set({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "role": "assistant",
            "content": full_reply,
            "attachments": [],
            "type": "message",
            "generation": generation,
            "session_id": session_id,
        })

        # timeline entry for the assistant message
        await db.collection("sessions").document(session_id).collection("timeline").add({
            "ts": firestore.SERVER_TIMESTAMP,
            "kind": "message_ref",
            "message_id": bot_msg_ref.id,
            "role": "assistant",
            "generation": generation,
        })


    # === NEW: log delta & totals ===
    # Snapshot BEFORE we add this reply’s tokens
    cont_before = (await cont_ref.get()).to_dict() or {}
    usage_before = int(cont_before.get("usage_tokens", 0))
    soft = int(cont_before.get("soft_limit_tokens", SOFT_LIMIT))
    hard = int(cont_before.get("hard_limit_tokens", HARD_LIMIT))

    # Heuristic usage update → possibly start summarizer (unchanged logic)
    approx_tokens = max(len(full_reply) // 4, 50)  # ~4 chars/token, 50 floor
    await maybe_start_summarizer(user_id, add_tokens=approx_tokens)

    # Snapshot AFTER update for an accurate “used vs limit”
    cont_after = (await cont_ref.get()).to_dict() or {}
    usage_after = int(cont_after.get("usage_tokens", usage_before))
    status_after = cont_after.get("status")

    # Emit two concise logs: delta & totals
    logger.info(
        "assistant_tokens_delta",
        extra={
            "user_id": user_id,
            "session_id": session_id,
            "delta_tokens": approx_tokens,
            "usage_before": usage_before,
            "usage_after": usage_after,
            "soft_limit_tokens": soft,
            "hard_limit_tokens": hard,
            "status_after": status_after,
        },
    )

    # (Optional) also re-log the “totals” for quick grepping
    await log_session_usage(user_id, session_id, note="after_usage_update")

# ------------------------------------------------------------------------------
#  Helper – publish chunks (Non-blocking)
# ------------------------------------------------------------------------------
async def publish_non_blocking(
    publisher: pubsub_v1.PublisherClient, topic_path: str, ordering_key: str,
    user_id: str, conversation_id: str, chunk: str, message_id: str,
) -> None:
    """
    Publishes a single chunk without waiting for the result.
    This is a "fire-and-forget" approach suitable for high-throughput streaming.
    The Google Cloud client library handles batching and sending in the background.
    """
    # --- MODIFICATION ---
    # We no longer `await` the result. We just send the request and move on.
    publisher.publish(                         # fire-and-forget
        topic_path, chunk.encode("utf-8"),
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        ordering_key=ordering_key,             # Pub/Sub-level ordering key
    )
    logger.debug("Queued chunk for publishing to convo '%s'", conversation_id)


# --------------------------------------------------------------------
# Helper – publish chunk to Redis  (fire-and-forget, no await)
# --------------------------------------------------------------------
def publish_non_blocking_redis(ordering_key: str, chunk: str) -> None:
    """
    Publishes the same chunk to Redis Pub/Sub **in parallel** with Cloud Pub/Sub.
    No-op if STREAM_REDIS_* env-vars are not set.
    """
    if not redis_stream:
        return   # Redis disabled
    # Schedule the I/O outside the caller's await chain
    asyncio.create_task(redis_stream.publish(ordering_key, chunk))
    logger.info("Queued chunk for Redis publish to convo '%s'", ordering_key)
    logger.info("🔼 Redis publish queued  key=%s  data=%s…",
                 ordering_key, chunk[:40].replace("\n", " "))
    

##### summarizer helpers ####
import os
from datetime import datetime, date, timezone
from google.cloud.firestore_v1.base_query import FieldFilter  # query fix

# --- Summarizer config (env-driven) -------------------------------------------
# Default to a fast, current model; override via env as needed.
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gemini-2.0-flash-001")
SUMMARIZER_MAX_INPUT_TOKENS = int(os.getenv("SUMMARIZER_MAX_INPUT_TOKENS", "8000"))

# Developer API key (Gemini Developer API). For Vertex, set GOOGLE_GENAI_USE_VERTEXAI=true
# and GCP_PROJECT_ID/GCP_LOCATION; leave this empty.
SUMMARIZER_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def _approx_tokens(s: str) -> int:
    # quick-and-dirty; swap with a real tokenizer when ready
    return max(1, len(s) // 4)


def _to_notes(facts) -> str:
    """Turn list[str] or list[dict] into a bullet list string."""
    if not facts:
        return ""
    if isinstance(facts, list):
        out = []
        for f in facts:
            if isinstance(f, str):
                out.append(f"- " + f.strip())
            elif isinstance(f, dict):
                if "text" in f:
                    out.append(f"- " + str(f["text"]).strip())
                else:
                    ev = " / ".join(str(f.get(k)) for k in ("entity", "key", "value") if f.get(k))
                    if ev:
                        out.append("- " + ev)
        return "\n".join(out)
    return str(facts)


def _ts_iso(ts):
    if ts is None:
        return None
    try:
        return ts.isoformat()
    except Exception:
        return str(ts)


def json_sanitize(obj):
    """
    Recursively convert Firestore/Datetime objects to JSON-friendly values.
    - Datetime/Date/DatetimeWithNanoseconds -> ISO 8601 strings
    - Tuples -> lists
    """
    try:
        from google.api_core.datetime_helpers import DatetimeWithNanoseconds
        DTWNS = DatetimeWithNanoseconds
    except Exception:
        DTWNS = datetime  # minimal duck type fallback

    if isinstance(obj, (datetime, date, DTWNS)):
        return _ts_iso(obj)
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_sanitize(v) for v in obj]
    return obj


@broker.task
async def summarizer_task(user_id: int, generation: int):
    """
    Single-pass Gemini summarization (google-genai SDK):
      - Build a bounded transcript up to summarize_started_at
      - Ask for JSON: {summary: str, facts: [str]}
      - Persist as carry_over {summary, memory_delta.notes}
      - Mark ready_to_rollover
    """
    import re, json, asyncio

    cont_ref = db.collection("continuums").document(str(user_id))
    conv_id = f"u{user_id}:continuum"
    msg_ref = db.collection("conversations").document(conv_id).collection("messages")

    # ---- Marker: summarization started ---------------------------------------
    await msg_ref.add({
        "timestamp": firestore.SERVER_TIMESTAMP,
        "role": "system",
        "type": "marker",
        "marker": "summarize_started",
        "generation": generation
    })

    # Read summarize_started_at (set when SOFT limit tripped)
    cont_doc = (await cont_ref.get()).to_dict() or {}
    started_at = cont_doc.get("summarize_started_at")
    if not started_at:
        # Fallback: now (should be rare)
        started_at = datetime.now(timezone.utc)
        logger.warning("summarizer_task: missing summarize_started_at; defaulting to now",
                       extra={"user_id": user_id})

    # ---- Collect messages COVERAGE: everything <= started_at ------------------
    # (Keeps your later 'tail' clean: tail is strictly > started_at)
    # Query uses FieldFilter to avoid composite index; we filter type='message' in Python.
    query = (
        msg_ref
        .where(filter=FieldFilter("timestamp", "<=", started_at))
        .order_by("timestamp")
    )
    docs = [d async for d in query.stream()]

    # Keep only real chat messages
    msgs = []
    for d in docs:
        m = d.to_dict() or {}
        if m.get("type") != "message":
            continue
        msgs.append((d.id, m))

    # Build a token-bounded transcript, newest-last
    lines, used_ids, first_ts, last_ts = [], [], None, None
    budget = SUMMARIZER_MAX_INPUT_TOKENS
    cur = 0
    for doc_id, m in msgs:
        ts = m.get("timestamp")
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        line = f"[{doc_id}] {ts.isoformat() if hasattr(ts, 'isoformat') else ts} | {role.upper()}: {content}"
        t = _approx_tokens(line)
        if cur + t > budget and lines:
            break
        lines.append(line)
        used_ids.append(doc_id)
        cur += t
        if not first_ts:
            first_ts = ts
        last_ts = ts

    transcript = "\n".join(lines)
    input_chars = len(transcript)

    # ---- Call Gemini once (google-genai) -------------------------------------
    try:
        from google import genai
        from google.genai import types

        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
        if use_vertex:
            client = genai.Client(
                vertexai=True,
                project=os.getenv("GCP_PROJECT_ID"),
                location=os.getenv("GCP_LOCATION", "us-central1"),
            )
        else:
            client = genai.Client(api_key=SUMMARIZER_API_KEY)

        PROMPT = """You are a careful assistant that summarizes chat transcripts.
Return ONLY JSON with this schema:
{"summary": "5-10 sentences covering key points, decisions, and user intents",
 "facts": ["bullet-level concrete facts & preferences; concise; no speculation"]}
Rules:
- Do NOT invent facts; stick to the transcript verbatim.
- Prefer durable details (preferences, entities, decisions, numbers, dates).
- Keep it neutral and concise.
- If content is thin, return shorter output (empty 'facts' is OK).
TRANSCRIPT:
"""

        def _call():
            resp = client.models.generate_content(
                model=SUMMARIZER_MODEL,
                contents=[PROMPT, transcript],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "facts": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["summary", "facts"],
                        "additionalProperties": False,
                    },
                    max_output_tokens=1024,
                    temperature=0.2,
                ),
            )
            return resp.text or ""

        raw = await asyncio.to_thread(_call)

        # ---- Parse JSON defensively ------------------------------------------
        data = {}
        try:
            data = json.loads(raw)
        except Exception:
            # Try to snip JSON from fences or extra prose
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    pass

        summary = (data.get("summary") or "").strip() if isinstance(data, dict) else ""
        facts = data.get("facts") if isinstance(data, dict) else None

        if not summary and not facts:
            # Fallback: very small heuristic summary
            head = (lines[0] if lines else "")[:300]
            tail = (lines[-1] if lines else "")[:300]
            summary = f"Conversation recap:\n- Opening: {head}\n- Latest: {tail}"
            facts = []

        notes = _to_notes(facts)

        # ---- Persist carry_over (JSON-safe) and flip status -------------------
        await cont_ref.update({
            "carry_over": json_sanitize({
                "summary": summary,
                "memory_delta": {"notes": notes},
                "metadata": {
                    "model": SUMMARIZER_MODEL,
                    "input_characters": input_chars,
                    "used_message_count": len(used_ids),
                    "coverage": {
                        "from_ts": _ts_iso(first_ts),
                        "to_ts": _ts_iso(last_ts),
                        "message_ids": used_ids,
                    },
                    "prompt_version": "v1-single-pass-genai",
                }
            }),
            "summarize_ready_at": firestore.SERVER_TIMESTAMP,
            "status": "ready_to_rollover",
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

        # Marker + logs
        await msg_ref.add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "role": "system",
            "type": "marker",
            "marker": "summarize_ready",
            "generation": generation
        })

        payload = {}
        if LOG_PAYLOADS:
            payload = {
                "summary_snippet": _snippet(summary),
                "notes_snippet": _snippet(notes),
                "used_message_count": len(used_ids)
            }
        await log_event_fs(user_id, "summarize_ready", payload)

        logger.info("summarize_ready", extra={
            "user_id": user_id,
            "generation": generation,
            "used_message_count": len(used_ids),
            **({"summary_snippet": _snippet(summary),
                "notes_snippet": _snippet(notes)} if LOG_PAYLOADS else {})
        })

    except Exception as e:
        logger.error("summarizer_task failed, falling back to dummy summary: %s", e, exc_info=True)
        # Safe fallback so your pipeline still rolls forward
        dummy_summary = "[automatic summary unavailable; using fallback]"
        dummy_notes = ""
        await cont_ref.update({
            "carry_over": {
                "summary": dummy_summary,
                "memory_delta": {"notes": dummy_notes},
                "metadata": {"error": str(e), "prompt_version": "v1-single-pass-genai", "fallback": True}
            },
            "summarize_ready_at": firestore.SERVER_TIMESTAMP,
            "status": "ready_to_rollover",
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        await msg_ref.add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "role": "system",
            "type": "marker",
            "marker": "summarize_ready",
            "generation": generation
        })
        await log_event_fs(user_id, "summarize_ready", {
            **({"summary_snippet": _snippet(dummy_summary)} if LOG_PAYLOADS else {})
        })
