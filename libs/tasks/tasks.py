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

# --- ADK / Vertex --------------------------------------------------------------
# from adk_client import (
#     runner,
#     run_config,
#     session_service,
# )

from common.adk_client import ensure_session, run_agent_stream, create_session
import httpx




from google.cloud import storage
from google.genai import types   # ADK uses Part/Blob underneath

# ------------------------------------------------------------------------------
#  Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][Worker] %(message)s",
)
logger = logging.getLogger(__name__)



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
    conversation_id: str,
    message: str,
    attachments: list[dict] | None = None,   #  ← NEW
) -> None:
    """Handle one inbound user message with performance logging."""
    task_start_time = time.perf_counter()
    logger.info("Processing message for user '%s' in conversation '%s'", user_id, conversation_id)
    
    convo_doc_ref = db.collection("conversations").document(conversation_id)
    
    session_id = None
    
    @firestore.async_transactional
    async def get_or_create_convo(
        transaction: firestore.AsyncTransaction,  # ← must stay
    ) -> None:
        """
        Make sure a conversation document exists and that `session_id` is valid
        on the ADK backend.  If Firestore has a stale session-id (e.g. ADK DB
        was reset), we transparently create a new session and patch the doc.
        """
        nonlocal session_id
        start_t = time.perf_counter()
        logger.debug("[TXN] ⇢ enter  tx=%s", id(transaction))

        try:
            # ── 1️⃣  Read conversation doc (starts the transaction) ────────────
            snapshot = await convo_doc_ref.get(transaction=transaction)
            logger.debug("[TXN] snapshot.exists=%s  (%.4fs)",
                        snapshot.exists, time.perf_counter() - start_t)

            # ───────────────────── Existing conversation ─────────────────────
            if snapshot.exists:
                session_id = snapshot.get("agent_session_id")
                logger.debug("[TXN] loaded session_id=%s", session_id)

                if session_id:
                    t0 = time.perf_counter()
                    try:
                        await ensure_session(session_id, str(user_id))
                    # 400 / 404 from the ADK service → session missing
                    except (httpx.HTTPStatusError, RuntimeError) as e:
                        code = getattr(e, "response", None) and e.response.status_code
                        if code in (400, 404) or isinstance(e, RuntimeError):
                            logger.warning(
                                "[ensure_session] %s → creating new session for convo '%s'",
                                code, conversation_id
                            )
                            session_id = (await create_session(str(user_id)))["id"]
                            # update Firestore with the fresh id
                            transaction.update(
                                convo_doc_ref, {"agent_session_id": session_id}
                            )
                        else:
                            raise
                    logger.debug("[PERF] ensure_session/create_session %.4fs",
                                time.perf_counter() - t0)

                # always bump the last-message timestamp
                transaction.update(
                    convo_doc_ref,
                    {"last_message_at": datetime.now(timezone.utc)},
                )
                logger.debug("[TXN] last_message_at patched")

            # ───────────────────── New conversation ─────────────────────
            else:
                logger.info("New conversation '%s' – creating doc + agent session",
                            conversation_id)

                # fetch pet data (external)
                t0 = time.perf_counter()
                pets = await get_all_pet_details_by_user_id(user_id)
                logger.debug("[PERF] get_all_pet_details_by_user_id %.4fs",
                            time.perf_counter() - t0)

                init_state = {"info_mascotas": pets or {}}

                # create ADK session
                t0 = time.perf_counter()
                session_id = (await create_session(str(user_id), state=init_state))["id"]
                logger.debug("[PERF] create_session %.4fs", time.perf_counter() - t0)

                # write Firestore doc
                transaction.set(
                    convo_doc_ref,
                    {
                        "user_id": str(user_id),
                        "agent_session_id": session_id,
                        "subject": "New Conversation",
                        "created_at": datetime.now(timezone.utc),
                        "last_message_at": datetime.now(timezone.utc),
                    },
                )
                logger.debug("[TXN] conversation doc created")

        except Exception as exc:
            logger.error("🚨 get_or_create_convo inner failure: %s", exc, exc_info=True)
            raise  # Firestore will roll back / retry

        finally:
            logger.debug("[TXN] ⇠ exit  (%.4fs)", time.perf_counter() - start_t)
    
    try:
        get_convo_start = time.perf_counter()
        await get_or_create_convo(db.transaction())
        #await get_or_create_convo()
        get_convo_end = time.perf_counter()
        logger.info("[PERF] Get/Create conversation took: %.4f seconds", get_convo_end - get_convo_start)
    except Exception as e:
        # 👇 THIS IS THE ONLY CHANGE YOU NEED TO MAKE
        logger.error(
            "Transaction to get/create conversation failed. Task will not proceed: %s",
            e,
            exc_info=True
        )
        return

    if not session_id:
        logger.error("Failed to get or create a session ID. Aborting task.")
        return

    persist_user_msg_start = time.perf_counter()
    try:
        user_msg_ref = convo_doc_ref.collection("messages").document()
        await user_msg_ref.set({
            "timestamp": datetime.now(timezone.utc),
            "role": "user",
            "content": message,
            "attachments": attachments or [],
        })
    except Exception as exc:
        logger.error("Firestore write (user msg) failed: %s", exc, exc_info=True)
    persist_user_msg_end = time.perf_counter()
    logger.info("[PERF] Persisting user message took: %.4f seconds", persist_user_msg_end - persist_user_msg_start)

    # ─────────────────────────── Load attachments ────────────────────────────
    def _fix_mime(att):
        mt = att.get("mime_type") or mimetypes.guess_type(att.get("file_name",""))[0]
        return mt or "image/jpeg"
    parts: list[types.Part] = []
    parts = []
    if attachments:
        for att in attachments:
            try:
                blob = gcs.bucket(att["bucket"]).blob(att["object_path"])
                await asyncio.to_thread(blob.reload)  # HEAD to get size, content_type
                data = await asyncio.to_thread(blob.download_as_bytes)
                mt = _fix_mime(att)
                logger.info("ATT ✓ %s bytes=%d mime=%s name=%s",
                            att["object_path"], len(data), mt, att.get("file_name"))
                parts.append(types.Part.from_bytes(data=data, mime_type=mt))
            except Exception as e:
                logger.error("ATT ✗ %s/%s: %s", att["bucket"], att["object_path"], e)

    stream_agent_start = time.perf_counter()
    message_id = str(uuid.uuid4())
    full_reply_parts: list[str] = []
    first_token_time = None


    

    try:
       

    
       async for chunk, is_first in run_agent_stream(
            str(user_id),
            session_id,
            message,
            attachments=parts,              #  ← NEW
        ):
            if is_first:
                first_token_time = time.perf_counter()
                logger.info("[PERF] Time to first token: %.4f seconds", first_token_time - stream_agent_start)
            logger.info("PUBLISH → key=%s data=%s…",
            conversation_id, chunk[:40].replace("\n", " "))
            # --- Pub/Sub path (unchanged) ---------------------------
            await publish_non_blocking(
                publisher, TOPIC_PATH, ordering_key=conversation_id,
                user_id=str(user_id), conversation_id=conversation_id,
                chunk=chunk, message_id=message_id,
            )
            
            # --- Redis path (new, fire-and-forget) -------------------
            print(f"Just before publishing to redis")
            publish_non_blocking_redis(
                conversation_id, chunk,
            )
            print(f"just after publishing to redis")


            full_reply_parts.append(chunk)

    except Exception as exc:
        logger.error("Streaming/publish failure: %s", exc, exc_info=True)
    
    stream_agent_end = time.perf_counter()
    logger.info("[PERF] Agent streaming and publishing took: %.4f seconds", stream_agent_end - stream_agent_start)

    persist_bot_msg_start = time.perf_counter()
    full_reply = "".join(full_reply_parts)
    if full_reply:
        try:
            bot_msg_ref = convo_doc_ref.collection("messages").document()
            await bot_msg_ref.set({"timestamp": datetime.now(timezone.utc), "role": "bot", "content": full_reply})
        except Exception as exc:
            logger.error("Firestore write (bot msg) failed: %s", exc, exc_info=True)
    persist_bot_msg_end = time.perf_counter()
    logger.info("[PERF] Persisting bot message took: %.4f seconds", persist_bot_msg_end - persist_bot_msg_start)

    task_end_time = time.perf_counter()
    logger.info("[PERF] TOTAL task duration for conversation '%s': %.4f seconds", conversation_id, task_end_time - task_start_time)

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
    


