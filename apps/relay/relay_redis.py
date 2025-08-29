# ==============================================================================
# File: relay_redis.py  –  SSE + Redis Pub/Sub (user_id & conversation_id in path)
# Purpose:  ▸ When the first browser connects for a conversation, SUBSCRIBE to
#             that Redis channel (`<conversation_id>`)
#           ▸ Fan-out each message to all connected SSE clients for that convo
#           ▸ When the last client disconnects, UNSUBSCRIBE + cancel task
# ==============================================================================

import os
import json
import logging
import asyncio
import janus
from typing import Dict, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv(override=True)

# ── CONFIG ─────────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", 8001))

STREAM_REDIS_HOST = os.getenv("STREAM_REDIS_HOST", "localhost")
STREAM_REDIS_PORT = int(os.getenv("STREAM_REDIS_PORT", "6379"))
STREAM_REDIS_SSL  = os.getenv("STREAM_REDIS_SSL", "false").lower() == "true"

# ── LOGGING ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][relay-redis] %(message)s"
)
log = logging.getLogger("relay-redis")

# ── FASTAPI APP ────────────────────────────────────────────────────────
app = FastAPI(title="Chat relay (Redis)")
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://bepetz-chatbot-ui-dev.web.app",
    # "https://bepetz-chatbot-ui-dev--felipe-preview.web.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ── GLOBAL REDIS CLIENT (shared) ───────────────────────────────────────
redis_stream = aioredis.Redis(
    host=STREAM_REDIS_HOST,
    port=STREAM_REDIS_PORT,
    ssl=STREAM_REDIS_SSL,
    encoding="utf-8",
    decode_responses=True,  # get strings instead of bytes
)

@app.on_event("startup")
async def _startup():
    log.info("🚀 relay-redis startup  (Redis=%s:%s SSL=%s)",
             STREAM_REDIS_HOST, STREAM_REDIS_PORT, STREAM_REDIS_SSL)
    try:
        pong = await redis_stream.ping()
        log.info("Redis PING -> %s", pong)
    except Exception as e:
        log.error("Redis PING FAILED: %s", e, exc_info=True)

# ── REGISTRIES ─────────────────────────────────────────────────────────
# conversation_id ➜ list of (janus.Queue, owning_event_loop)
CLIENTS: Dict[str, List[Tuple[janus.Queue, asyncio.AbstractEventLoop]]] = {}
# conversation_id ➜ background Redis listener task
SUB_TASKS: Dict[str, asyncio.Task] = {}

# ── SSE ENDPOINT ───────────────────────────────────────────────────────
@app.get("/stream/{user_id}/{conversation_id}")
async def stream(user_id: str, conversation_id: str):
    """
    For each connection:
      • Create a janus.Queue and register it under the conversation.
      • Lazily start the Redis SUBSCRIBE loop once per conversation.
      • Yield SSE events from the queue until the client disconnects.
    """
    q    = janus.Queue()
    loop = asyncio.get_running_loop()
    CLIENTS.setdefault(conversation_id, []).append((q, loop))

    # ── LAZY SUBSCRIBE – start once per conversation ───────────────────
    if conversation_id not in SUB_TASKS:
        async def fanout():
            pubsub = redis_stream.pubsub()
            await pubsub.subscribe(conversation_id)
            log.info("🔗 SUBSCRIBE %s  (task started)", conversation_id)
            try:
                async for msg in pubsub.listen():
                    if msg.get("type") != "message":
                        continue
                    data = msg.get("data")
                    # snapshot list to avoid mutation during iteration
                    for queue, _lp in list(CLIENTS.get(conversation_id, [])):
                        try:
                            queue.async_q.put_nowait(data)
                        except Exception:
                            # if queue is closed, ignore
                            pass
            except asyncio.CancelledError:
                pass
            finally:
                try:
                    await pubsub.unsubscribe(conversation_id)
                    await pubsub.close()
                finally:
                    log.info("⏏︎ UNSUB %s", conversation_id)

        SUB_TASKS[conversation_id] = asyncio.create_task(fanout())

    log.info("👋 attach  conv=%s user=%s", conversation_id, user_id)

    async def event_gen():
        try:
            while True:
                raw = await q.async_q.get()  # str (decode_responses=True)

                # 1) Map sentinel to a named 'done' event (and also keep raw for back-compat)
                if isinstance(raw, str) and raw.strip() == "[END-OF-STREAM]":
                    yield {"event": "done", "data": "{}"}
                    yield {"data": raw}
                    continue

                # 2) Interpret control envelopes as named SSE events:
                #    {"event": "status", "data": {...}}
                obj = None
                if isinstance(raw, str) and raw and raw[0] in "{[":
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        obj = None

                if isinstance(obj, dict) and "event" in obj:
                    evt = str(obj["event"])
                    payload = obj.get("data", {})
                    try:
                        payload_str = json.dumps(payload)
                    except Exception:
                        payload_str = "{}"
                    yield {"event": evt, "data": payload_str}
                    continue

                # 3) Fallback: treat as a normal token chunk
                yield {"data": raw}

        finally:
            # Browser closed SSE connection
            # Remove this client's queue
            clients = CLIENTS.get(conversation_id, [])
            try:
                clients.remove((q, loop))
            except ValueError:
                pass
            if not clients:
                CLIENTS.pop(conversation_id, None)
                # If nobody else is listening → stop Redis task
                task = SUB_TASKS.pop(conversation_id, None)
                if task:
                    task.cancel()
            await q.aclose()
            log.info("👋 detach  conv=%s user=%s", conversation_id, user_id)

    return EventSourceResponse(
        event_gen(),
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # helps prevent proxy buffering
        },
    )

# ── RUN ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info("🚀 relay-redis starting  (Redis=%s:%s SSL=%s)",
             STREAM_REDIS_HOST, STREAM_REDIS_PORT, STREAM_REDIS_SSL)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
