# ==============================================================================
# File: relay_redis.py  –  SSE + Redis Pub/Sub (user_id & conversation_id in path)
# Purpose:  ▸ Whenever the first browser connects for a conversation
#           ▸ SUBSCRIBE to that Redis channel (`chat:<conversation_id>`)
#           ▸ Push every token chunk to the SSE client(s)
#           ▸ If the last client disconnects, UNSUBSCRIBE + cancel task
# ==============================================================================

import os, logging, asyncio, janus
from typing import Dict, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
import redis.asyncio as aioredis      # ← NEW (async Redis client)
from dotenv import load_dotenv
load_dotenv(override=True)

# ── CONFIG ─────────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", 8001))

STREAM_REDIS_HOST = os.getenv("STREAM_REDIS_HOST", "localhost")
STREAM_REDIS_PORT = int(os.getenv("STREAM_REDIS_PORT", "6379"))
STREAM_REDIS_SSL  = os.getenv("STREAM_REDIS_SSL", "false").lower() == "true"

# ── LOGGING ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][%(levelname)s][relay-redis] %(message)s")
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


@app.on_event("startup")
async def _startup():
    log.info("🚀 relay-redis startup  (Redis=%s:%s SSL=%s)",
             STREAM_REDIS_HOST, STREAM_REDIS_PORT, STREAM_REDIS_SSL)
    try:
        pong = await redis_stream.ping()
        log.info("Redis PING -> %s", pong)
    except Exception as e:
        log.error("Redis PING FAILED: %s", e, exc_info=True)

# ── GLOBAL REDIS CLIENT (shared) ───────────────────────────────────────
redis_stream = aioredis.Redis(
    host=STREAM_REDIS_HOST,
    port=STREAM_REDIS_PORT,
    ssl=STREAM_REDIS_SSL,
    encoding="utf-8",
    decode_responses=True,
)

# ── REGISTRIES ─────────────────────────────────────────────────────────
#  conversation_id ➜ (janus.Queue, owning_event_loop)
CLIENTS:   Dict[str, Tuple[janus.Queue, asyncio.AbstractEventLoop]] = {}
#  conversation_id ➜ background Redis listener task
SUB_TASKS: Dict[str, asyncio.Task] = {}

# ── SSE ENDPOINT ───────────────────────────────────────────────────────
@app.get("/stream/{user_id}/{conversation_id}")
async def stream(user_id: str, conversation_id: str):
    """
    – When the first browser calls this endpoint we:
        * create a janus.Queue
        * register it in CLIENTS
        * start (lazily) a Redis SUBSCRIBE task for that conversation
    – When the last browser disconnects we:
        * remove the queue
        * cancel the Redis task (which auto-UNSUBSCRIBEs)
    """
    q    = janus.Queue()
    loop = asyncio.get_running_loop()
    CLIENTS[conversation_id] = (q, loop)

    # ── LAZY SUBSCRIBE – start once per conversation ───────────────────
    if conversation_id not in SUB_TASKS:
        async def fanout():
            pubsub = redis_stream.pubsub()
            await pubsub.subscribe(conversation_id)
            log.info("🔗 SUBSCRIBE chat:%s  (task started)", conversation_id)  # add this
            log.info("🔗 SUBSCRIBE chat:%s", conversation_id)
            try:
                async for msg in pubsub.listen():
                    if msg["type"] != "message":
                        continue
                    data = msg["data"]
                    pair = CLIENTS.get(conversation_id)
                    if pair:
                        pair[0].async_q.put_nowait(data)
            except asyncio.CancelledError:
                pass
            finally:
                await pubsub.unsubscribe(conversation_id)
                await pubsub.close()
                log.info("⏏︎ UNSUB chat:%s", conversation_id)

        SUB_TASKS[conversation_id] = asyncio.create_task(fanout())

    log.info("👋 attach  conv=%s user=%s", conversation_id, user_id)

    async def event_gen():
        try:
            while True:
                chunk = await q.async_q.get()
                yield {"data": chunk}
        finally:
            # Browser closed SSE connection
            CLIENTS.pop(conversation_id, None)
            await q.aclose()
            # If nobody else is listening → stop Redis task
            if conversation_id not in CLIENTS and conversation_id in SUB_TASKS:
                SUB_TASKS.pop(conversation_id).cancel()
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
