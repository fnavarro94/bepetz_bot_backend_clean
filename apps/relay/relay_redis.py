# ==============================================================================
# File: relay_redis.py  –  SSE + Redis Pub/Sub with instant open + ready latch
# Purpose:
#   ▸ Flush a first SSE frame immediately so the browser fires `onopen` fast
#   ▸ Keep connections warm with periodic pings
#   ▸ Maintain a per-channel "ready" flag in Redis while at least one client
#     is attached, so the API can wait briefly before publishing to avoid
#     lost-first-message races
# ==============================================================================

import os
import json
import logging
import asyncio
import janus
from typing import Dict, List, Tuple, Optional

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

# How long the "ready" key should live (refreshed periodically)
READY_TTL_SECONDS = int(os.getenv("READY_TTL_SECONDS", "15"))
# How often to refresh the ready TTL while at least one client is connected
READY_REFRESH_SECONDS = int(os.getenv("READY_REFRESH_SECONDS", "5"))
# Ping interval for SSE keepalive (comment frames)
SSE_PING_MS = int(os.getenv("SSE_PING_MS", "15000"))

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
# channel_name ➜ list of (janus.Queue, owning_event_loop)
CLIENTS: Dict[str, List[Tuple[janus.Queue, asyncio.AbstractEventLoop]]] = {}
# channel_name ➜ background Redis listener task (Pub/Sub)
SUB_TASKS: Dict[str, asyncio.Task] = {}
# channel_name ➜ background task that refreshes the "ready" TTL
READY_TASKS: Dict[str, asyncio.Task] = {}

# ── READY LATCH HELPERS ────────────────────────────────────────────────
def _ready_key(channel_name: str) -> str:
    # Channel names already include prefixes like "vet:<session_id>"
    return f"sse:ready:{channel_name}"

async def _set_ready(channel_name: str):
    try:
        await redis_stream.setex(_ready_key(channel_name), READY_TTL_SECONDS, "1")
    except Exception as e:
        log.warning("Failed to set ready key for %s: %s", channel_name, e)

async def _clear_ready(channel_name: str):
    try:
        await redis_stream.delete(_ready_key(channel_name))
    except Exception:
        pass

def _ensure_ready_refresher(channel_name: str):
    """Start/ensure a single refresher task per channel that keeps the ready key alive."""
    if channel_name in READY_TASKS:
        return

    async def refresher():
        try:
            # Initial mark as ready quickly
            await _set_ready(channel_name)
            while True:
                await asyncio.sleep(READY_REFRESH_SECONDS)
                # If no clients remain, stop refreshing
                if not CLIENTS.get(channel_name):
                    break
                await _set_ready(channel_name)
        except asyncio.CancelledError:
            pass
        finally:
            # If we exit, clear the key explicitly (or let TTL expire)
            await _clear_ready(channel_name)
            READY_TASKS.pop(channel_name, None)
            log.info("⏹ ready-refresher stopped for %s", channel_name)

    READY_TASKS[channel_name] = asyncio.create_task(refresher())
    log.info("▶ ready-refresher started for %s", channel_name)

# ── CORE SSE HANDLER ───────────────────────────────────────────────────
async def _sse_for_channel(channel_name: str, attach_label: str = ""):
    """
    Generic SSE streamer for any Redis Pub/Sub channel.
    - Registers client
    - Lazily starts a single Redis SUBSCRIBE loop per channel
    - Starts a single "ready" refresher per channel
    - Flushes an initial frame immediately so EventSource.onopen fires fast
    - Fans out messages to all attached clients
    """
    q    = janus.Queue()
    loop = asyncio.get_running_loop()
    CLIENTS.setdefault(channel_name, []).append((q, loop))

    # Start/reset the "ready" refresher (keeps the ready key alive)
    _ensure_ready_refresher(channel_name)

    # Start one background Redis listener per channel
    if channel_name not in SUB_TASKS:
        async def fanout():
            pubsub = redis_stream.pubsub()
            await pubsub.subscribe(channel_name)
            log.info("🔗 SUBSCRIBE %s (task started)", channel_name)
            try:
                async for msg in pubsub.listen():
                    if msg.get("type") != "message":
                        continue
                    data = msg.get("data")
                    # snapshot list to avoid mutation during iteration
                    for queue, _lp in list(CLIENTS.get(channel_name, [])):
                        try:
                            queue.async_q.put_nowait(data)
                        except Exception:
                            pass
            except asyncio.CancelledError:
                pass
            finally:
                try:
                    await pubsub.unsubscribe(channel_name)
                    await pubsub.close()
                finally:
                    log.info("⏏︎ UNSUB %s", channel_name)

        SUB_TASKS[channel_name] = asyncio.create_task(fanout())

    log.info("👋 attach  %s channel=%s", attach_label or "client", channel_name)

    async def event_gen():
        # --- FLUSH ASAP: yield a first tiny frame so the client "opens" instantly.
        # Also set a quick retry for clients.
        yield {"event": "ready", "data": "{}", "retry": 1000}
        # Maintain the ready TTL immediately (in case refresher hasn't ticked yet)
        await _set_ready(channel_name)

        try:
            while True:
                raw = await q.async_q.get()  # str (decode_responses=True)

                # Map sentinel used by chat flow
                if isinstance(raw, str) and raw.strip() == "[END-OF-STREAM]":
                    yield {"event": "done", "data": "{}"}
                    yield {"data": raw}
                    continue

                # Interpret control envelopes as named SSE events
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

                # Otherwise stream as plain chunk
                yield {"data": raw}
        finally:
            # Browser closed SSE connection – cleanup
            clients = CLIENTS.get(channel_name, [])
            try:
                clients.remove((q, loop))
            except ValueError:
                pass

            if not clients:
                CLIENTS.pop(channel_name, None)
                # Stop Redis listener if this was the last client
                task = SUB_TASKS.pop(channel_name, None)
                if task:
                    task.cancel()
                # Stop ready refresher if any
                rtask = READY_TASKS.pop(channel_name, None)
                if rtask:
                    rtask.cancel()
                # Also clear the ready key now
                await _clear_ready(channel_name)

            await q.aclose()
            log.info("👋 detach %s channel=%s", attach_label or "client", channel_name)

    return EventSourceResponse(
        event_gen(),
        # Starlette will set Content-Type to text/event-stream; we add useful headers.
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",      # helps prevent proxy buffering
            "Connection": "keep-alive",
        },
        ping=SSE_PING_MS,  # keepalive comment every N ms
    )

# ── SSE ENDPOINTS ──────────────────────────────────────────────────────
@app.get("/stream/{user_id}/{conversation_id}")
async def stream(user_id: str, conversation_id: str):
    """
    Multiparty generic stream keyed by conversation id.
    """
    channel = conversation_id
    return await _sse_for_channel(channel, attach_label=f"conv {conversation_id} user {user_id}")

# Worker publishes to:
#   • vet:{session_id}                         (all vet events)
#   • vet:{session_id}:{kind}                  (optional per-workflow channel)
# where kind ∈ {"diagnostics","additional_exams","prescription","complementary_treatments"}

@app.get("/vet-stream/{session_id}")
async def vet_stream(session_id: str):
    """
    Subscribe to all vet events for this session id.
    """
    channel = f"vet:{session_id}"
    return await _sse_for_channel(channel, attach_label=f"vet-session {session_id}")

@app.get("/vet-stream/{session_id}/{kind}")
async def vet_stream_kind(session_id: str, kind: str):
    """
    Subscribe to a specific vet workflow channel (e.g., diagnostics).
    Frontend can call: /vet-stream/{session_id}/diagnostics
    """
    channel = f"vet:{session_id}:{kind}"
    return await _sse_for_channel(channel, attach_label=f"vet-{kind} {session_id}")

# ── OPTIONAL: READY CHECK ENDPOINTS (useful if API won't touch Redis directly) ──
@app.get("/ready/vet/{session_id}")
async def ready_vet(session_id: str):
    """
    Returns {"ready": true/false} indicating if any client is attached to vet:{session_id}.
    """
    channel = f"vet:{session_id}"
    val = await redis_stream.get(_ready_key(channel))
    return {"ready": bool(val)}

@app.get("/ready/conv/{conversation_id}")
async def ready_conv(conversation_id: str):
    """
    Returns {"ready": true/false} indicating if any client is attached to {conversation_id}.
    """
    channel = conversation_id
    val = await redis_stream.get(_ready_key(channel))
    return {"ready": bool(val)}


# relay_redis.py  (add near the other routes)

@app.get("/vet-chat-stream/{consultation_id}")
async def vet_chat_stream(consultation_id: str):
    """
    Subscribe to the vet-chat stream for a consultation.
    Worker publishes to Redis channel: vet_chat:{consultation_id}
    Frontend connects with:  GET /vet-chat-stream/{consultation_id}
    """
    channel = f"vet_chat:{consultation_id}"
    return await _sse_for_channel(channel, attach_label=f"vet-chat {consultation_id}")

@app.get("/ready/vet-chat/{consultation_id}")
async def ready_vet_chat(consultation_id: str):
    """
    Returns {"ready": true/false} if any client is attached to vet_chat:{consultation_id}.
    Useful for the API/worker to avoid lost-first-chunk races.
    """
    channel = f"vet_chat:{consultation_id}"
    val = await redis_stream.get(_ready_key(channel))
    return {"ready": bool(val)}


# ── RUN ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info("🚀 relay-redis starting  (Redis=%s:%s SSL=%s)",
             STREAM_REDIS_HOST, STREAM_REDIS_PORT, STREAM_REDIS_SSL)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
