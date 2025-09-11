# worker.py

import os
import sys
import json
import time
import uuid
import logging
from typing import Optional, Dict, Any

import asyncio
import redis.asyncio as aioredis
from google.cloud import firestore

from common.broker import vet_chat_broker
from vet_chat_tasks.openai_helper import OpenAIChatHelper
import contextlib
# worker.py (top, with other imports)
from vet_chat_tasks.prompt import VET_SYSTEM_PROMPT

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
# Firestore (metadata + full conversation)
# ──────────────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB   = os.getenv("CHATS_FIRESTORE_DB")
db = firestore.AsyncClient(project=GCP_PROJECT_ID, database=FIRESTORE_DB)

CONSULTATIONS_COLL = os.getenv("VET_CHAT_COLLECTION", "vet_chat_consultations")

def _consult_ref(consultation_id: str):
    return db.collection(CONSULTATIONS_COLL).document(consultation_id)

def _messages_coll(consultation_id: str):
    # Subcollection to store the full conversation (ordered by created_at)
    return _consult_ref(consultation_id).collection("messages")

async def _ensure_thread_doc(consultation_id: str) -> None:
    ref = _consult_ref(consultation_id)
    snap = await ref.get()
    if not snap.exists:
        await ref.set({
            "consultation_id": consultation_id,
            # Legacy (unused) but kept for compatibility
            "openai_thread_id": None,
            # Responses-based continuity fields
            "openai_app_id": None,                 # our logical "app" container id
            "openai_app": None,                    # {"model","instructions","reasoning_effort",...}
            "openai_last_response_id": None,       # chain anchor for previous_response_id
            # Turn state (persisted)
            "turn_id": None,
            "turn_status": None,                   # "started" | "error" | "done" | "cancelled"
            "turn_started_at": None,
            "turn_completed_at": None,
            "turn_error_message": None,
            "turn_user_message_id": None,
            "turn_assistant_message_id": None,
            "turn_last_response_id": None,
            "turn_model": None,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "turn_updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

async def _touch_model(consultation_id: str, *, model: Optional[str]) -> None:
    await _consult_ref(consultation_id).set(
        {"model": model, "updated_at": firestore.SERVER_TIMESTAMP},
        merge=True,
    )

# Helper to persist a message in the subcollection
async def _persist_message(
    consultation_id: str,
    *,
    role: str,
    content: str,
    **extra: Any,
):
    data: Dict[str, Any] = {
        "role": role,                 # "user" | "assistant" | "system"
        "content": content,           # full text (not chunked)
        "created_at": firestore.SERVER_TIMESTAMP,
        **extra,
    }
    return await _messages_coll(consultation_id).add(data)

# ── Persisted turn-state helper ───────────────────────────────────────────────
async def _set_turn_state(consultation_id: str, *, status: Optional[str] = None, **fields: Any) -> None:
    """
    Merge-write the 'turn_*' fields on the consultation doc so the FE can
    recover state if SSE disconnects. Always bumps turn_updated_at.
    """
    payload: Dict[str, Any] = {**fields, "turn_updated_at": firestore.SERVER_TIMESTAMP}
    if status is not None:
        payload["turn_status"] = status
    await _consult_ref(consultation_id).set(payload, merge=True)

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


# Inline citation patch: FE places a link next to the supported text span.
async def _publish_citation_patch(
    consultation_id: str,
    *,
    start: int | None,
    end: int | None,
    url: str | None,
    title: str | None = None,
) -> None:
    if not redis_stream or not url:
        return
    ev = {"event": "citation", "data": {"start": start, "end": end, "url": url, "title": title}}
    await redis_stream.publish(_channel(consultation_id), json.dumps(ev))


# ──────────────────────────────────────────────────────────────────────────────
# Config knobs
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL            = os.getenv("VET_CHAT_MODEL", "gpt-5-mini")
DEFAULT_INSTRUCTIONS     = VET_SYSTEM_PROMPT
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
# Cancellation helpers (turn-scoped)
# ──────────────────────────────────────────────────────────────────────────────
class UserCancelled(Exception):
    pass

def _cancel_control_channel(consultation_id: str) -> str:
    return f"vet_chat:{consultation_id}:control"

def _cancel_flag_keys(consultation_id: str, turn_id: Optional[str]) -> list[str]:
    """
    Return the possible sticky keys that indicate cancellation.
    We check the specific turn first, then a generic 'any' key.
    """
    keys = []
    if turn_id:
        keys.append(f"vet_chat:{consultation_id}:cancelled:{turn_id}")
    keys.append(f"vet_chat:{consultation_id}:cancelled:any")
    return keys

async def _make_cancel_event_chat(consultation_id: str, turn_id: Optional[str]) -> asyncio.Event:
    ev = asyncio.Event()
    if not redis_stream:
        return ev

    channel = _cancel_control_channel(consultation_id)
    keys    = _cancel_flag_keys(consultation_id, turn_id)

    pubsub = redis_stream.pubsub()
    await pubsub.subscribe(channel)

    # 1) Check sticky flags first (coop-cancel if already requested)
    try:
        for k in keys:
            if await redis_stream.get(k):
                ev.set()
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                except Exception:
                    pass
                return ev
    except Exception:
        # ignore sticky check failure; proceed to live listening
        pass

    # 2) Live listen for control 'cancel' with matching turn_id (or generic)
    async def listener():
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                try:
                    data = json.loads(msg["data"])
                except Exception:
                    continue
                if data.get("event") == "cancel":
                    requested_turn = (data.get("data") or {}).get("turn_id")
                    if (turn_id and requested_turn == turn_id) or (requested_turn is None):
                        ev.set()
                        break
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(channel)
                await pubsub.close()

    asyncio.create_task(listener())
    return ev

async def _handle_chat_cancelled(consultation_id: str, *, partial_text: str | None, response_id: Optional[str]) -> None:
    """
    Persist turn_status='cancelled', optionally save a partial assistant message,
    DO NOT update openai_last_response_id (so the chain anchor stays at the
    previous completed response), and notify the stream channel.
    """
    assistant_doc_id = None

    if partial_text:
        try:
            # Save partial assistant message flagged as incomplete/cancelled
            doc_ref, _ = await _messages_coll(consultation_id).add({
                "role": "assistant",
                "content": partial_text,
                "created_at": firestore.SERVER_TIMESTAMP,
                "complete": False,
                "cancelled": True,
                "error": None,
                "response_id": response_id,
            })
            assistant_doc_id = getattr(doc_ref, "id", None)
        except Exception as e:
            logger.error("vet_chat_persist_partial_cancel_failed", extra={"consultation_id": consultation_id, "err": str(e)})

    # Mark the turn as cancelled
    try:
        await _set_turn_state(
            consultation_id,
            status="cancelled",
            turn_assistant_message_id=assistant_doc_id,
            turn_last_response_id=response_id,
            turn_completed_at=firestore.SERVER_TIMESTAMP,
        )
    except Exception as e:
        logger.error("vet_chat_set_turn_cancelled_failed", extra={"consultation_id": consultation_id, "err": str(e)})

    # Notify UI
    await _publish_status(consultation_id, phase="cancelled")

    # Best-effort: clear sticky keys so future turns aren't instantly cancelled
    if redis_stream:
        with contextlib.suppress(Exception):
            # Clear specific and generic flags (specific may 404 if turn has changed; that's fine)
            keys = _cancel_flag_keys(consultation_id, None)
            await redis_stream.delete(*keys)

# ──────────────────────────────────────────────────────────────────────────────
# TaskIQ task
# ──────────────────────────────────────────────────────────────────────────────
@vet_chat_broker.task
async def process_vet_chat_message_task(
    consultation_id: str,
    message: str,
    attachments: list[dict] | None = None,   # reserved for future use; persisted verbatim
    *,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    reasoning_effort: Optional[str] = None,  # GPT-5 only
    max_output_tokens: Optional[int] = None,
) -> None:
    """
    Stream a model response using the OpenAI Responses API, publish deltas over Redis SSE,
    and persist the FULL conversation (user + assistant messages) + TURN STATE in Firestore.

    Flow:
      1) Ensure a Firestore doc for this consultation (holds our "app" & chain id).
      2) Create a new turn_id and persist TURN STATE = 'started' (+ user message).
      3) Publish started/accepted statuses.
      4) Wait briefly for the FE relay 'ready' latch (avoid losing the first chunk).
      5) Stream deltas via Responses API; publish each chunk and accumulate buffer.
      6) Persist the ASSISTANT message with the full text and response.id anchor.
      7) Persist TURN STATE = 'done' (or 'error' on failure).
      8) Publish explicit EOS marker and final 'done' status.
    """
    await _ensure_thread_doc(consultation_id)

    model             = model or DEFAULT_MODEL
    system_prompt     = instructions if instructions is not None else DEFAULT_INSTRUCTIONS
    reasoning_effort  = reasoning_effort or DEFAULT_REASONING_EFFORT
    max_output_tokens = max_output_tokens if max_output_tokens is not None else DEFAULT_MAX_OUTPUT

    # touch metadata
    await _touch_model(consultation_id, model=model)

    channel = _channel(consultation_id)
    turn_id = f"turn_{uuid.uuid4().hex}"
    had_error = False
    was_cancelled = False

    # ── 2) Persist initial TURN STATE = started (before streaming)
    try:
        # also persist the USER message now (linked on the turn)
        user_doc_ref, _ = await _persist_message(
            consultation_id,
            role="user",
            content=message,
            attachments=attachments or [],
            # Optional fields for later retrieval/filters
            model=model,
            system_instructions=system_prompt,
        )
        await _set_turn_state(
            consultation_id,
            status="started",
            turn_id=turn_id,
            turn_started_at=firestore.SERVER_TIMESTAMP,
            turn_error_message=None,
            turn_completed_at=None,
            turn_user_message_id=getattr(user_doc_ref, "id", None),
            turn_assistant_message_id=None,
            turn_last_response_id=None,
            turn_model=model,
        )
    except Exception as persist_user_err:
        logger.error("vet_chat_persist_user_or_turn_failed", extra={
            "consultation_id": consultation_id,
            "err": str(persist_user_err),
        })
        # We still proceed, but note: FE may not see "started" if this write failed
        user_doc_ref = None

    # Build a cancel event scoped to this turn
    cancel_event = await _make_cancel_event_chat(consultation_id, turn_id)

    # SSE status
    await _publish_status(consultation_id, phase="started")

    try:
        await _publish_status(consultation_id, phase="accepted", model=model)

        # Optional: let the relay mark 'ready' after FE connects
        await _wait_until_ready(channel)

        # Load (or create) a logical "app" container per consultation to group config/ids
        doc_snap = await _consult_ref(consultation_id).get()
        doc = doc_snap.to_dict() if doc_snap.exists else {}

        app_id = doc.get("openai_app_id")
        if not app_id:
            app_id = f"app_{uuid.uuid4().hex}"
            await _consult_ref(consultation_id).set(
                {
                    "openai_app_id": app_id,
                    "openai_app": {
                        "model": model,
                        "instructions": system_prompt,
                        "reasoning_effort": reasoning_effort,
                        "created_at": firestore.SERVER_TIMESTAMP,
                    },
                    "updated_at": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
        else:
            # Optionally keep the stored app config fresh if env overrides change
            to_merge: Dict[str, Any] = {"updated_at": firestore.SERVER_TIMESTAMP}
            app = (doc.get("openai_app") or {}).copy()
            patch = {}
            if model and app.get("model") != model:
                patch["model"] = model
            if system_prompt is not None and app.get("instructions") != system_prompt:
                patch["instructions"] = system_prompt
            if reasoning_effort is not None and app.get("reasoning_effort") != reasoning_effort:
                patch["reasoning_effort"] = reasoning_effort
            if patch:
                to_merge["openai_app"] = {**app, **patch}
                await _consult_ref(consultation_id).set(to_merge, merge=True)

        prev_res_id = doc.get("openai_last_response_id")

        helper = OpenAIChatHelper(model=model)

        delta_count = 0
        last_log_t  = time.monotonic()
        assistant_text_chunks: list[str] = []

        # 5) Stream using Responses API (helper always uses Responses)
        # 5) Stream using Responses API (helper always uses Responses)
        citations_inline: list[dict] = []
        async def _publish_loop():
            nonlocal delta_count, last_log_t, was_cancelled
            for chunk in helper.stream_text(
                user_text=message,
                system_prompt=system_prompt,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                previous_response_id=prev_res_id,
                metadata={"consultation_id": consultation_id, "app_id": app_id, "turn_id": turn_id},
                store=True,
            ):
                # Cooperative cancel
                if cancel_event.is_set():
                    was_cancelled = True
                    try:
                        helper.cancel_current()
                    finally:
                        raise UserCancelled()

                if not chunk:
                    continue

                # ---- NEW: inline citation events (dict) ----
                if isinstance(chunk, dict) and chunk.get("event") == "citation":
                    await _publish_citation_patch(
                        consultation_id,
                        start=chunk.get("start"),
                        end=chunk.get("end"),
                        url=chunk.get("url"),
                        title=chunk.get("title"),
                    )
                    citations_inline.append({
                        "start": chunk.get("start"),
                        "end": chunk.get("end"),
                        "url": chunk.get("url"),
                        "title": chunk.get("title"),
                    })
                    continue

                # ---- Existing text streaming (str) ----
                if isinstance(chunk, str):
                    assistant_text_chunks.append(chunk)
                    delta_count += 1
                    await _publish_delta(consultation_id, chunk)

                    if LOG_STREAM:
                        now = time.monotonic()
                        if delta_count == 1 or (now - last_log_t) >= STREAM_LOG_HEARTBEAT_S:
                            logger.info(
                                "vet_chat_stream_delta",
                                extra={
                                    "consultation_id": consultation_id,
                                    "chunks": delta_count,
                                    "delta_len": len(chunk),
                                    "delta_snippet": chunk[:80],
                                },
                            )
                            last_log_t = now


        try:
            await _publish_loop()
        except UserCancelled:
            # Persist partial + mark 'cancelled'; DO NOT advance openai_last_response_id
            partial_text = "".join(assistant_text_chunks) if assistant_text_chunks else None
            await _handle_chat_cancelled(
                consultation_id,
                partial_text=partial_text,
                response_id=getattr(helper, "last_response_id", None),
            )
            # Short-circuit the normal completion path
            return

        # Consolidate assistant text
        assistant_full_text = "".join(assistant_text_chunks)

        # 6) Persist the ASSISTANT message and update chain + turn
        try:
            assistant_fields: Dict[str, Any] = {
                "reply_to": getattr(user_doc_ref, "id", None),
                "model": model,
                "app_id": app_id,
                "previous_response_id": prev_res_id,
                "response_id": getattr(helper, "last_response_id", None),
                "complete": True,
                "chunks": delta_count,
                "system_instructions": system_prompt,
                "citations_inline": citations_inline,  # <- optional
            }
            assistant_doc_ref, _ = await _persist_message(
                consultation_id,
                role="assistant",
                content=assistant_full_text,
                **assistant_fields,
            )

            # Update chain anchor on the consultation doc (only on successful completion)
            if getattr(helper, "last_response_id", None):
                await _consult_ref(consultation_id).set(
                    {"openai_last_response_id": helper.last_response_id, "updated_at": firestore.SERVER_TIMESTAMP},
                    merge=True,
                )

            # Persist turn 'done' with assistant message linkage
            await _set_turn_state(
                consultation_id,
                status="done",
                turn_assistant_message_id=getattr(assistant_doc_ref, "id", None),
                turn_last_response_id=getattr(helper, "last_response_id", None),
                turn_completed_at=firestore.SERVER_TIMESTAMP,
            )

        except Exception as persist_assistant_err:
            logger.error("vet_chat_persist_assistant_failed", extra={
                "consultation_id": consultation_id,
                "err": str(persist_assistant_err),
            })
            # In case assistant write failed but we finished streaming, still mark as done
            await _set_turn_state(
                consultation_id,
                status="done",
                turn_assistant_message_id=None,
                turn_last_response_id=getattr(helper, "last_response_id", None),
                turn_completed_at=firestore.SERVER_TIMESTAMP,
            )

        await _publish_status(consultation_id, phase="completed")
        logger.info("vet_chat_stream_completed", extra={"consultation_id": consultation_id, "chunks": delta_count})

    except Exception as e:
        had_error = True

        # Persist a partial assistant/system message if anything was generated
        try:
            partial_text = "".join(locals().get("assistant_text_chunks", []) or [])
            if partial_text:
                partial_ref, _ = await _persist_message(
                    consultation_id,
                    role="assistant",
                    content=partial_text,
                    complete=False,
                    error="stream_interrupted",
                    response_id=getattr(locals().get("helper", None), "last_response_id", None),
                )
                # Link partial assistant to turn (best-effort)
                await _set_turn_state(
                    consultation_id,
                    status="error",
                    turn_assistant_message_id=getattr(partial_ref, "id", None),
                    turn_last_response_id=getattr(locals().get("helper", None), "last_response_id", None),
                    turn_error_message=str(e),
                    turn_completed_at=firestore.SERVER_TIMESTAMP,
                )
            else:
                # No partial text captured
                await _set_turn_state(
                    consultation_id,
                    status="error",
                    turn_error_message=str(e),
                    turn_completed_at=firestore.SERVER_TIMESTAMP,
                )
        except Exception as persist_partial_err:
            logger.error("vet_chat_persist_partial_failed", extra={
                "consultation_id": consultation_id,
                "err": str(persist_partial_err),
            })
            # Even if message persistence fails, set the turn state to error
            try:
                await _set_turn_state(
                    consultation_id,
                    status="error",
                    turn_error_message=str(e),
                    turn_completed_at=firestore.SERVER_TIMESTAMP,
                )
            except Exception:
                pass

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

        # Make sure a successful run ends with 'done' turn state (don't override 'error' or 'cancelled')
        if not had_error and not was_cancelled:
            try:
                await _set_turn_state(
                    consultation_id,
                    status="done",
                    # If we didn't set completed_at earlier (e.g., if assistant persist failed),
                    # this will ensure there's a completion timestamp.
                    turn_completed_at=firestore.SERVER_TIMESTAMP,
                )
            except Exception as set_done_err:
                logger.error("vet_chat_set_done_failed", extra={
                    "consultation_id": consultation_id,
                    "err": str(set_done_err),
                })

        logger.info("vet_chat_stream_done", extra={"consultation_id": consultation_id})
