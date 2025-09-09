# tasks/cancel_helpers.py
# Cooperative cancel helpers for the chat worker (Postgres clone + cutoff).
import os, json, asyncio, contextlib, logging, time, uuid
from dataclasses import dataclass
from typing import Callable, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta

from google.cloud import firestore

# Postgres (psycopg 3.x)
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# ── env/config ────────────────────────────────────────────────────────────────
ADK_SESSION_BASE_URL      = os.getenv("ADK_SESSION_BASE_URL")      # ← Postgres DSN
APP_NAME                  = os.getenv("APP_NAME", "pet_parent_agent")
CANCEL_STICKY_TTL_SECONDS = int(os.getenv("CANCEL_STICKY_TTL_SECONDS", "3600"))
CANCEL_GC_DELAY_SECONDS   = int(os.getenv("CANCEL_GC_DELAY_SECONDS", "600"))

# Clone behavior toggles
INCLUDE_PARTIALS = os.getenv("CANCEL_CLONE_INCLUDE_PARTIALS", "false").lower() == "true"
DROP_ERRORS      = os.getenv("CANCEL_CLONE_DROP_ERRORS", "true").lower() == "true"
DROP_INTERRUPTED = os.getenv("CANCEL_CLONE_DROP_INTERRUPTED", "true").lower() == "true"
USE_CUTOFF_TS    = os.getenv("CANCEL_CLONE_CUTOFF_AT_CANCEL", "true").lower() == "true"

# ── small domain exception ────────────────────────────────────────────────────
class UserCancelled(Exception):
    """Raised to short-circuit the streaming loop when a cancel is requested."""
    pass

# ── lightweight DI for the worker’s resources ────────────────────────────────
@dataclass
class CancelCtx:
    db: firestore.AsyncClient
    redis: Optional[Any]                         # aioredis.Redis | None
    publish_status: Callable[..., asyncio.Future]
    log_event: Callable[[int, str, dict | None], asyncio.Future]
    logger: logging.Logger
    continuum_id: Callable[[int], str]

# ── internal helpers ─────────────────────────────────────────────────────────
def _control_channel(ctx: CancelCtx, user_id: int) -> str:
    return f"{ctx.continuum_id(user_id)}:control"

def _cancel_flags(ctx: CancelCtx, user_id: int, turn_id: Optional[str]) -> list[str]:
    conv = ctx.continuum_id(user_id)
    keys = [f"{conv}:cancelled:any"]
    if turn_id:
        keys.insert(0, f"{conv}:cancelled:{turn_id}")
    return keys

def _new_session_id() -> str:
    # Match ADK client format: YYYYMMDD-HHMMSS-ffffff (UTC)
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")

def _normalize_pg_ts(dt_obj: Optional[datetime]) -> Optional[datetime]:
    """
    Ensure a naive datetime suitable for Postgres TIMESTAMP (no tz).
    """
    if dt_obj is None:
        return None
    if dt_obj.tzinfo is not None:
        return dt_obj.astimezone(timezone.utc).replace(tzinfo=None)
    return dt_obj

# ── public: build a cancel event for this turn ───────────────────────────────
async def make_cancel_event(ctx: CancelCtx, user_id: int, turn_id: Optional[str]):
    ev = asyncio.Event()
    if not ctx.redis:
        return ev

    channel = _control_channel(ctx, user_id)
    keys    = _cancel_flags(ctx, user_id, turn_id)
    pubsub  = ctx.redis.pubsub()
    await pubsub.subscribe(channel)

    # sticky check
    try:
        for k in keys:
            if await ctx.redis.get(k):
                ev.set()
                with contextlib.suppress(Exception):
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                return ev
    except Exception:
        pass

    async def _listener():
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
                        ctx.logger.info("cancel_event_set", extra={
                            "user_id": user_id, "turn_id": turn_id, "channel": channel
                        })
                        ev.set()
                        break
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(channel)
                await pubsub.close()

    asyncio.create_task(_listener())
    return ev

# ── Postgres utilities --------------------------------------------------------
def _pg_get_last_user_ts_sync(
    *, dsn: str, app_name: str, user_id: str, session_id: str
) -> Optional[datetime]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT MAX(timestamp) AS ts
                FROM events
                WHERE session_id = %s AND app_name = %s AND user_id = %s AND author = 'user'
            """, (session_id, app_name, user_id))
            row = cur.fetchone()
            return row["ts"] if row else None

async def _pg_get_last_user_ts(
    *, dsn: str, app_name: str, user_id: str, session_id: str
) -> Optional[datetime]:
    return await asyncio.to_thread(
        _pg_get_last_user_ts_sync,
        dsn=dsn, app_name=app_name, user_id=user_id, session_id=session_id
    )

def _pg_mark_cancelled_turn_sync(
    *, dsn: str, app_name: str, user_id: str, session_id: str, user_msg_id: Optional[str]
) -> int:
    """
    OPTIONAL: Mark assistant rows in the open turn as interrupted (non-destructive).
    Not strictly required if we use a strict cutoff, but helpful for audit.
    Returns number of rows updated.
    """
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MAX(timestamp) AS ts
                    FROM events
                    WHERE session_id = %s AND app_name = %s AND user_id = %s AND author = 'user'
                """, (session_id, app_name, user_id))
                row = cur.fetchone()
                last_user_ts = row["ts"] if row else None

                if last_user_ts:
                    cur.execute("""
                        UPDATE events
                           SET interrupted = TRUE
                         WHERE session_id = %s AND app_name = %s AND user_id = %s
                           AND author = 'assistant'
                           AND (turn_complete IS DISTINCT FROM TRUE)
                           AND timestamp >= %s
                    """, (session_id, app_name, user_id, last_user_ts))
                else:
                    cur.execute("""
                        UPDATE events
                           SET interrupted = TRUE
                         WHERE session_id = %s AND app_name = %s AND user_id = %s
                           AND author = 'assistant'
                           AND (turn_complete IS DISTINCT FROM TRUE)
                    """, (session_id, app_name, user_id))
                affected = cur.rowcount

                # Insert a cancel marker for audit (system event)
                cur.execute("""
                    INSERT INTO events (
                      id, app_name, user_id, session_id, invocation_id, author, branch,
                      timestamp, content, actions, long_running_tool_ids_json,
                      grounding_metadata, partial, turn_complete, error_code,
                      error_message, interrupted
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    uuid.uuid4().hex, app_name, user_id, session_id,
                    None, "system", None,
                    Jsonb({"event": "cancel", "user_message_id": user_msg_id}),
                    None, None,
                    Jsonb({}), False, False, None, None, True
                ))
                return affected

async def _pg_mark_cancelled_turn(
    *, dsn: str, app_name: str, user_id: str, session_id: str, user_msg_id: Optional[str]
) -> int:
    return await asyncio.to_thread(
        _pg_mark_cancelled_turn_sync,
        dsn=dsn, app_name=app_name, user_id=user_id,
        session_id=session_id, user_msg_id=user_msg_id,
    )

# ── Postgres cloning (sync) with cutoff --------------------------------------
def _pg_clone_session_sync(
    *,
    dsn: str,
    app_name: str,
    user_id: str,
    old_session_id: str,
    include_partials: bool = False,
    drop_errors: bool = True,
    drop_interrupted: bool = True,
    cutoff_ts: Optional[datetime] = None,
    new_session_id_override: Optional[str] = None,
) -> tuple[str, int, float]:
    """
    Clone session + events in Postgres (blocking).
    Returns: (new_sid, copied_events_count, elapsed_ms)
    """
    t0 = time.perf_counter()
    new_sid = new_session_id_override or _new_session_id()
    cutoff_ts = _normalize_pg_ts(cutoff_ts)

    

    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # 1) Load source session (validate owner/app)
                cur.execute(
                    "SELECT app_name, user_id, state FROM sessions WHERE id = %s",
                    (old_session_id,)
                )
                src = cur.fetchone()
                if not src:
                    raise RuntimeError(f"Source session {old_session_id!r} not found")
                if str(src["app_name"]) != str(app_name) or str(src["user_id"]) != str(user_id):
                    raise RuntimeError("Session owner/app mismatch.")

                # 2) Create the new session row
                cur.execute("""
                    INSERT INTO sessions (id, app_name, user_id, state, create_time, update_time)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                """, (
                    new_sid, app_name, user_id,
                    Jsonb(src["state"]) if src["state"] is not None else None
                ))

                # 3) Select events to copy (ordered)
                where, params = ["session_id = %s"], [old_session_id]
                if not include_partials:
                    where.append("partial IS NOT TRUE")       # treat NULL as ok
                if drop_interrupted:
                    where.append("interrupted IS NOT TRUE")   # treat NULL as ok
                if drop_errors:
                    where.append("error_code IS NULL")
                if cutoff_ts and USE_CUTOFF_TS:
                    where.append("timestamp <= %s")
                    params.append(cutoff_ts)
                where_sql = " AND ".join(where)

                cur.execute(f"""
                    SELECT id, app_name, user_id, invocation_id, author, branch,
                           timestamp, content, actions, long_running_tool_ids_json,
                           grounding_metadata, partial, turn_complete, error_code,
                           error_message, interrupted
                    FROM events
                    WHERE {where_sql}
                    ORDER BY timestamp, id
                """, params)
                events = cur.fetchall()

                # 4) Insert cloned events with fresh ids and new session_id
                if events:
                    insert_sql = """
                        INSERT INTO events (
                          id, app_name, user_id, session_id, invocation_id, author, branch,
                          timestamp, content, actions, long_running_tool_ids_json,
                          grounding_metadata, partial, turn_complete, error_code,
                          error_message, interrupted
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """
                    batch = []
                    for e in events:
                        batch.append((
                            uuid.uuid4().hex,
                            e["app_name"], e["user_id"], new_sid,
                            e["invocation_id"], e["author"], e["branch"],
                            e["timestamp"],
                            Jsonb(e["content"]) if e["content"] is not None else None,
                            e["actions"],
                            e["long_running_tool_ids_json"],
                            Jsonb(e["grounding_metadata"]) if e["grounding_metadata"] is not None else None,
                            e["partial"], e["turn_complete"], e["error_code"],
                            e["error_message"], e["interrupted"]
                        ))
                    cur.executemany(insert_sql, batch)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return new_sid, (len(events) if events else 0), elapsed_ms

async def _pg_clone_session(
    *,
    dsn: str,
    app_name: str,
    user_id: str,
    old_session_id: str,
    include_partials: bool = False,
    drop_errors: bool = True,
    drop_interrupted: bool = True,
    cutoff_ts: Optional[datetime] = None,
    new_session_id_override: Optional[str] = None,
) -> tuple[str, int, float]:
    return await asyncio.to_thread(
        _pg_clone_session_sync,
        dsn=dsn,
        app_name=app_name,
        user_id=user_id,
        old_session_id=old_session_id,
        include_partials=include_partials,
        drop_errors=drop_errors,
        drop_interrupted=drop_interrupted,
        cutoff_ts=cutoff_ts,
        new_session_id_override=new_session_id_override,
    )

# ── clone ADK session & flip control plane (with explicit cutoff) ────────────
async def clone_and_swap_active_session(
    ctx: CancelCtx, user_id: int, old_sid: str, *, cutoff_ts: Optional[datetime] = None
) -> str:
    """
    Clones the ADK session in Postgres using a hard cutoff (timestamp <= cutoff_ts),
    then updates Firestore control-plane to point to the new session.
    """
    if not ADK_SESSION_BASE_URL:
        ctx.logger.error("cancel_pg_clone_failed", extra={
            "user_id": user_id, "old_sid": old_sid, "err": "ADK_SESSION_BASE_URL (DSN) is not set"
        })
        return old_sid

    cont_ref  = ctx.db.collection("continuums").document(str(user_id))
    convo_ref = ctx.db.collection("conversations").document(ctx.continuum_id(user_id))
    old_ref   = ctx.db.collection("sessions").document(old_sid)

    cont = (await cont_ref.get()).to_dict() or {}
    generation = int(cont.get("generation", 0))

    try:
        new_sid, copied, elapsed_ms = await _pg_clone_session(
            dsn=ADK_SESSION_BASE_URL,
            app_name=APP_NAME,
            user_id=str(user_id),
            old_session_id=old_sid,
            include_partials=INCLUDE_PARTIALS,
            drop_errors=DROP_ERRORS,
            drop_interrupted=DROP_INTERRUPTED,
            cutoff_ts=cutoff_ts,
        )
        ctx.logger.info("cancel_pg_clone_done", extra={
            "user_id": user_id, "old_sid": old_sid, "new_sid": new_sid,
            "copied_events": copied, "elapsed_ms": round(elapsed_ms, 2),
            "cutoff_ts": str(cutoff_ts)
        })
    except Exception as e:
        ctx.logger.error("cancel_pg_clone_failed", extra={
            "user_id": user_id, "old_sid": old_sid, "err": str(e), "cutoff_ts": str(cutoff_ts)
        })
        await ctx.log_event(user_id, "cancel_swap_noop", {"old_session_id": old_sid, "error": str(e)})
        return old_sid

    new_ref = ctx.db.collection("sessions").document(new_sid)

    # Flip control-plane pointers in Firestore
    batch = ctx.db.batch()
    batch.set(old_ref, {
        "user_id": str(user_id),
        "status": "closed",
        "end_at": firestore.SERVER_TIMESTAMP,
        "gc_after": datetime.now(timezone.utc) + timedelta(seconds=CANCEL_GC_DELAY_SECONDS),
        "gc_reason": "cancel_swap",
        "child_session_id": new_sid,
    }, merge=True)
    batch.set(new_ref, {
        "user_id": str(user_id),
        "status": "active",
        "start_at": firestore.SERVER_TIMESTAMP,
        "generation": generation,
        "seed": {"source": "cancel_pg_clone", "parent_session_id": old_sid},
    }, merge=True)
    batch.update(cont_ref, {
        "active_session_id": new_sid,
        "status": "active",
        "updated_at": firestore.SERVER_TIMESTAMP,
    })
    batch.set(convo_ref, {"agent_session_id": new_sid, "last_message_at": firestore.SERVER_TIMESTAMP}, merge=True)
    await batch.commit()

    await old_ref.collection("timeline").add({
        "ts": firestore.SERVER_TIMESTAMP, "kind": "session_closed",
        "reason": "cancel_swap", "generation": generation
    })
    await new_ref.collection("timeline").add({
        "ts": firestore.SERVER_TIMESTAMP, "kind": "session_opened",
        "source": "cancel_pg_clone", "generation": generation
    })

    await ctx.log_event(user_id, "cancel_swap", {
        "old_session_id": old_sid, "new_session_id": new_sid, "cutoff_ts": str(cutoff_ts)
    })
    ctx.logger.info("cancel_swap_done", extra={
        "user_id": user_id, "old_sid": old_sid, "new_sid": new_sid, "cutoff_ts": str(cutoff_ts)
    })
    return new_sid

# ── persist a partial message flagged as cancelled in Firestore  ─────────────
async def persist_partial_cancelled_reply(
    ctx: CancelCtx, *, msg_ref, session_id: str, generation: int, partial_text: Optional[str]
) -> Optional[str]:
    if not partial_text:
        return None
    bot_ref = msg_ref.document()
    await bot_ref.set({
        "timestamp": firestore.SERVER_TIMESTAMP,
        "role": "assistant",
        "content": partial_text,
        "attachments": [],
        "type": "message",
        "generation": generation,
        "session_id": session_id,
        "complete": False,
        "cancelled": True,
    })
    await ctx.db.collection("sessions").document(session_id).collection("timeline").add({
        "ts": firestore.SERVER_TIMESTAMP,
        "kind": "message_ref",
        "message_id": bot_ref.id,
        "role": "assistant",
        "generation": generation,
    })
    return bot_ref.id

# ── high-level: everything to do when a turn is cancelled ────────────────────
async def handle_cancelled(
    ctx: CancelCtx, *, user_id: int, conv_id: str, msg_ref, user_msg_id: str,
    session_id: str, generation: int, partial_text: Optional[str],
    worker_cutoff_ts: Optional[datetime] = None,  # ← NEW: pass from worker
) -> None:
    """
    worker_cutoff_ts should be the timestamp representing the start of this user turn
    (e.g., when the worker accepted the user message). We'll reconcile it with DB.
    """
    # 1) persist partial (if any)
    assistant_message_id = await persist_partial_cancelled_reply(
        ctx, msg_ref=msg_ref, session_id=session_id, generation=generation, partial_text=partial_text
    )

    # 2) tell UI this was cancelled
    await ctx.publish_status(
        conv_id, "status",
        phase="cancelled",
        user_id=str(user_id),
        user_message_id=user_msg_id,
        **({"assistant_message_id": assistant_message_id} if assistant_message_id else {}),
    )

    # 2a) OPTIONAL: mark in-flight assistant as interrupted (for audit only)
    try:
        updated = await _pg_mark_cancelled_turn(
            dsn=ADK_SESSION_BASE_URL,
            app_name=APP_NAME,
            user_id=str(user_id),
            session_id=session_id,
            user_msg_id=user_msg_id,
        )
        ctx.logger.info("cancel_pg_marked_interrupted", extra={
            "user_id": user_id, "old_sid": session_id, "updated_rows": int(updated)
        })
    except Exception as e:
        ctx.logger.warning("cancel_pg_mark_failed", extra={
            "user_id": user_id, "old_sid": session_id, "err": str(e)
        })

    # 2b) Determine effective cutoff:
    #  - Prefer the worker-provided cutoff
    #  - Reconcile with DB's last user ts (use the later of the two to ensure we KEEP the user event)
    db_last_user_ts = None
    try:
        db_last_user_ts = await _pg_get_last_user_ts(
            dsn=ADK_SESSION_BASE_URL,
            app_name=APP_NAME,
            user_id=str(user_id),
            session_id=session_id,
        )
    except Exception as e:
        ctx.logger.warning("cancel_pg_last_user_ts_failed", extra={
            "user_id": user_id, "session_id": session_id, "err": str(e)
        })

    worker_cutoff_norm = _normalize_pg_ts(worker_cutoff_ts)
    db_last_user_norm  = _normalize_pg_ts(db_last_user_ts)

    if worker_cutoff_norm and db_last_user_norm:
        # Use the later timestamp so we don't accidentally exclude the user event
        effective_cutoff = max(worker_cutoff_norm, db_last_user_norm)
    else:
        effective_cutoff = worker_cutoff_norm or db_last_user_norm

    ctx.logger.info("cancel_cutoff_resolved", extra={
        "user_id": user_id,
        "session_id": session_id,
        "worker_cutoff_ts": str(worker_cutoff_norm),
        "db_last_user_ts": str(db_last_user_norm),
        "effective_cutoff_ts": str(effective_cutoff)
    })

    # 3) clone in Postgres & swap session in Firestore (hard cutoff)
    new_sid = await clone_and_swap_active_session(ctx, user_id, session_id, cutoff_ts=effective_cutoff)
    cont_ref = ctx.db.collection("continuums").document(str(user_id))
    await cont_ref.update({
        "ignore_cancelled_turn": True,
        "ignore_cancelled_turn_user_msg_id": user_msg_id,
        "ignore_cancelled_turn_set_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })
    ctx.logger.info("cancel_ignore_flag_set", extra={
        "user_id": user_id,
        "old_sid": session_id,
        "new_sid": new_sid,
        "ignore_user_msg_id": user_msg_id,
    })

    # 4) clear sticky flags so next turn isn't insta-cancelled
    if ctx.redis:
        with contextlib.suppress(Exception):
            keys = list(dict.fromkeys(
                _cancel_flags(ctx, user_id, user_msg_id) + _cancel_flags(ctx, user_id, None)
            ))
            await ctx.redis.delete(*keys)

    # 5) final 'done'
    await ctx.publish_status(
        conv_id, "status",
        phase="done",
        user_id=str(user_id),
        user_message_id=user_msg_id,
        **({"assistant_message_id": assistant_message_id} if assistant_message_id else {}),
        new_session_id=new_sid,
    )
