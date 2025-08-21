# ──────────────────────────────────────────────────────────────────────────────
# File: adk_client.py             (Cloud-Run helper – robust + SSE streaming)
# Purpose:
#   • Provide Vertex-like helpers around an already-deployed ADK Reasoning
#     Engine that you fronted with Cloud-Run.
#   • Hide all auth / SSE plumbing behind three familiar calls:
#         create_session()          → returns {"id": "..."}
#         ensure_session()          → idempotent session put
#         run_agent_stream(...)     → async generator of (chunk, is_first)
#
#   The API mirrors the Vertex example you already have, so you can reuse the
#   same failure-recovery logic in your worker.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os, json, asyncio, logging, subprocess
from typing import AsyncGenerator, Tuple
from datetime import datetime, timezone
import httpx
import base64

# If you already imported this in tasks.py you can reuse; otherwise:
from google.genai import types     # Part / Blob convenience wrapper
from dotenv import load_dotenv

load_dotenv(override=True)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("TASKS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s][%(levelname)s][adk_client] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Mandatory ENV vars -- validate at import time
# ──────────────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("BASE_URL")           # e.g. https://my-agent-xxx.a.run.app
logger.info("BASE_URL IS  %s", BASE_URL)
APP_NAME = os.getenv("APP_NAME")           # e.g. pet_parent_agent
if not BASE_URL or not APP_NAME:
    raise RuntimeError("Missing required env vars: BASE_URL, APP_NAME")

RUN_URL     = f"{BASE_URL.rstrip('/')}/run_sse"
SESSION_URL = f"{BASE_URL.rstrip('/')}/apps/{APP_NAME}/users"

# ──────────────────────────────────────────────────────────────────────────────
# Cloud-Run auth – cached Bearer token
# ──────────────────────────────────────────────────────────────────────────────
_ID_TOKEN: str | None = os.getenv("ID_TOKEN")

async def _get_id_token() -> str:
    """
    Return an empty string.  We’re hitting a public Cloud-Run URL that
    doesn’t require authentication, so we skip gcloud / ID_TOKEN entirely.
    """
    return ""

async def _headers(sse: bool = False) -> dict[str, str]:
    base = {
        "Content-Type": "application/json",
        "User-Agent":   "adk-client/1.0",
    }
    if sse:                                 # only for /run_sse
        base.update({
            "Accept":          "text/event-stream",
            "Accept-Encoding": "identity",   # disable gzip buffering
        })
    return base

# ═════════════════════════════════════════════════════════════════════════════
# Public helper #1 – create_session  (parity with Vertex remote_app.create_session)
# ═════════════════════════════════════════════════════════════════════════════
async def create_session(
    user_id: str,
    *,
    state: dict | None = None,
) -> dict:
    """
    Fire-and-forget creation.  Returns the canonical session object from
    the server so the caller can grab the .id
    """
    

    new_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    url    = f"{SESSION_URL}/{user_id}/sessions/{new_id}"
    payload = {"state": {"created_at": datetime.now(timezone.utc).isoformat()},
               **(state or {})}
   
    logger.info("making request with url %s with headers %s and with payload %s", url, await _headers(), payload)
    async with httpx.AsyncClient(http2=True, timeout=30) as client:
        r = await client.post(url, headers=await _headers(), json=payload)

    if r.status_code not in (200, 201, 409):
        raise RuntimeError(f"create_session failed: {r.status_code} → {r.text[:400]}")
    return {"id": new_id}

# ═════════════════════════════════════════════════════════════════════════════
# Public helper #2 – ensure_session (idempotent PUT – safe to call every turn)
# ═════════════════════════════════════════════════════════════════════════════
async def ensure_session(
    session_id: str,
    user_id: str,
    state: dict | None = None,
) -> None:
    url = f"{SESSION_URL}/{user_id}/sessions/{session_id}"
    payload = {"state": {"created_at": datetime.now(timezone.utc).isoformat()}, **(state or {})}

    async with httpx.AsyncClient(http2=True, timeout=30) as client:
        r = await client.post(url, headers=await _headers(), json=payload)

        # Cloud-Run returns 400 if the session already exists (instead of 409)
    if r.status_code in (200, 201, 409):
        return
    if r.status_code == 400 and "Session already exists" in r.text:
        logger.debug("Session already existed – treated as OK.")
        return
    raise RuntimeError(f"ensure_session failed: {r.status_code} → {r.text[:400]}")

# ═════════════════════════════════════════════════════════════════════════════
# Public helper #3 – run_agent_stream  (async generator of token chunks)
# ═════════════════════════════════════════════════════════════════════════════
async def run_agent_stream(
     user_id: str,
     session_id: str,
     message: str,
    *,
    attachments: list[types.Part] | None = None,   # NEW
     max_tokens_per_chunk: int = 1,
 ) -> AsyncGenerator[Tuple[str, bool], None]:
    """
    Yields   (text_chunk, is_first)   exactly like Vertex runner.run_async().
    is_first lets your worker record TTFB easily.
    """
    # --------------------------------------------------------------
    # Convert attachments → JSON parts accepted by the ADK endpoint
    # --------------------------------------------------------------
    msg_parts: list[dict] = [{"text": message}]

    for p in attachments or []:
        inl = p.inline_data          # google.genai.types.Blob
        # Cloud-Run JSON must carry base64-encoded bytes
        msg_parts.append({
            "inline_data": {
                "mime_type": inl.mime_type,
                "data": base64.b64encode(inl.data).decode("ascii"),
            }
        })

    payload = {
        "app_name":   APP_NAME,
        "user_id":    user_id,
        "session_id": session_id,
        "new_message": {"role": "user", "parts": msg_parts},
        "streaming": True,
        "stream_options": {"max_tokens_per_chunk": max_tokens_per_chunk},
    }

    async with httpx.AsyncClient(http2=True, timeout=None) as client:
        async with client.stream("POST", RUN_URL, headers=await _headers(sse=True), json=payload) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise RuntimeError(f"run_sse failed ({r.status_code}): {body[:400]!r}")

            first = True
            async for raw_line in r.aiter_lines():
                if not raw_line.startswith("data:"):
                    continue

                
                data = raw_line[5:].strip()
                if data == '{"done":true}':
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("⚠️  Malformed SSE chunk: %s", data)
                    continue

                if not obj.get("partial", False):
                    continue

                # Tolerate multiple server builds
                logger.info("This is the stream object %s", obj )
                text = (
                    obj.get("text")                           # canonical
                    or obj.get("chunk")
                    or obj.get("delta", {}).get("text")       # Gemini style
                    or obj.get("content", {}).get("parts", [{}])[0].get("text")
                )
                if text is None:
                    continue
                
                yield text, first
                first = False

# ═════════════════════════════════════════════════════════════════════════════
__all__ = ["create_session", "ensure_session", "run_agent_stream"]
