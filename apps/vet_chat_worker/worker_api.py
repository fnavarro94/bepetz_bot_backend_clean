# apps/vet_chat_worker/worker_api.py
import os, subprocess, signal
from contextlib import asynccontextmanager
from fastapi import FastAPI

# TaskIQ worker command:
#   - module:  vet_chat_tasks.vet_chat_tasks
#   - broker:  vet_chat_broker   (imported there from common.broker)
WORKER_CMD = [
    "taskiq",
    "worker",
    "vet_chat_tasks.vet_chat_tasks:vet_chat_broker",
    "--max-async-tasks", os.getenv("TASKIQ_MAX_ASYNC", "200"),
]

# Optional: you can append flags from env (e.g., prefetch, extra queues)
EXTRA_FLAGS = os.getenv("TASKIQ_EXTRA_FLAGS", "")
if EXTRA_FLAGS:
    WORKER_CMD.extend(EXTRA_FLAGS.split())

print("[vet-chat-worker] starting:", " ".join(WORKER_CMD))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cloud Run starts Uvicorn (:8080) and we spawn TaskIQ as a child process.
    On SIGTERM we forward the signal and give it time to drain cleanly.
    """
    p = subprocess.Popen(WORKER_CMD)
    try:
        yield
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()

app = FastAPI(lifespan=lifespan)

@app.get("/healthz", include_in_schema=False)
def health():
    return {"status": "ok"}
