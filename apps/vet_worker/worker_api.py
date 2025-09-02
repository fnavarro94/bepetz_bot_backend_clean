# apps/vet_worker/worker_api.py
import os, subprocess, signal
from contextlib import asynccontextmanager
from fastapi import FastAPI

WORKER_CMD = [
    "taskiq",
    "worker",
    "vet_tasks:vet_broker",           # <— broker entrypoint in the module
    "--max-async-tasks", os.getenv("TASKIQ_MAX_ASYNC", "200"),
]

print("[vet-worker] starting:", " ".join(WORKER_CMD))

@asynccontextmanager
async def lifespan(app: FastAPI):
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

@app.get("/healthz")
def health():
    return {"status": "ok"}
