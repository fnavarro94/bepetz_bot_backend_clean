# worker_api.py
import subprocess, signal, os
from contextlib import asynccontextmanager

from fastapi import FastAPI

#
# Command that starts the TaskIQ worker exactly the way you do locally
#
WORKER_CMD: list[str] = [
    "taskiq",
    "worker",
    "tasks:broker",          # <-- your broker entry‑point
    "--max-async-tasks", "200",
    # add any other TaskIQ flags you need (prefetch, queues, etc.)
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cloud Run starts our container, Uvicorn binds to :8080,
    *and* we spawn TaskIQ as a **child process**.  
    When Cloud Run sends SIGTERM we shut the worker down cleanly.
    """
    worker = subprocess.Popen(WORKER_CMD)
    try:
        yield
    finally:
        # Forward Cloud Run’s SIGTERM to the child so no jobs are lost
        worker.send_signal(signal.SIGTERM)
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()

app = FastAPI(lifespan=lifespan)

@app.get("/healthz", include_in_schema=False)
def health() -> dict[str, str]:
    """Only one route is required so Cloud Run can probe the service."""
    return {"status": "ok"}
