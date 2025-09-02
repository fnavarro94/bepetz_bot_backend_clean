# ────────────────────────────────────────────────────────────────────────
# Dockerfile.vetworker – Cloud Run image for the Vet TaskIQ worker
# ────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Put shared libs on PYTHONPATH (same as your current worker image)
COPY libs/common /opt/libs/common
COPY libs/vet_tasks  /opt/libs/vet_tasks
ENV PYTHONPATH="/opt/libs:${PYTHONPATH}"

# Dependencies for the worker
# (reuse or split requirements as you prefer; key libs include:
#  fastapi, uvicorn, taskiq, redis, google-cloud-firestore, google-auth)
COPY apps/vet_worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# API wrapper that spawns the TaskIQ worker process
COPY apps/vet_worker/worker_api.py ./worker_api.py

EXPOSE 8080
CMD ["uvicorn", "worker_api:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
