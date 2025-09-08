# ────────────────────────────────────────────────────────────────────────
# Dockerfile.vet_chat_worker – Cloud Run image for the Vet-Chat TaskIQ worker
# ────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Build deps (for cryptography, grpc, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Put shared libs on PYTHONPATH (same pattern as your other workers)
COPY libs/common /opt/libs/common
COPY libs/vet_chat_tasks /opt/libs/vet_chat_tasks
ENV PYTHONPATH="/opt/libs:${PYTHONPATH}"

# Dependencies (pin however you like in your repo)
COPY apps/vet_chat_worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# API wrapper that spawns the TaskIQ worker process
COPY apps/vet_chat_worker/worker_api.py ./worker_api.py

EXPOSE 8080
CMD ["uvicorn", "worker_api:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
