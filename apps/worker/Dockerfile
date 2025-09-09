# ────────────────────────────────────────────────────────────────────────
# Dockerfile.worker  –  Cloud-Run-ready image for the TaskIQ worker
# ────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Add ffmpeg here (keep image small with --no-install-recommends)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# shared libs on PYTHONPATH
COPY libs/common /opt/libs/common
COPY libs/tasks  /opt/libs/tasks
ENV PYTHONPATH="/opt/libs:${PYTHONPATH}"

# worker deps
COPY apps/worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# worker code
COPY apps/worker/worker_app.py ./worker_app.py

# (optional) sanity check during build – fails fast if ffmpeg missing
RUN ffmpeg -hide_banner -version

EXPOSE 8080
CMD ["uvicorn", "worker_app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
