# ───────────────────────────────────────────────────────────────
# Dockerfile.api  –  Chatbot REST gateway (FastAPI)
# ───────────────────────────────────────────────────────────────
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (grpc/cffi)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# 1) copy shared libs and set PYTHONPATH (build context = repo root)
COPY libs/common /opt/libs/common
COPY libs/tasks  /opt/libs/tasks
COPY libs/vet_tasks /opt/libs/vet_tasks

ENV PYTHONPATH="/opt/libs:${PYTHONPATH}"

# 2) app deps
COPY apps/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3) app code
COPY apps/api/api.py ./api.py

EXPOSE 8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
