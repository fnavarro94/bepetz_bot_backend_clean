# -----------------------------------------------------------------
# Dockerfile.relay  –  Redis‑to‑SSE relay service
# -----------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY apps/relay/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/relay/relay_redis.py ./relay_redis.py

EXPOSE 8080
CMD ["uvicorn", "relay_redis:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
