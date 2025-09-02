import os
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv(override=True)

# Chat/default pipeline
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
print(f"[redis_conn] REDIS_URL = {REDIS_URL}")

# Vet pipeline (separate DB index)
VET_REDIS_URL = os.getenv("VET_REDIS_URL", "redis://redis:6379/0")
print(f"[redis_conn] VET_REDIS_URL = {VET_REDIS_URL}")

# ---- connection pools (optional for app usage) -----------------------
_redis_pool: redis.ConnectionPool | None = None
_vet_redis_pool: redis.ConnectionPool | None = None

def get_redis_pool() -> redis.ConnectionPool:
    global _redis_pool
    if _redis_pool is None:
        print(f"[redis_conn] init pool for {REDIS_URL}")
        _redis_pool = redis.ConnectionPool.from_url(REDIS_URL)
    return _redis_pool

def get_vet_redis_pool() -> redis.ConnectionPool:
    global _vet_redis_pool
    if _vet_redis_pool is None:
        print(f"[redis_conn] init VET pool for {VET_REDIS_URL}")
        _vet_redis_pool = redis.ConnectionPool.from_url(VET_REDIS_URL)
    return _vet_redis_pool

async def get_redis_connection() -> redis.Redis:
    return redis.Redis(connection_pool=get_redis_pool())

async def get_vet_redis_connection() -> redis.Redis:
    return redis.Redis(connection_pool=get_vet_redis_pool())
