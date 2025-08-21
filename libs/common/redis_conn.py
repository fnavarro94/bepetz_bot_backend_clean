import os
import redis.asyncio as redis
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(override=True)

# --- The single source of truth for the Redis URL, for other modules to import ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")
print(f"REDIS_URL IS {REDIS_URL}")
# --- The Redis Singleton Connection Pool Instance ---
_redis_pool: redis.ConnectionPool | None = None

def get_redis_pool() -> redis.ConnectionPool:
    """
    Initializes and returns a singleton Redis connection pool.
    Using a connection pool is more efficient than creating a new connection
    for every request.
    """
    global _redis_pool
    if _redis_pool is None:
        print(f"Initializing Redis connection pool for URL: {REDIS_URL}")
        _redis_pool = redis.ConnectionPool.from_url(REDIS_URL)
    return _redis_pool

async def get_redis_connection() -> redis.Redis:
    """
    FastAPI dependency that provides a Redis connection from the pool.
    """
    pool = get_redis_pool()
    # redis-py's from_pool method is async safe and manages connections for you.
    return redis.Redis(connection_pool=pool)
