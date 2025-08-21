from taskiq_redis import RedisStreamBroker
from common.redis_conn import REDIS_URL  # updated import path

# This broker is the central point for sending and receiving tasks.
# It now correctly receives the URL string.
# We are using RedisStreamBroker as you correctly pointed out.
broker = RedisStreamBroker(
    url=REDIS_URL
)
