# common/broker.py
import os, inspect
from taskiq_redis import RedisStreamBroker, RedisAsyncResultBackend
from common.redis_conn import REDIS_URL

CHAT_QUEUE = os.getenv("TASKIQ_CHAT_QUEUE", "q:chat")
VET_QUEUE  = os.getenv("TASKIQ_VET_QUEUE", "q:vet")
VET_CHAT_QUEUE = os.getenv("TASKIQ_VET_CHAT_QUEUE", "q:vet_chat")

def _log(label, url, q):
    print(f"[BrokerInit] {label}: file={inspect.getsourcefile(RedisStreamBroker)} redis={url} queue={q}")

# Primary chat broker
_log("CHAT", REDIS_URL, CHAT_QUEUE)
broker = RedisStreamBroker(url=REDIS_URL, queue_name=CHAT_QUEUE).with_result_backend(
    RedisAsyncResultBackend(REDIS_URL, prefix_str="result:chat")
)

# Vet broker
_log("VET", REDIS_URL, VET_QUEUE)
vet_broker = RedisStreamBroker(url=REDIS_URL, queue_name=VET_QUEUE).with_result_backend(
    RedisAsyncResultBackend(REDIS_URL, prefix_str="result:vet")
)

# Vet-chat broker (new)
_log("VET_CHAT", REDIS_URL, VET_CHAT_QUEUE)
vet_chat_broker = RedisStreamBroker(url=REDIS_URL, queue_name=VET_CHAT_QUEUE).with_result_backend(
    RedisAsyncResultBackend(REDIS_URL, prefix_str="result:vet_chat")
)
