import redis


# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

try:
    redis_client.ping()
    logging.info("Connected to Redis")
except redis.ConnectionError as e:
    logging.error(f"Redis connection error: {e}")