# utils.py
import os
import asyncio
import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_env():
    from dotenv import load_dotenv
    load_dotenv()

def safe_json_load(s: str):
    import re, json
    # try to extract JSON substring
    match = re.search(r'(\{.*\}|\[.*\])', s, re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None

# Retry decorator for async functions
def async_retry(max_attempts: int = 3):
    def deco(fn):
        @retry(stop=stop_after_attempt(max_attempts), wait=wait_exponential(multiplier=1, min=1, max=10),
               retry=retry_if_exception_type(Exception), reraise=True)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)
        return wrapper
    return deco
