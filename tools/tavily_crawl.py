import os
from typing import Any, Dict

import requests
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from config.settings import config
from logger import logger


_CLIENT = TavilyClient(api_key=config.tavily_api_key)

TAVILY_TIMEOUT_S = int(os.getenv("TAVILY_TIMEOUT_S", "60"))
TAVILY_RETRIES = int(os.getenv("TAVILY_RETRIES", "3"))


@retry(
    reraise=True,
    stop=stop_after_attempt(TAVILY_RETRIES),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type((TavilyTimeoutError, requests.exceptions.RequestException)),
)
def tavily_crawl(seed_url: str, *, max_depth: int = 2, limit: int = 15) -> Dict[str, Any]:
    """Tavily crawl wrapper with retries/timeouts/logging."""
    logger.info("TavilyCrawl | seed_url={} | max_depth={} | limit={}", seed_url, max_depth, limit)
    return _CLIENT.crawl(base_url=seed_url, max_depth=max_depth, limit=limit, timeout=TAVILY_TIMEOUT_S)


