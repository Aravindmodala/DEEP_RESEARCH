import os
from typing import Any, Dict, List

import requests
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from config.settings import config
from logger import logger


_CLIENT = TavilyClient(api_key=config.tavily_api_key)

TAVILY_TIMEOUT_S = int(os.getenv("TAVILY_TIMEOUT_S", "60"))
TAVILY_RETRIES = int(os.getenv("TAVILY_RETRIES", "3"))
TAVILY_EXTRACT_DEPTH = os.getenv("TAVILY_EXTRACT_DEPTH", "advanced")
TAVILY_EXTRACT_CHUNK = int(os.getenv("TAVILY_EXTRACT_CHUNK", "2"))


def _chunks(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


@retry(
    reraise=True,
    stop=stop_after_attempt(TAVILY_RETRIES),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type((TavilyTimeoutError, requests.exceptions.RequestException)),
)
def _extract_once(urls: List[str], *, extract_depth: str) -> Dict[str, Any]:
    return _CLIENT.extract(urls=urls, extract_depth=extract_depth, timeout=TAVILY_TIMEOUT_S)


def tavily_extract(urls: List[str], *, extract_depth: str | None = None) -> Dict[str, Any]:
    """Tavily extract wrapper with chunking + retries/timeouts/logging."""
    extract_depth = extract_depth if extract_depth is not None else TAVILY_EXTRACT_DEPTH
    urls = [u for u in urls if u]
    logger.info("TavilyExtract | depth={} | urls_count={}", extract_depth, len(urls))

    results: List[Any] = []
    failed: List[Any] = []
    for chunk in _chunks(urls, TAVILY_EXTRACT_CHUNK):
        logger.info("TavilyExtract chunk | urls={}", chunk)
        try:
            res = _extract_once(chunk, extract_depth=extract_depth)
            results.extend(res.get("results") or [])
            failed.extend(res.get("failed_results") or [])
        except Exception as e:
            # retries happen inside _extract_once; this is post-retry
            logger.warning("TavilyExtract failed after retries | urls={} | err={}", chunk, e)
            failed.append({"urls": chunk, "error": str(e)})

    return {"results": results, "failed_results": failed}


