import os
from typing import Any, Dict

import requests
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from config.settings import config
from logger import logger, preview


_CLIENT = TavilyClient(api_key=config.tavily_api_key)

TAVILY_TIMEOUT_S = int(os.getenv("TAVILY_TIMEOUT_S", "60"))
TAVILY_RETRIES = int(os.getenv("TAVILY_RETRIES", "3"))
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
TAVILY_SEARCH_MAX_RESULTS = int(os.getenv("TAVILY_SEARCH_MAX_RESULTS", "10"))


@retry(
    reraise=True,
    stop=stop_after_attempt(TAVILY_RETRIES),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type((TavilyTimeoutError, requests.exceptions.RequestException)),
)
def tavily_search(query: str, *, max_results: int | None = None) -> Dict[str, Any]:
    """Tavily search wrapper with retries/timeouts/logging."""
    max_results = max_results if max_results is not None else TAVILY_SEARCH_MAX_RESULTS
    logger.info("TavilySearch | depth={} | max_results={} | query={}", TAVILY_SEARCH_DEPTH, max_results, query)
    res = _CLIENT.search(
        query,
        search_depth=TAVILY_SEARCH_DEPTH,
        max_results=max_results,
        include_raw_content=False,
        timeout=TAVILY_TIMEOUT_S,
    )
    try:
        preview_rows = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "score": r.get("score"),
                "snippet": (r.get("content") or "")[:220],
            }
            for r in (res.get("results") or [])[:10]
        ]
        logger.debug("TavilySearch results preview={}", preview(preview_rows, 4000))
    except Exception as e:
        logger.debug("TavilySearch preview failed: {}", e)
    return res


