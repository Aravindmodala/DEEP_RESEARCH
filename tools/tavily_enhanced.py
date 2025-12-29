from logger import logger
from tools.tavily_extract import tavily_extract
from tools.tavily_search import tavily_search


def deep_research(query: str) -> dict:
    """
    Compatibility wrapper (older callers).
    New pipeline uses: tavily_search -> LLM URL selection -> tavily_extract -> (optional crawl last).
    """
    logger.warning("tools.tavily_enhanced.deep_research is deprecated; prefer tavily_search/tavily_extract/tavily_crawl")
    search = tavily_search(query)
    urls = [r.get("url") for r in (search.get("results") or []) if r.get("url")][:5]
    extracted = tavily_extract(urls)
    return {"search": search, "extracted": extracted, "crawled": []}