import json
import os
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from logger import logger, preview
from tools.openai import get_openai_llm
from tools.tavily_crawl import tavily_crawl
from tools.tavily_extract import tavily_extract
from tools.tavily_search import tavily_search


with open("prompts/researcher_prompt.txt", "r") as f:
    RESEARCHER_PROMPT = f.read()

with open("prompts/url_select_prompt.txt", "r") as f:
    URL_SELECT_PROMPT = f.read()


compressor_prompt = ChatPromptTemplate.from_messages([
    ("system", RESEARCHER_PROMPT),
    ("human", "{section}\n\n{data}"),
])

url_select_prompt = ChatPromptTemplate.from_messages([
    ("system", URL_SELECT_PROMPT),
    ("human", "Select URLs to extract."),
])

compressor_llm = get_openai_llm(temperature=0.2, max_tokens=4096)
url_selector_llm = get_openai_llm(temperature=0.0, max_tokens=2048)

compressor = compressor_prompt | compressor_llm
url_selector = url_select_prompt | url_selector_llm


RESEARCH_CHUNK_CHARS = int(os.getenv("RESEARCH_CHUNK_CHARS", "12000"))
RESEARCH_MAX_CHUNKS = int(os.getenv("RESEARCH_MAX_CHUNKS", "10"))
EXTRACT_BATCH = int(os.getenv("EXTRACT_BATCH", "3"))  # 3 then +3 fallback
MAX_FALLBACK_URLS = int(os.getenv("MAX_FALLBACK_URLS", "6"))
CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "2"))
CRAWL_LIMIT = int(os.getenv("CRAWL_LIMIT", "15"))
CRAWL_EXTRACT_LIMIT = int(os.getenv("CRAWL_EXTRACT_LIMIT", "10"))


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _strip_json_fence(s: str) -> str:
    s2 = s.strip()
    if "```json" in s2:
        return s2.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in s2:
        return s2.split("```", 1)[1].split("```", 1)[0].strip()
    return s2


def _chunk_text(s: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        return [s]
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


def _extracted_text_len(extracted: Dict[str, Any]) -> int:
    total = 0
    for r in (extracted.get("results") or []):
        content = r.get("raw_content") or r.get("content") or ""
        total += len(content)
    return total


def _build_corpus_from_extract(extracted: Dict[str, Any]) -> str:
    parts: List[str] = []
    for r in (extracted.get("results") or []):
        url = r.get("url")
        title = r.get("title")
        content = r.get("raw_content") or r.get("content") or ""
        if not content:
            continue
        parts.append(f"[SOURCE]\nTitle: {title}\nURL: {url}\n\n{content}\n")
    return "\n\n".join(parts)


def _select_urls_for_query(original_query: str, search_query: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for r in (search_results.get("results") or []):
        rows.append(
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "score": r.get("score"),
                "snippet": (r.get("content") or "")[:300],
            }
        )
    payload = {
        "original_query": original_query,
        "search_query": search_query,
        "results": rows[:10],
    }

    logger.info("URLSelect | search_query={}", search_query)
    logger.debug("URLSelect candidates={}", preview(payload, 4000))

    resp = url_selector.invoke(
        {
            "original_query": original_query,
            "search_query": search_query,
            "results_json": json.dumps(payload, ensure_ascii=False),
        }
    )
    content = _strip_json_fence(resp.content)
    logger.debug("URLSelect raw output={}", preview(content, 3000))
    try:
        parsed = json.loads(content)
    except Exception as e:
        logger.error("URLSelect JSON parse failed: {} | content={}", e, preview(content, 2000))
        parsed = {"official_domains": [], "extract_urls": [], "fallback_urls": []}

    extract_urls = [u for u in (parsed.get("extract_urls") or []) if isinstance(u, str) and u.strip()]
    fallback_urls = [u for u in (parsed.get("fallback_urls") or []) if isinstance(u, str) and u.strip()]
    official_domains = [d for d in (parsed.get("official_domains") or []) if isinstance(d, str) and d.strip()]

    # enforce caps
    extract_urls = extract_urls[:EXTRACT_BATCH]
    fallback_urls = fallback_urls[:MAX_FALLBACK_URLS]

    return {
        "official_domains": official_domains,
        "extract_urls": extract_urls,
        "fallback_urls": fallback_urls,
    }


def _crawl_last_resort(seed_url: str) -> Dict[str, Any]:
    try:
        crawl_res = tavily_crawl(seed_url, max_depth=CRAWL_MAX_DEPTH, limit=CRAWL_LIMIT)
    except Exception as e:
        logger.warning("Crawl failed | seed_url={} | err={}", seed_url, e)
        return {"results": []}

    urls: List[str] = []
    for r in (crawl_res.get("results") or []):
        u = r.get("url") or r.get("source_url") or r.get("link")
        if u:
            urls.append(u)

    # dedupe, keep order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)

    return {"results": deduped[:CRAWL_EXTRACT_LIMIT]}


def _research_one_search_query(original_query: str, search_query: str) -> Tuple[str, str]:
    # 1) Tavily search
    search_res = tavily_search(search_query)

    # 2) LLM selects official-first URLs to extract (+ fallback)
    selection = _select_urls_for_query(original_query, search_query, search_res)
    extract_urls = selection["extract_urls"]
    fallback_urls = selection["fallback_urls"]
    official_domains = selection["official_domains"]

    logger.info("Selected extract_urls={}", extract_urls)
    logger.info("Selected fallback_urls={}", fallback_urls[:EXTRACT_BATCH])
    logger.info("Selected official_domains={}", official_domains)

    # 3) Extract primary URLs
    extracted_primary = tavily_extract(extract_urls)
    total_chars = _extracted_text_len(extracted_primary)
    logger.info("Extract primary done | chars={}", total_chars)

    # 4) If insufficient, extract fallback (+3)
    extracted_all = {"results": list(extracted_primary.get("results") or []), "failed_results": list(extracted_primary.get("failed_results") or [])}
    if total_chars < 8000 and fallback_urls:
        fb = fallback_urls[:EXTRACT_BATCH]
        extracted_fb = tavily_extract(fb)
        extracted_all["results"].extend(extracted_fb.get("results") or [])
        extracted_all["failed_results"].extend(extracted_fb.get("failed_results") or [])
        total_chars = _extracted_text_len(extracted_all)
        logger.info("Extract fallback done | chars={}", total_chars)

    # 5) Crawl last resort: only if still insufficient AND we have an official domain/seed
    if total_chars < 8000 and official_domains:
        # pick a seed URL from candidates that matches official domain if possible
        official_dom = official_domains[0].lower()
        seed = None
        for r in (search_res.get("results") or []):
            u = r.get("url") or ""
            if _domain(u) == official_dom:
                seed = u
                break
        seed = seed or (extract_urls[0] if extract_urls else None)
        if seed:
            logger.info("Crawl last resort | seed={}", seed)
            crawl_urls = _crawl_last_resort(seed).get("results") or []
            if crawl_urls:
                extracted_crawl = tavily_extract(crawl_urls)
                extracted_all["results"].extend(extracted_crawl.get("results") or [])
                extracted_all["failed_results"].extend(extracted_crawl.get("failed_results") or [])
                total_chars = _extracted_text_len(extracted_all)
                logger.info("Extract from crawl done | chars={}", total_chars)

    corpus = _build_corpus_from_extract(extracted_all)
    if not corpus:
        corpus = json.dumps(search_res, ensure_ascii=False)

    # 6) Summarize (send full corpus without chunking; GPT-5.2 can handle it)
    # NOTE: We intentionally bypass chunking to preserve all details.
    chunks = [corpus]
    summaries: List[str] = []
    for idx, ch in enumerate(chunks, start=1):
        logger.info(
            "Summarize | search_query={} | chunk {}/{} | chars={}",
            search_query,
            idx,
            len(chunks),
            len(ch),
        )
        resp = compressor.invoke({"section": search_query, "data": ch})
        summaries.append(resp.content)
    merged = "\n\n".join(summaries)
    return search_query, merged


def research_sections(state: dict) -> dict:
    # Planner produces exactly 4 search queries
    search_queries = state.get("search_queries") or []
    original_query = state.get("query") or ""
    logger.info("NODE researcher | search_queries_count={} | iteration={}", len(search_queries), state.get("iteration", 0))
    logger.info("Search queries={}", search_queries)

    results: List[Tuple[str, str]] = []
    for q in search_queries:
        if not q:
            continue
        results.append(_research_one_search_query(original_query, q))

    research_findings: Dict[str, str] = {}
    compressed_findings: List[str] = []
    for section, summary in results:
        research_findings[section] = summary
        compressed_findings.append(f"### {section}\n{summary}")

    findings_str = "\n\n".join(compressed_findings)
    logger.debug("researcher combined findings chars={} preview={}", len(findings_str), preview(findings_str, 2000))
    return {"research_findings": research_findings, "messages": [AIMessage(content=findings_str)]}

