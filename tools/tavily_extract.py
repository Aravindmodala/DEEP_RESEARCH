"""Tavily Extract API for extracting content from URLs."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential


class ExtractedContent(BaseModel):
    """Content extracted from a URL using Tavily Extract."""

    url: str
    raw_content: str = ""
    success: bool = True
    error: str | None = None


class TavilyExtractTool:
    """
    Tavily Extract API wrapper for extracting content from URLs.
    
    This is separate from Tavily Search - used specifically for
    extracting full page content from URLs that the LLM decides to scrape.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        self.client = TavilyClient(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def extract(self, urls: list[str]) -> list[ExtractedContent]:
        """
        Extract content from multiple URLs using Tavily Extract API.
        
        Args:
            urls: List of URLs to extract content from (max 10 per call)
            
        Returns:
            List of ExtractedContent with raw_content from each URL
        """
        if not urls:
            return []
            
        # Tavily Extract accepts up to 10 URLs at once
        urls_to_extract = urls[:10]
        logger.info(f"[TAVILY EXTRACT] Extracting content from {len(urls_to_extract)} URLs")
        
        try:
            response = self.client.extract(urls=urls_to_extract)
            
            results = []
            
            # Process successful extractions
            for item in response.get("results", []):
                results.append(
                    ExtractedContent(
                        url=item.get("url", ""),
                        raw_content=item.get("raw_content", ""),
                        success=True,
                    )
                )
                logger.debug(f"[TAVILY EXTRACT] Successfully extracted: {item.get('url', '')[:50]}...")
            
            # Process failed extractions
            for failed_item in response.get("failed_results", []):
                results.append(
                    ExtractedContent(
                        url=failed_item.get("url", ""),
                        raw_content="",
                        success=False,
                        error=failed_item.get("error", "Unknown error"),
                    )
                )
                logger.warning(f"[TAVILY EXTRACT] Failed: {failed_item.get('url', '')} - {failed_item.get('error', '')}")
            
            return results
            
        except Exception as e:
            logger.error(f"[TAVILY EXTRACT] Error: {e}")
            raise

    def extract_single(self, url: str) -> ExtractedContent:
        """
        Extract content from a single URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            ExtractedContent with the page content
        """
        results = self.extract([url])
        if results:
            return results[0]
        return ExtractedContent(
            url=url,
            raw_content="",
            success=False,
            error="No result returned from extraction",
        )

    def extract_batch(
        self,
        urls: list[str],
        batch_size: int = 10,
    ) -> list[ExtractedContent]:
        """
        Extract content from many URLs in batches.
        
        Args:
            urls: List of URLs to extract
            batch_size: Number of URLs per batch (max 10)
            
        Returns:
            List of all ExtractedContent results
        """
        all_results = []
        batch_size = min(batch_size, 10)  # Tavily max is 10
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            logger.info(f"[TAVILY EXTRACT] Processing batch {i // batch_size + 1}: {len(batch)} URLs")
            results = self.extract(batch)
            all_results.extend(results)
        
        return all_results


# =============================================================================
# LangChain Tool Wrappers
# =============================================================================


def create_tavily_extract_tools(api_key: str | None = None) -> list:
    """Create LangChain-compatible tools for Tavily Extract."""
    extractor = TavilyExtractTool(api_key)

    @tool
    def extract_url_content(url: str) -> str:
        """
        Extract full content from a single URL using Tavily Extract.
        
        Use this to get detailed content from a webpage after identifying
        it as relevant from search results.
        
        Args:
            url: Full URL of the webpage to extract
        """
        logger.info(f"[TOOL] Extracting content from: {url}")
        result = extractor.extract_single(url)
        
        if not result.success:
            return f"Failed to extract content from {url}: {result.error}"
        
        return f"""
URL: {url}
Status: Successfully extracted

Content:
{result.raw_content}
"""

    @tool
    def extract_multiple_urls(urls: list[str]) -> str:
        """
        Extract content from multiple URLs at once using Tavily Extract.
        More efficient than extracting one URL at a time.
        
        The LLM should analyze search results and select the most relevant
        URLs to extract (e.g., menu pages, official websites, review pages).
        
        Args:
            urls: List of URLs to extract content from (max 10)
        """
        logger.info(f"[TOOL] Extracting content from {len(urls)} URLs")
        results = extractor.extract(urls[:10])
        
        output = []
        for r in results:
            if r.success:
                output.append({
                    "url": r.url,
                    "status": "success",
                    "content_length": len(r.raw_content),
                    "content": r.raw_content[:6000],  # Limit per URL
                })
            else:
                output.append({
                    "url": r.url,
                    "status": "failed",
                    "error": r.error,
                })
        
        return str(output)

    @tool
    def extract_menu_page(url: str, restaurant_name: str) -> str:
        """
        Extract menu information from a restaurant's menu page.
        
        Use this specifically for URLs that appear to be menu pages.
        The extracted content will include prices, items, and categories
        that the LLM can then analyze.
        
        Args:
            url: URL of the menu page
            restaurant_name: Name of the restaurant for context
        """
        logger.info(f"[TOOL] Extracting menu for {restaurant_name} from: {url}")
        result = extractor.extract_single(url)
        
        if not result.success:
            return f"Failed to extract menu from {url}: {result.error}"
        
        return f"""
=== MENU EXTRACTION ===
Restaurant: {restaurant_name}
Source URL: {url}
Status: Successfully extracted

=== RAW CONTENT ===
{result.raw_content}

=== INSTRUCTIONS FOR LLM ===
Analyze the above content to identify:
1. Menu categories (appetizers, mains, desserts, drinks, etc.)
2. Individual menu items with names
3. Prices for each item
4. Any special offerings or signature dishes
"""

    @tool
    def extract_business_info(url: str, business_name: str) -> str:
        """
        Extract business information from a webpage.
        
        Use for extracting details like hours, location, contact info,
        about us content, etc.
        
        Args:
            url: URL to extract from
            business_name: Name of the business
        """
        logger.info(f"[TOOL] Extracting business info for {business_name} from: {url}")
        result = extractor.extract_single(url)
        
        if not result.success:
            return f"Failed to extract from {url}: {result.error}"
        
        return f"""
=== BUSINESS INFO EXTRACTION ===
Business: {business_name}
Source URL: {url}

=== CONTENT ===
{result.raw_content}

=== LOOK FOR ===
- Operating hours
- Address / Location
- Phone number
- About / History
- Services offered
- Pricing information
"""

    return [
        extract_url_content,
        extract_multiple_urls,
        extract_menu_page,
        extract_business_info,
    ]

