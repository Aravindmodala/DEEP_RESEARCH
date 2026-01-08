"""
Web content extraction tools using Tavily Extract API.

This module provides web scraping functionality WITHOUT BeautifulSoup.
All content extraction is done via Tavily Extract API.
"""

from __future__ import annotations

from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field

from tools.tavily_extract import TavilyExtractTool


class ScrapedPage(BaseModel):
    """Scraped web page content."""

    url: str
    content: str = ""
    success: bool = True
    error: str | None = None


class MenuData(BaseModel):
    """Extracted menu data structure."""

    restaurant_name: str
    source_url: str
    raw_content: str = ""
    extraction_success: bool = True


def create_scraper_tools(api_key: str | None = None) -> list:
    """
    Create web scraping tools using Tavily Extract.
    
    NOTE: This replaces BeautifulSoup-based scraping with Tavily Extract API.
    All extraction is done server-side by Tavily for better reliability.
    """
    extractor = TavilyExtractTool(api_key)

    @tool
    def scrape_webpage(url: str) -> str:
        """
        Scrape content from a webpage using Tavily Extract.
        
        Args:
            url: Full URL of the webpage to scrape
        """
        logger.info(f"[SCRAPER] Scraping: {url}")
        result = extractor.extract_single(url)
        
        if not result.success:
            return f"Error scraping {url}: {result.error}"
        
        page = ScrapedPage(
            url=url,
            content=result.raw_content,
            success=True,
        )
        
        return f"""
URL: {page.url}
Content Length: {len(page.content)} characters

=== PAGE CONTENT ===
{page.content}
"""

    @tool
    def scrape_multiple_pages(urls: list[str]) -> str:
        """
        Scrape content from multiple webpages efficiently.
        Uses Tavily Extract batch processing.
        
        Args:
            urls: List of URLs to scrape (max 10)
        """
        logger.info(f"[SCRAPER] Batch scraping {len(urls)} pages")
        results = extractor.extract(urls[:10])
        
        output = []
        for r in results:
            output.append({
                "url": r.url,
                "success": r.success,
                "content_preview": r.raw_content[:1000] if r.success else None,
                "content_length": len(r.raw_content) if r.success else 0,
                "error": r.error if not r.success else None,
            })
        
        return str(output)

    @tool
    def scrape_restaurant_menu(url: str, restaurant_name: str) -> str:
        """
        Scrape menu information from a restaurant website.
        
        Use this after identifying a menu URL from search results.
        The LLM should then analyze the extracted content to identify
        menu items, prices, and categories.
        
        Args:
            url: URL of the restaurant's menu page
            restaurant_name: Name of the restaurant
        """
        logger.info(f"[SCRAPER] Extracting menu for {restaurant_name}")
        result = extractor.extract_single(url)
        
        if not result.success:
            return f"Failed to extract menu: {result.error}"
        
        menu_data = MenuData(
            restaurant_name=restaurant_name,
            source_url=url,
            raw_content=result.raw_content,
            extraction_success=True,
        )
        
        return f"""
=== RESTAURANT MENU EXTRACTION ===
Restaurant: {menu_data.restaurant_name}
Source: {menu_data.source_url}
Content Length: {len(menu_data.raw_content)} chars

=== MENU CONTENT ===
{menu_data.raw_content}

=== ANALYSIS INSTRUCTIONS ===
Parse the above content to extract:
- Menu categories
- Item names and descriptions
- Prices (look for $ amounts)
- Signature/featured items
"""

    @tool
    def scrape_competitor_sites(urls: list[str], competitor_names: list[str]) -> str:
        """
        Scrape multiple competitor websites to gather intelligence.
        
        Args:
            urls: List of competitor website URLs
            competitor_names: Corresponding names for each URL
        """
        logger.info(f"[SCRAPER] Scraping {len(urls)} competitor sites")
        
        # Ensure we have matching names
        if len(competitor_names) < len(urls):
            competitor_names.extend(["Unknown"] * (len(urls) - len(competitor_names)))
        
        results = extractor.extract(urls[:10])
        
        output = []
        for i, r in enumerate(results):
            name = competitor_names[i] if i < len(competitor_names) else "Unknown"
            output.append({
                "competitor_name": name,
                "url": r.url,
                "success": r.success,
                "content": r.raw_content[:3000] if r.success else None,
                "error": r.error if not r.success else None,
            })
        
        return str(output)

    return [
        scrape_webpage,
        scrape_multiple_pages,
        scrape_restaurant_menu,
        scrape_competitor_sites,
    ]
