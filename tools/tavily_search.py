"""Tavily Search API integration for market research."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential


class SearchResult(BaseModel):
    """Structured search result."""

    title: str
    url: str
    content: str
    score: float = 0.0


class TavilySearchTool:
    """Tavily Search API wrapper for market research."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        self.client = TavilyClient(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 10,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Perform a web search using Tavily.
        
        Args:
            query: Search query
            search_depth: "basic" or "advanced" (more thorough)
            max_results: Maximum results to return
            include_domains: Optional list of domains to include
            exclude_domains: Optional list of domains to exclude
        """
        try:
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            results = []
            for item in response.get("results", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        score=item.get("score", 0.0),
                    )
                )

            return results
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_search_context(
        self,
        query: str,
        max_tokens: int = 4000,
        search_depth: str = "advanced",
    ) -> str:
        """
        Get a synthesized context from search results.
        Useful for getting a comprehensive answer.
        """
        try:
            context = self.client.get_search_context(
                query=query,
                search_depth=search_depth,
                max_tokens=max_tokens,
            )
            return context
        except Exception as e:
            logger.error(f"Tavily context error: {e}")
            raise

    def search_restaurant_news(
        self,
        restaurant_name: str,
        location: str,
    ) -> list[SearchResult]:
        """Search for recent news about a restaurant."""
        query = f"{restaurant_name} restaurant {location} news reviews"
        return self.search(
            query=query,
            search_depth="advanced",
            max_results=10,
            exclude_domains=["yelp.com", "tripadvisor.com"],  # Get news, not review sites
        )

    def search_market_trends(
        self,
        cuisine_type: str,
        location: str,
    ) -> list[SearchResult]:
        """Search for restaurant market trends in a location."""
        query = f"{cuisine_type} restaurant market trends {location} 2024 2025"
        return self.search(
            query=query,
            search_depth="advanced",
            max_results=10,
        )

    def search_local_economy(self, location: str) -> list[SearchResult]:
        """Search for local economic indicators."""
        query = f"{location} economy business growth retail dining trends"
        return self.search(
            query=query,
            search_depth="basic",
            max_results=8,
        )

    def search_competitor_intelligence(
        self,
        competitor_names: list[str],
        location: str,
    ) -> dict[str, list[SearchResult]]:
        """Search for intelligence on multiple competitors."""
        results = {}
        for name in competitor_names[:5]:  # Limit to top 5 competitors
            query = f"{name} restaurant {location}"
            results[name] = self.search(
                query=query,
                search_depth="basic",
                max_results=5,
            )
        return results

    def search_foot_traffic_indicators(
        self,
        location: str,
        neighborhood: str | None = None,
    ) -> list[SearchResult]:
        """Search for foot traffic and demand indicators."""
        area = neighborhood or location
        query = f"{area} foot traffic retail activity dining demand restaurants busy popular"
        return self.search(
            query=query,
            search_depth="advanced",
            max_results=10,
        )


# =============================================================================
# LangChain Tool Wrappers
# =============================================================================


def create_tavily_tools(api_key: str | None = None) -> list:
    """Create LangChain-compatible tools from TavilySearchTool."""
    tavily = TavilySearchTool(api_key)

    @tool
    def web_search(query: str, max_results: int = 10) -> str:
        """
        Search the web for information using Tavily.
        Use for market research, news, trends, and general information.
        
        Args:
            query: Search query - be specific and descriptive
            max_results: Maximum number of results (default 10)
        """
        results = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
        )
        output = [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content[:500],  # Truncate for context
                "relevance": r.score,
            }
            for r in results
        ]
        return str(output)

    @tool
    def search_restaurant_market(
        restaurant_name: str,
        location: str,
        research_type: str = "general",
    ) -> str:
        """
        Search for restaurant-specific market intelligence.
        
        Args:
            restaurant_name: Name of the restaurant
            location: City, State location
            research_type: Type of research - "news", "trends", "economy", or "general"
        """
        if research_type == "news":
            results = tavily.search_restaurant_news(restaurant_name, location)
        elif research_type == "trends":
            results = tavily.search_market_trends(restaurant_name, location)
        elif research_type == "economy":
            results = tavily.search_local_economy(location)
        else:
            results = tavily.search(
                query=f"{restaurant_name} restaurant {location} review market",
                search_depth="advanced",
            )

        output = [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content[:500],
            }
            for r in results
        ]
        return str(output)

    @tool
    def search_foot_traffic(location: str, neighborhood: str = "") -> str:
        """
        Search for foot traffic and demand indicators in an area.
        
        Args:
            location: City, State location
            neighborhood: Specific neighborhood (optional)
        """
        results = tavily.search_foot_traffic_indicators(
            location=location,
            neighborhood=neighborhood if neighborhood else None,
        )
        output = [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content[:500],
            }
            for r in results
        ]
        return str(output)

    @tool
    def get_market_context(query: str) -> str:
        """
        Get a synthesized, comprehensive context for a market research query.
        Best for getting a holistic answer rather than individual results.
        
        Args:
            query: Detailed research question
        """
        context = tavily.get_search_context(
            query=query,
            max_tokens=4000,
            search_depth="advanced",
        )
        return context

    return [web_search, search_restaurant_market, search_foot_traffic, get_market_context]

