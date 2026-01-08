"""RESEARCHER Agent - Autonomous ReAct-style research execution."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from config import AgentConfig
from prompts import RESEARCHER_SYSTEM_PROMPT
from tools.google_maps import create_google_maps_tools
from tools.tavily_search import create_tavily_tools
from tools.tavily_extract import create_tavily_extract_tools
from tools.web_scraper import create_scraper_tools
from utils import create_llm


class ResearcherAgent:
    """
    NODE 2 — RESEARCHER AGENT (AUTONOMOUS)
    
    Executes research plan using ReAct loop.
    Returns plain dict (no strict validation - Critic can handle flexible data).
    """

    MAX_REACT_ITERATIONS = 15

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config)
        self.tools = self._initialize_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

    def _initialize_tools(self) -> list:
        """Initialize all research tools."""
        tools = []
        
        try:
            tools.extend(create_google_maps_tools(self.config.google_maps_api_key))
            logger.info("[RESEARCHER] Loaded Google Maps tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Google Maps tools unavailable: {e}")

        try:
            tools.extend(create_tavily_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Tavily Search tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Tavily Search tools unavailable: {e}")

        try:
            tools.extend(create_tavily_extract_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Tavily Extract tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Tavily Extract tools unavailable: {e}")

        try:
            tools.extend(create_scraper_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Web Scraper tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Web Scraper tools unavailable: {e}")

        return tools

    def research(self, plan: dict[str, Any]) -> dict[str, Any]:
        """
        Execute research based on the planner's output.
        
        Args:
            plan: Research plan dict from Planner
            
        Returns:
            Plain dict with research findings
        """
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        intent = plan.get("intent", "")
        cuisine_type = plan.get("cuisine_type", "unknown")
        search_queries = plan.get("search_queries", [])
        
        logger.info(f"[RESEARCHER] Starting research for {target}")
        logger.info(f"[RESEARCHER] Location: {location}")

        llm_with_tools = self.llm.bind_tools(self.tools)

        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=self._format_plan_message(plan)),
        ]

        # Track collected data
        collected_data = {
            "competitors": [],
            "reviews": [],
            "menus": [],
            "market_data": [],
            "sources": [],
        }

        # ReAct loop
        iteration = 0
        while iteration < self.MAX_REACT_ITERATIONS:
            iteration += 1
            logger.info(f"[RESEARCHER] ReAct iteration {iteration}")

            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                logger.info("[RESEARCHER] No more tool calls - synthesizing results")
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call.get("id", f"call_{iteration}")

                logger.info(f"[RESEARCHER] Calling tool: {tool_name}")

                try:
                    tool = self.tool_map.get(tool_name)
                    if tool:
                        result = tool.invoke(tool_args)
                        self._collect_data(collected_data, tool_name, result)
                        
                        collected_data["sources"].append({
                            "source_type": self._get_source_type(tool_name),
                            "title": f"{tool_name} call",
                            "data_summary": str(result)[:200],
                            "accessed_at": datetime.now().isoformat(),
                        })
                    else:
                        result = f"Tool {tool_name} not found"
                        logger.warning(result)
                except Exception as e:
                    result = f"Tool error: {str(e)}"
                    logger.error(f"[RESEARCHER] Tool {tool_name} error: {e}")

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        # Build output from collected data
        output = self._build_output(collected_data, plan)
        
        logger.info(f"[RESEARCHER] Research complete. Found {len(output.get('competitors', []))} competitors")
        return output

    def _format_plan_message(self, plan: dict[str, Any]) -> str:
        """Format the plan into a message for the researcher."""
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        cuisine_type = plan.get("cuisine_type", "unknown")
        intent = plan.get("intent", "")
        queries = plan.get("search_queries", [])
        
        cuisine_info = f"**Cuisine Type:** {cuisine_type}" if cuisine_type != "unknown" else "**Cuisine Type:** (to be determined)"
        
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
        
        return f"""## Research Plan

**Target Restaurant:** {target}
**Location:** {location}
{cuisine_info}
**Intent:** {intent}

**Research Queries:**
{queries_text}

## IMPORTANT REQUIREMENTS:

1. **Find competitors of the SAME cuisine type** ({cuisine_type or 'same as target'})
2. **Scrape menus for TARGET + TOP 3 COMPETITORS**
3. **Get reviews for target restaurant**

Begin your research by first finding the target restaurant on Google Maps, then discovering nearby competitors."""

    def _get_source_type(self, tool_name: str) -> str:
        """Map tool name to source type."""
        if "restaurant" in tool_name or "competitor" in tool_name or "review" in tool_name:
            return "google_maps"
        elif "search" in tool_name or "traffic" in tool_name or "market" in tool_name:
            return "tavily"
        elif "extract" in tool_name or "crawl" in tool_name:
            return "website_scrape"
        return "api"

    def _collect_data(self, collected_data: dict, tool_name: str, result: Any) -> None:
        """Collect and organize data from tool results."""
        try:
            parsed = result
            if isinstance(result, str):
                if result.startswith("{") or result.startswith("["):
                    try:
                        parsed = json.loads(result)
                    except:
                        pass

            if "competitor" in tool_name.lower():
                if isinstance(parsed, dict):
                    if "competitors" in parsed:
                        collected_data["competitors"].extend(parsed["competitors"])
                    if "target" in parsed and parsed["target"]:
                        collected_data["target"] = parsed["target"]
            elif "review" in tool_name.lower():
                if isinstance(parsed, list):
                    collected_data["reviews"].extend(parsed)
            elif "menu" in tool_name.lower():
                collected_data["menus"].append(parsed)
            elif "search" in tool_name.lower() or "market" in tool_name.lower():
                if isinstance(parsed, list):
                    collected_data["market_data"].extend(parsed)
                else:
                    collected_data["market_data"].append(parsed)

        except Exception as e:
            logger.warning(f"[RESEARCHER] Error collecting data: {e}")

    def _build_output(self, collected_data: dict, plan: dict[str, Any]) -> dict[str, Any]:
        """Build research output dict from collected data."""
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        
        # Process competitors - just keep as list of dicts
        competitors = []
        for c in collected_data.get("competitors", []):
            if isinstance(c, dict):
                competitors.append({
                    "name": c.get("name", "Unknown"),
                    "address": c.get("address", ""),
                    "distance_miles": c.get("distance_miles", 0.0),
                    "rating": c.get("rating"),
                    "review_count": c.get("review_count"),
                    "price_level": c.get("price_level"),
                    "cuisine_type": c.get("cuisine_type", "restaurant"),
                    "website": c.get("website"),
                    "place_id": c.get("place_id"),
                })
        
        # Limit competitors
        competitors = competitors[:self.config.max_competitors]

        # Build sentiment analysis
        sentiment = self._analyze_sentiment(collected_data.get("reviews", []))

        # Build pricing analysis
        pricing = self._build_pricing_analysis(competitors)

        # Build market signals
        market_signals = self._build_market_signals(competitors, collected_data.get("market_data", []))

        # Build menu comparison
        menu_comparison = self._build_menu_comparison(collected_data.get("menus", []), target)

        return {
            "competitors": competitors,
            "menu_comparison": menu_comparison,
            "pricing_analysis": pricing,
            "sentiment_analysis": sentiment,
            "market_signals": market_signals,
            "raw_sources": collected_data.get("sources", []),
            "research_notes": [
                f"Analyzed {len(competitors)} competitors in {location}",
                f"Research intent: {plan.get('intent', 'Unknown')}",
            ],
        }

    def _analyze_sentiment(self, reviews: list) -> dict[str, Any]:
        """Analyze sentiment from collected reviews."""
        if not reviews:
            return {
                "target_overall_sentiment": None,
                "positive_drivers": ["Insufficient review data"],
                "common_complaints": ["No reviews collected"],
                "sample_reviews": [],
            }

        ratings = []
        positive_themes = []
        negative_themes = []

        positive_keywords = ["great", "excellent", "delicious", "amazing", "best", "love", "fantastic", "friendly", "recommend", "fresh"]
        negative_keywords = ["slow", "cold", "expensive", "rude", "wait", "disappointing", "mediocre", "overpriced", "dirty", "bad"]

        for review in reviews:
            if isinstance(review, dict):
                rating = review.get("rating")
                if rating:
                    ratings.append(float(rating))

                text = review.get("text", "").lower()
                for kw in positive_keywords:
                    if kw in text:
                        positive_themes.append(kw)
                for kw in negative_keywords:
                    if kw in text:
                        negative_themes.append(kw)

        avg_rating = sum(ratings) / len(ratings) if ratings else None
        overall = avg_rating / 5 if avg_rating else None

        top_positive = [t[0] for t in Counter(positive_themes).most_common(5)]
        top_negative = [t[0] for t in Counter(negative_themes).most_common(5)]

        return {
            "target_overall_sentiment": overall,
            "positive_drivers": top_positive if top_positive else ["Good overall ratings"],
            "common_complaints": top_negative if top_negative else ["No major complaints identified"],
            "sample_reviews": reviews[:3],
        }

    def _build_pricing_analysis(self, competitors: list[dict]) -> dict[str, Any]:
        """Build pricing analysis from competitor data."""
        price_levels = []
        for c in competitors:
            if c.get("price_level"):
                level = len(c["price_level"].replace(" ", ""))
                price_levels.append(level)

        avg_level = sum(price_levels) / len(price_levels) if price_levels else 2

        if avg_level <= 1.5:
            position = "budget"
        elif avg_level <= 2.5:
            position = "mid-range"
        elif avg_level <= 3.5:
            position = "premium"
        else:
            position = "luxury"

        return {
            "price_position": position,
            "target_avg_price": None,
            "market_avg_price": None,
            "competitor_price_range": {
                c["name"]: {"level": c.get("price_level", "$$")}
                for c in competitors[:5]
            },
        }

    def _build_market_signals(self, competitors: list[dict], market_data: list) -> dict[str, Any]:
        """Build market signals from collected data."""
        competitor_count = len(competitors)
        
        if competitor_count <= 3:
            saturation = "low"
        elif competitor_count <= 8:
            saturation = "moderate"
        elif competitor_count <= 15:
            saturation = "high"
        else:
            saturation = "oversaturated"

        ratings = [c.get("rating") for c in competitors if c.get("rating")]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        foot_traffic = []
        growth = []
        risks = []

        for data in market_data:
            if isinstance(data, dict):
                content = data.get("content", "").lower()
                if "busy" in content or "popular" in content or "traffic" in content:
                    foot_traffic.append(data.get("title", "")[:50])
                if "growth" in content or "expanding" in content or "new" in content:
                    growth.append(data.get("title", "")[:50])
                if "closing" in content or "decline" in content or "struggle" in content:
                    risks.append(data.get("title", "")[:50])

        return {
            "competitor_density": competitor_count,
            "market_saturation": saturation,
            "avg_competitor_rating": avg_rating,
            "foot_traffic_indicators": foot_traffic[:5] if foot_traffic else ["Foot traffic data not available"],
            "growth_indicators": growth[:5] if growth else ["Growth signals not found"],
            "risk_indicators": risks[:5] if risks else ["No significant risk signals"],
        }

    def _build_menu_comparison(self, menus: list, target_name: str) -> dict[str, Any]:
        """Build menu comparison from collected data."""
        target_menu = None
        competitor_menus = []

        for menu_data in menus:
            if isinstance(menu_data, str):
                try:
                    menu_data = json.loads(menu_data)
                except:
                    continue

            if isinstance(menu_data, dict):
                name = menu_data.get("restaurant_name", "")
                if target_name.lower() in name.lower():
                    target_menu = menu_data
                else:
                    competitor_menus.append(menu_data)

        return {
            "target_menu": target_menu,
            "competitor_menus": competitor_menus,
            "unique_offerings": [],
        }

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        planner_output = state.get("planner_output")
        if not planner_output:
            raise ValueError("No planner_output found in state")

        try:
            output = self.research(planner_output)  # planner_output is already a dict
            return {
                **state,
                "researcher_output": output,  # Plain dict
                "current_node": "critic",
                "iteration_count": state.get("iteration_count", 0) + 1,
            }
        except Exception as e:
            logger.error(f"[RESEARCHER] Error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Researcher error: {str(e)}"],
                "current_node": "error",
            }
