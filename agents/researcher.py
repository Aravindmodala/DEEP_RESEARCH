"""RESEARCHER Agent - Autonomous ReAct-style research execution."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from config import AgentConfig
from models.schemas import (
    CompetitorInfo,
    MarketSignals,
    MenuComparison,
    PricingAnalysis,
    ResearcherOutput,
    RestaurantInfo,
    RestaurantSentiment,
    SentimentAnalysis,
    SourceReference,
)
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
    Returns validated ResearcherOutput Pydantic model.
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

    def research(self, plan: dict[str, Any]) -> ResearcherOutput:
        """
        Execute research based on the planner's output.
        
        Args:
            plan: Research plan dict from Planner
            
        Returns:
            Validated ResearcherOutput Pydantic model
        """
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        
        logger.info(f"[RESEARCHER] Starting research for {target}")
        logger.info(f"[RESEARCHER] Location: {location}")

        llm_with_tools = self.llm.bind_tools(self.tools)

        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=self._format_plan_message(plan)),
        ]

        # Track collected data
        collected_data = {
            "target": None,
            "competitors": [],
            "reviews": {},  # {restaurant_name: {"rating": 4.5, "review_count": 234, "reviews": [...]}}
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
                        self._collect_data(collected_data, tool_name, tool_args, result)
                        
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

        # Analyze sentiment using LLM for target + top 3 competitors
        sentiment_analysis = self._analyze_sentiment_with_llm(collected_data)

        # Build output
        output = self._build_output(collected_data, plan, sentiment_analysis)
        
        logger.info(f"[RESEARCHER] Research complete. Found {len(output.competitors)} competitors")
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

1. **Find the target restaurant** using find_restaurant tool to get its place_id
2. **Find competitors** of the SAME cuisine type using find_competitors
3. **Get reviews for TARGET restaurant** using get_restaurant_reviews with the target's place_id
4. **Get reviews for TOP 3 COMPETITORS** using get_restaurant_reviews for each

The reviews are critical for sentiment analysis. Make sure to collect reviews for:
- The target restaurant
- At least 3 top-rated competitors

Begin your research now."""

    def _get_source_type(self, tool_name: str) -> str:
        """Map tool name to source type."""
        if "restaurant" in tool_name or "competitor" in tool_name or "review" in tool_name:
            return "google_maps"
        elif "search" in tool_name or "traffic" in tool_name or "market" in tool_name:
            return "tavily"
        elif "extract" in tool_name or "crawl" in tool_name:
            return "website_scrape"
        return "api"

    def _collect_data(self, collected_data: dict, tool_name: str, tool_args: dict, result: Any) -> None:
        """Collect and organize data from tool results."""
        try:
            parsed = result
            if isinstance(result, str):
                if result.startswith("{") or result.startswith("["):
                    try:
                        parsed = json.loads(result)
                    except:
                        # Try eval for dict-like strings
                        try:
                            parsed = eval(result)
                        except:
                            pass

            tool_lower = tool_name.lower()
            
            if "find_competitor" in tool_lower:
                if isinstance(parsed, dict):
                    if "competitors" in parsed:
                        collected_data["competitors"].extend(parsed["competitors"])
                    if "target" in parsed and parsed["target"]:
                        collected_data["target"] = parsed["target"]
                        
            elif "find_restaurant" in tool_lower:
                if isinstance(parsed, dict) and parsed.get("name"):
                    collected_data["target"] = parsed
                    
            elif "review" in tool_lower:
                # Store reviews with restaurant identifier
                if isinstance(parsed, list):
                    # Try to identify which restaurant these reviews are for
                    place_id = tool_args.get("place_id", "unknown")
                    
                    # Find restaurant name from place_id
                    restaurant_name = self._find_restaurant_name(collected_data, place_id)
                    
                    # Get rating from target or competitors
                    rating_info = self._get_rating_info(collected_data, place_id)
                    
                    collected_data["reviews"][restaurant_name] = {
                        "place_id": place_id,
                        "rating": rating_info.get("rating"),
                        "review_count": rating_info.get("review_count"),
                        "reviews": parsed,
                    }
                    
            elif "menu" in tool_lower:
                collected_data["menus"].append(parsed)
                
            elif "search" in tool_lower or "market" in tool_lower:
                if isinstance(parsed, list):
                    collected_data["market_data"].extend(parsed)
                else:
                    collected_data["market_data"].append(parsed)

        except Exception as e:
            logger.warning(f"[RESEARCHER] Error collecting data: {e}")

    def _find_restaurant_name(self, collected_data: dict, place_id: str) -> str:
        """Find restaurant name from place_id."""
        # Check target
        if collected_data.get("target", {}).get("place_id") == place_id:
            return collected_data["target"].get("name", "Target Restaurant")
        
        # Check competitors
        for comp in collected_data.get("competitors", []):
            if comp.get("place_id") == place_id:
                return comp.get("name", "Unknown")
        
        return f"Restaurant_{place_id[:8]}"

    def _get_rating_info(self, collected_data: dict, place_id: str) -> dict:
        """Get rating info for a restaurant by place_id."""
        # Check target
        target = collected_data.get("target", {})
        if target.get("place_id") == place_id:
            return {
                "rating": target.get("rating"),
                "review_count": target.get("user_ratings_total") or target.get("review_count"),
            }
        
        # Check competitors
        for comp in collected_data.get("competitors", []):
            if comp.get("place_id") == place_id:
                return {
                    "rating": comp.get("rating"),
                    "review_count": comp.get("user_ratings_total") or comp.get("review_count"),
                }
        
        return {}

    def _analyze_sentiment_with_llm(self, collected_data: dict) -> dict[str, Any]:
        """
        Use LLM to analyze sentiment from ratings and reviews.
        Analyzes target + competitors.
        """
        reviews_data = collected_data.get("reviews", {})
        
        if not reviews_data:
            return {
                "target_sentiment": None,
                "competitor_sentiments": [],
                "summary": "No reviews collected for sentiment analysis.",
            }
        
        # Build context for LLM
        reviews_context = []
        for restaurant_name, data in reviews_data.items():
            rating = data.get("rating", "N/A")
            review_count = data.get("review_count", "N/A")
            reviews = data.get("reviews", [])
            
            review_texts = "\n".join([
                f"  - Rating: {r.get('rating', 'N/A')}/5 - \"{r.get('text', '')[:200]}...\""
                for r in reviews[:5]
            ])
            
            reviews_context.append(f"""
### {restaurant_name}
- **Overall Rating:** {rating}/5.0 (based on {review_count} reviews)
- **Sample Reviews:**
{review_texts}
""")
        
        prompt = f"""Analyze the sentiment for these restaurants based on their ratings and reviews.

{chr(10).join(reviews_context)}

For each restaurant, provide:
1. **Sentiment Score** (0.0 to 1.0, where 1.0 is most positive)
2. **Key Strengths** (2-3 positive themes from reviews)
3. **Key Concerns** (2-3 negative themes or areas for improvement)
4. **Summary** (1-2 sentence assessment)

Output as JSON:
{{
    "restaurants": [
        {{
            "name": "Restaurant Name",
            "sentiment_score": 0.85,
            "key_strengths": ["strength1", "strength2"],
            "key_concerns": ["concern1", "concern2"],
            "summary": "Brief assessment"
        }}
    ],
    "comparative_summary": "How does the target compare to competitors?"
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._extract_json(response.content)
            
            if result:
                return {
                    "restaurants": result.get("restaurants", []),
                    "comparative_summary": result.get("comparative_summary", ""),
                    "raw_reviews": reviews_data,
                }
        except Exception as e:
            logger.error(f"[RESEARCHER] Error in LLM sentiment analysis: {e}")
        
        # Fallback: return raw data
        return {
            "restaurants": [],
            "comparative_summary": "Sentiment analysis pending.",
            "raw_reviews": reviews_data,
        }

    def _convert_price_level(self, level: int | str | None) -> str | None:
        """Convert price_level from int to string representation."""
        if level is None:
            return None
        if isinstance(level, str):
            return level  # Already a string
        # Convert int to string representation
        mapping = {0: "$", 1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
        return mapping.get(level, "$$")

    def _build_output(self, collected_data: dict, plan: dict[str, Any], sentiment_analysis: dict) -> ResearcherOutput:
        """Build validated ResearcherOutput from collected data."""
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        
        # Process target restaurant
        target_data = collected_data.get("target")
        target_info = None
        if target_data and isinstance(target_data, dict):
            target_info = RestaurantInfo(
                name=target_data.get("name", "Unknown"),
                address=target_data.get("address", ""),
                place_id=target_data.get("place_id"),
                rating=target_data.get("rating"),
                review_count=target_data.get("user_ratings_total") or target_data.get("review_count"),
                price_level=self._convert_price_level(target_data.get("price_level")),
                cuisine_type=target_data.get("cuisine_type", "restaurant"),
                website=target_data.get("website"),
            )
        
        # Process competitors
        competitors = []
        for c in collected_data.get("competitors", []):
            if isinstance(c, dict):
                competitors.append(CompetitorInfo(
                    name=c.get("name", "Unknown"),
                    address=c.get("address", ""),
                    distance_miles=c.get("distance_miles", 0.0),
                    rating=c.get("rating"),
                    review_count=c.get("review_count") or c.get("user_ratings_total"),
                    price_level=self._convert_price_level(c.get("price_level")),
                    cuisine_type=c.get("cuisine_type", "restaurant"),
                    website=c.get("website"),
                    place_id=c.get("place_id"),
                ))
        
        competitors = competitors[:self.config.max_competitors]

        # Build pricing analysis
        pricing_dict = self._build_pricing_analysis([c.model_dump() for c in competitors])
        pricing = PricingAnalysis(**pricing_dict)

        # Build market signals
        market_signals_dict = self._build_market_signals(
            [c.model_dump() for c in competitors], 
            collected_data.get("market_data", [])
        )
        market_signals = MarketSignals(**market_signals_dict)

        # Build menu comparison
        menu_dict = self._build_menu_comparison(collected_data.get("menus", []), target)
        menu_comparison = MenuComparison(**menu_dict)

        # Build sentiment analysis
        restaurants_sentiment = []
        for r in sentiment_analysis.get("restaurants", []):
            if isinstance(r, dict):
                restaurants_sentiment.append(RestaurantSentiment(**r))
        
        sentiment = SentimentAnalysis(
            restaurants=restaurants_sentiment,
            comparative_summary=sentiment_analysis.get("comparative_summary", ""),
            raw_reviews=sentiment_analysis.get("raw_reviews", {}),
        )

        # Build source references
        sources = []
        for s in collected_data.get("sources", []):
            if isinstance(s, dict):
                sources.append(SourceReference(**s))

        return ResearcherOutput(
            target=target_info,
            competitors=competitors,
            menu_comparison=menu_comparison,
            pricing_analysis=pricing,
            sentiment_analysis=sentiment,
            market_signals=market_signals,
            raw_sources=sources,
            research_notes=[
                f"Analyzed {len(competitors)} competitors in {location}",
                f"Collected reviews for {len(sentiment_analysis.get('raw_reviews', {}))} restaurants",
                f"Research intent: {plan.get('intent', 'Unknown')}",
            ],
        )

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON from text."""
        try:
            return json.loads(text)
        except:
            pass

        json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end])
            except:
                pass

        return None

    def _build_pricing_analysis(self, competitors: list[dict]) -> dict[str, Any]:
        """Build pricing analysis from competitor data."""
        price_levels = []
        for c in competitors:
            if c.get("price_level"):
                level = len(str(c["price_level"]).replace(" ", ""))
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
            "foot_traffic_indicators": foot_traffic[:5] if foot_traffic else [],
            "growth_indicators": growth[:5] if growth else [],
            "risk_indicators": risks[:5] if risks else [],
        }

    def _build_menu_comparison(self, menus: list, target_name: str) -> dict[str, Any]:
        """Build menu comparison from collected data."""
        target_menu = None
        competitor_menus = []

        for menu_data in menus:
            if isinstance(menu_data, str):
                try:
                    menu_data = json.loads(menu_data)
                except Exception:
                    # Don't drop raw menu content; keep it as unstructured fallback
                    menu_data = {
                        "type": "menu_extraction_unstructured",
                        "restaurant_name": "",
                        "source_url": None,
                        "status": "unknown",
                        "raw_content": menu_data[:12000],
                    }

            if isinstance(menu_data, dict):
                name = menu_data.get("restaurant_name", "") or menu_data.get("Restaurant", "")
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
            output = self.research(planner_output)
            return {
                **state,
                "researcher_output": output.model_dump(),  # Convert Pydantic to dict for state
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
