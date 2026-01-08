"""RESEARCHER Agent - Autonomous ReAct-style research execution."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from config import AgentConfig
from models.schemas import (
    Competitor,
    MarketSignals,
    MenuComparison,
    PlannerOutput,
    PricingAnalysis,
    ResearcherOutput,
    RestaurantMenu,
    SentimentAnalysis,
    SentimentBreakdown,
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
    
    Responsibility: Execute the research plan end-to-end using ReAct loop.
    
    Loop: Think → Act → Observe → Iterate
    
    Tools Available:
    - Google Maps / Places API
    - Tavily Search
    - Web Scraper (Extract & Crawl)
    """

    MAX_REACT_ITERATIONS = 15  # Safety limit for ReAct loop

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config)
        self.tools = self._initialize_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

    def _initialize_tools(self) -> list:
        """Initialize all research tools."""
        tools = []
        
        # Google Maps tools
        try:
            tools.extend(create_google_maps_tools(self.config.google_maps_api_key))
            logger.info("[RESEARCHER] Loaded Google Maps tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Google Maps tools unavailable: {e}")

        # Tavily Search tools
        try:
            tools.extend(create_tavily_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Tavily Search tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Tavily Search tools unavailable: {e}")

        # Tavily Extract tools (for URL content extraction)
        try:
            tools.extend(create_tavily_extract_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Tavily Extract tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Tavily Extract tools unavailable: {e}")

        # Web scraper tools (uses Tavily Extract under the hood)
        try:
            tools.extend(create_scraper_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Web Scraper tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Web Scraper tools unavailable: {e}")

        return tools

    def research(self, plan: PlannerOutput) -> ResearcherOutput:
        """
        Execute research based on the planner's output.
        Uses ReAct loop: Think → Act → Observe → Iterate
        
        Args:
            plan: Structured research plan from Planner
            
        Returns:
            ResearcherOutput with all gathered intelligence
        """
        logger.info(f"[RESEARCHER] Starting research for {plan.target_restaurant}")
        logger.info(f"[RESEARCHER] Location: {plan.location}")
        logger.info(f"[RESEARCHER] Intent: {plan.intent}")

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize conversation with plan context
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

            # Get LLM response
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # Check if LLM is done (no tool calls)
            if not response.tool_calls:
                logger.info("[RESEARCHER] No more tool calls - synthesizing results")
                break

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call.get("id", f"call_{iteration}")

                logger.info(f"[RESEARCHER] Calling tool: {tool_name}")
                logger.debug(f"[RESEARCHER] Tool args: {tool_args}")

                try:
                    tool = self.tool_map.get(tool_name)
                    if tool:
                        result = tool.invoke(tool_args)
                        self._collect_data(collected_data, tool_name, result)
                        
                        # Record source
                        collected_data["sources"].append(
                            SourceReference(
                                source_type=self._get_source_type(tool_name),
                                title=f"{tool_name} call",
                                data_summary=str(result)[:200],
                                accessed_at=datetime.now().isoformat(),
                            )
                        )
                    else:
                        result = f"Tool {tool_name} not found"
                        logger.warning(result)
                except Exception as e:
                    result = f"Tool error: {str(e)}"
                    logger.error(f"[RESEARCHER] Tool {tool_name} error: {e}")

                # Add tool result to messages
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )

        # Ask LLM to synthesize final output
        synthesis_prompt = self._create_synthesis_prompt(collected_data, plan)
        messages.append(HumanMessage(content=synthesis_prompt))
        
        final_response = self.llm.invoke(messages)
        
        # Parse and structure the output
        output = self._parse_output(final_response.content, collected_data, plan)
        
        logger.info(
            f"[RESEARCHER] Research complete. Found {len(output.competitors)} competitors"
        )
        return output

    def _format_plan_message(self, plan: PlannerOutput) -> str:
        """Format the plan into a message for the researcher."""
        cuisine_info = f"**Cuisine Type:** {plan.cuisine_type}" if plan.cuisine_type and plan.cuisine_type != "unknown" else "**Cuisine Type:** (to be determined from research)"
        
        return f"""## Research Plan

**Target Restaurant:** {plan.target_restaurant}
**Location:** {plan.location}
{cuisine_info}
**Intent:** {plan.intent}

**Research Queries:**
1. {plan.search_queries[0]}
2. {plan.search_queries[1]}
3. {plan.search_queries[2]}
4. {plan.search_queries[3]}

## IMPORTANT REQUIREMENTS:

1. **Find competitors of the SAME cuisine type** ({plan.cuisine_type or 'same as target'})
   - Use find_competitors with cuisine_type="{plan.cuisine_type or 'restaurant'}"
   - Only include direct competitors (same cuisine category)

2. **Scrape menus for TARGET + TOP 3 COMPETITORS**:
   - Search for "{plan.target_restaurant} menu" and extract the menu page
   - For each of the top 3 competitors by rating:
     a. Search for "[Competitor Name] menu"
     b. Use extract_menu_page to get full menu content
     c. Analyze items and prices

3. **Get reviews for target restaurant** using get_restaurant_reviews

Begin your research by first finding the target restaurant on Google Maps, then discovering nearby {plan.cuisine_type or ''} competitors."""

    def _get_source_type(self, tool_name: str) -> str:
        """Map tool name to source type."""
        if "restaurant" in tool_name or "competitor" in tool_name or "review" in tool_name:
            return "google_maps"
        elif "search" in tool_name or "traffic" in tool_name or "market" in tool_name:
            return "tavily"
        elif "extract" in tool_name or "crawl" in tool_name:
            return "website_scrape"
        return "api"

    def _collect_data(
        self, collected_data: dict, tool_name: str, result: str
    ) -> None:
        """Collect and organize data from tool results."""
        try:
            # Try to parse as dict/list
            if isinstance(result, str):
                # Handle string representations of dicts/lists
                if result.startswith("{") or result.startswith("["):
                    try:
                        parsed = eval(result)  # Safe for our structured outputs
                    except:
                        parsed = result
                else:
                    parsed = result
            else:
                parsed = result

            # Route to appropriate collection
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

    def _create_synthesis_prompt(
        self, collected_data: dict, plan: PlannerOutput
    ) -> str:
        """Create prompt for LLM to synthesize findings."""
        return f"""## Synthesis Request

You have completed your research. Now synthesize ALL findings into a structured JSON output.

**Collected Data Summary:**
- Competitors found: {len(collected_data.get('competitors', []))}
- Reviews collected: {len(collected_data.get('reviews', []))}
- Menu data points: {len(collected_data.get('menus', []))}
- Market data points: {len(collected_data.get('market_data', []))}
- Sources used: {len(collected_data.get('sources', []))}

**Required Output Structure:**
{{
    "competitors": [...],  // List of competitor objects
    "menu_comparison": {{...}},  // Menu analysis
    "pricing_analysis": {{...}},  // Pricing position
    "sentiment_analysis": {{...}},  // Review sentiment
    "market_signals": {{...}},  // Market indicators
    "research_notes": [...]  // Key observations
}}

Synthesize everything you learned about {plan.target_restaurant} in {plan.location}.
Output ONLY the JSON - no additional text."""

    def _parse_output(
        self,
        response_text: str,
        collected_data: dict,
        plan: PlannerOutput,
    ) -> ResearcherOutput:
        """Parse LLM synthesis into structured output."""
        # Try to extract JSON from response
        try:
            json_data = self._extract_json(response_text)
        except:
            json_data = {}

        # Build competitors list from collected data
        competitors = []
        for c in collected_data.get("competitors", []):
            if isinstance(c, dict):
                try:
                    competitors.append(
                        Competitor(
                            name=c.get("name", "Unknown"),
                            address=c.get("address", ""),
                            distance_miles=c.get("distance_miles", 0.0),
                            rating=c.get("rating"),
                            review_count=c.get("review_count"),
                            price_level=c.get("price_level"),
                            cuisine_type=c.get("cuisine_type", "restaurant"),
                            website=c.get("website"),
                            place_id=c.get("place_id"),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error parsing competitor: {e}")

        # Build sentiment from reviews
        sentiment = self._analyze_sentiment(collected_data.get("reviews", []))

        # Build menu comparison
        menu_comparison = self._build_menu_comparison(
            collected_data.get("menus", []), plan.target_restaurant
        )

        # Build pricing analysis
        pricing = self._build_pricing_analysis(competitors, collected_data.get("menus", []))

        # Build market signals
        market_signals = self._build_market_signals(
            competitors, collected_data.get("market_data", [])
        )

        # Extract research notes from LLM response
        research_notes = json_data.get("research_notes", [])
        if not research_notes and isinstance(json_data, dict):
            research_notes = [
                f"Analyzed {len(competitors)} competitors in {plan.location}",
                f"Research intent: {plan.intent}",
            ]

        return ResearcherOutput(
            competitors=competitors[:self.config.max_competitors],
            menu_comparison=menu_comparison,
            pricing_analysis=pricing,
            sentiment_analysis=sentiment,
            market_signals=market_signals,
            raw_sources=collected_data.get("sources", []),
            research_notes=research_notes,
        )

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from text."""
        import re

        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass

        # Find JSON block
        json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        # Find raw JSON
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end])
            except:
                pass

        return {}

    def _analyze_sentiment(self, reviews: list) -> SentimentAnalysis:
        """Analyze sentiment from collected reviews."""
        if not reviews:
            return SentimentAnalysis(
                positive_drivers=["Insufficient review data"],
                common_complaints=["No reviews collected"],
            )

        # Extract ratings and text
        ratings = []
        positive_themes = []
        negative_themes = []

        positive_keywords = [
            "great", "excellent", "delicious", "amazing", "best",
            "love", "fantastic", "friendly", "recommend", "fresh",
        ]
        negative_keywords = [
            "slow", "cold", "expensive", "rude", "wait", "disappointing",
            "mediocre", "overpriced", "dirty", "bad",
        ]

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

        # Calculate overall sentiment
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        overall = avg_rating / 5 if avg_rating else None

        # Get top themes
        from collections import Counter
        top_positive = [t[0] for t in Counter(positive_themes).most_common(5)]
        top_negative = [t[0] for t in Counter(negative_themes).most_common(5)]

        return SentimentAnalysis(
            target_overall_sentiment=overall,
            target_sentiment_breakdown=SentimentBreakdown(
                food_quality=overall,
                service=overall,
            ),
            positive_drivers=top_positive if top_positive else ["Good overall ratings"],
            common_complaints=top_negative if top_negative else ["No major complaints identified"],
            sample_reviews=reviews[:3],
        )

    def _build_menu_comparison(
        self, menus: list, target_name: str
    ) -> MenuComparison:
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
                items = menu_data.get("items", [])
                
                menu = RestaurantMenu(
                    restaurant_name=name,
                    items=[],
                    signature_items=items[:3] if items else [],
                )
                
                if target_name.lower() in name.lower():
                    target_menu = menu
                else:
                    competitor_menus.append(menu)

        return MenuComparison(
            target_menu=target_menu,
            competitor_menus=competitor_menus,
            menu_breadth_comparison={},
            unique_offerings=[],
        )

    def _build_pricing_analysis(
        self, competitors: list[Competitor], menus: list
    ) -> PricingAnalysis:
        """Build pricing analysis from competitor data."""
        price_levels = []
        for c in competitors:
            if c.price_level:
                # Convert $ symbols to numbers
                level = len(c.price_level.replace(" ", ""))
                price_levels.append(level)

        avg_level = sum(price_levels) / len(price_levels) if price_levels else 2

        # Map to position
        if avg_level <= 1.5:
            position = "budget"
        elif avg_level <= 2.5:
            position = "mid-range"
        elif avg_level <= 3.5:
            position = "premium"
        else:
            position = "luxury"

        return PricingAnalysis(
            price_position=position,
            competitor_price_range={
                c.name: {"level": c.price_level or "$$"}
                for c in competitors[:5]
            },
        )

    def _build_market_signals(
        self, competitors: list[Competitor], market_data: list
    ) -> MarketSignals:
        """Build market signals from all collected data."""
        competitor_count = len(competitors)
        
        # Determine saturation
        if competitor_count <= 3:
            saturation = "low"
        elif competitor_count <= 8:
            saturation = "moderate"
        elif competitor_count <= 15:
            saturation = "high"
        else:
            saturation = "oversaturated"

        # Average competitor rating
        ratings = [c.rating for c in competitors if c.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        # Extract signals from market data
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

        return MarketSignals(
            competitor_density=competitor_count,
            market_saturation=saturation,
            avg_competitor_rating=avg_rating,
            foot_traffic_indicators=foot_traffic[:5] if foot_traffic else ["Foot traffic data not available"],
            growth_indicators=growth[:5] if growth else ["Growth signals not found"],
            risk_indicators=risks[:5] if risks else ["No significant risk signals"],
        )

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        planner_output = state.get("planner_output")
        if not planner_output:
            raise ValueError("No planner_output found in state")

        # Convert dict to PlannerOutput if needed
        if isinstance(planner_output, dict):
            planner_output = PlannerOutput(**planner_output)

        try:
            output = self.research(planner_output)
            return {
                **state,
                "researcher_output": output,
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

