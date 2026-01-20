"""RESEARCHER Agent - Autonomous ReAct-style research execution."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
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
    BusinessBackground,
    SentimentQuality,
)
from prompts.researcher import RESEARCHER_SYSTEM_PROMPT
from tools.google_maps import create_google_maps_tools
from tools.tavily_search import create_tavily_tools
from tools.tavily_extract import create_tavily_extract_tools
from tools.web_scraper import create_scraper_tools
from utils import create_llm


class ResearcherAgent:
    """
    NODE 2 — RESEARCHER AGENT (AUTONOMOUS)
    
    Executes research plan using a strict ReAct loop (Thought -> Action -> Observation).
    Returns validated ResearcherOutput Pydantic model.
    """

    MAX_REACT_ITERATIONS = 20  # Increased for "World Class" depth

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config)
        self.tools = self._initialize_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

    def _initialize_tools(self) -> list:
        """Initialize all research tools."""
        tools = []
        
        # 1. Google Maps (The "Gold Standard" for location data)
        try:
            tools.extend(create_google_maps_tools(self.config.google_maps_api_key))
            logger.info("[RESEARCHER] Loaded Google Maps tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Google Maps tools unavailable: {e}")

        # 2. Tavily Search (Broader context)
        try:
            tools.extend(create_tavily_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Tavily Search tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Tavily Search tools unavailable: {e}")

        # 3. Tavily Extract (Deep dive content)
        try:
            tools.extend(create_tavily_extract_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Tavily Extract tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Tavily Extract tools unavailable: {e}")

        # 4. Scraper (Fallback/Specific needs)
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
        
        logger.info(f"[RESEARCHER] 🔍 Starting WORLD CLASS research for {target} in {location}")

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize Context
        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=self._format_plan_message(plan)),
        ]

        # Data Store (The "Brain" of the operation)
        collected_data = {
            "target": None,
            "competitors": [],
            "reviews": {},  # {restaurant_name: {"rating": X, "count": Y, "reviews": [...]}}
            "menus": [],    # List of parsed/raw menu data
            "market_data": [],
            "sources": [],  # Audit trail
        }

        # --- ReAct Execution Loop ---
        iteration = 0
        while iteration < self.MAX_REACT_ITERATIONS:
            iteration += 1
            logger.info(f"[RESEARCHER] 🔄 Iteration {iteration}/{self.MAX_REACT_ITERATIONS}")

            # 1. THOUGHT (Invoke LLM)
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # Log the "Thought" if present in content
            if response.content:
                logger.info(f"[RESEARCHER] 💭 THOUGHT: {str(response.content)[:300]}...")

            # 2. ACTION (Check for tool calls)
            if not response.tool_calls:
                logger.info("[RESEARCHER] ✅ No more tool calls - Research Complete.")
                break

            # Execute Tools
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call.get("id", f"call_{iteration}")

                logger.info(f"[RESEARCHER] 🛠️ CALLING: {tool_name}")
                
                try:
                    tool = self.tool_map.get(tool_name)
                    if tool:
                        # Invoke Tool
                        result = tool.invoke(tool_args)
                        
                        # 3. OBSERVATION (Collect Data)
                        self._collect_data(collected_data, tool_name, tool_args, result)
                        
                        # Audit Trail
                        collected_data["sources"].append({
                            "source_type": self._get_source_type(tool_name),
                            "title": f"Tool: {tool_name}",
                            "url": tool_args.get("url") or "API",
                            "data_summary": str(result)[:200] + "...",
                            "accessed_at": datetime.now().isoformat(),
                        })
                    else:
                        result = f"Error: Tool '{tool_name}' not found."
                        logger.error(f"[RESEARCHER] ❌ {result}")
                except Exception as e:
                    result = f"Error executing tool '{tool_name}': {str(e)}"
                    logger.error(f"[RESEARCHER] ❌ {result}")

                # Append Observation to history
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        # --- Synthesis Phase ---
        logger.info("[RESEARCHER] 🧠 Synthesizing collected data...")
        
        # 1. Sentiment Analysis (LLM-based)
        sentiment_analysis = self._analyze_sentiment_with_llm(collected_data)
        
        # 2. Structure & Validate Output
        output = self._build_output(collected_data, plan, sentiment_analysis)
        
        logger.info(f"[RESEARCHER] 🎉 Research Complete. Competitors: {len(output.competitors)}")
        return output

    def _format_plan_message(self, plan: dict[str, Any]) -> str:
        """Format the plan into a strict mission briefing."""
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        cuisine = plan.get("cuisine_type", "Unknown")
        queries = plan.get("search_queries", [])
        
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
        
        return f"""## 🚀 RESEARCH MISSION BRIEF

**TARGET**: {target}
**LOCATION**: {location}
**CUISINE**: {cuisine}

**YOUR OBJECTIVES (Execute in Order):**
{queries_text}

**START NOW.** Review your tools, think about the first step (finding the target), and Begin."""

    def _get_source_type(self, tool_name: str) -> str:
        if "google" in tool_name or "restaurant" in tool_name: return "google_maps"
        if "tavily" in tool_name or "search" in tool_name: return "search"
        if "extract" in tool_name or "scrape" in tool_name: return "website"
        return "system"

    def _collect_data(self, collected_data: dict, tool_name: str, tool_args: dict, result: Any) -> None:
        """Intelligent data collection router."""
        try:
            # Parse result if it's a JSON string
            parsed = result
            if isinstance(result, str):
                if result.strip().startswith(("{", "[")):
                    try:
                        parsed = json.loads(result)
                    except:
                        pass # Keep as string if parse fails

            # Route by Tool Type
            tool_lower = tool_name.lower()
            
            # --- COMPETITOR / TARGET IDENTIFICATION ---
            if "find_competitor" in tool_lower:
                if isinstance(parsed, dict) and "competitors" in parsed:
                    found_count = len(parsed["competitors"])
                    logger.info(f"[RESEARCHER] 📥 Collected {found_count} competitors from tool.")
                    collected_data["competitors"].extend(parsed["competitors"])
            
            elif "find_restaurant" in tool_lower:
                if isinstance(parsed, dict) and parsed.get("name"):
                    collected_data["target"] = parsed

            # --- REVIEWS ---
            elif "review" in tool_lower:
                place_id = tool_args.get("place_id") or "unknown"
                restaurant_name = self._find_restaurant_name(collected_data, place_id)
                # Store under the restaurant name for easier lookup
                collected_data["reviews"][restaurant_name] = {
                    "place_id": place_id,
                    "reviews": parsed if isinstance(parsed, list) else [],
                    # We might get rating/count if the tool returns a wrapper, 
                    # but usually get_restaurant_reviews returns just a list.
                    # We rely on find_restaurant/competitors for the aggregate stats.
                }

            # --- MENUS ---
            elif "menu" in tool_lower:
                # Add metadata to the menu payload if missing
                if isinstance(parsed, dict):
                    if "restaurant_name" not in parsed:
                        # Try to infer from tool args
                        parsed["restaurant_name"] = tool_args.get("restaurant_name", "Unknown")
                    if "source_url" not in parsed:
                        parsed["source_url"] = tool_args.get("url", "Unknown")
                
                collected_data["menus"].append(parsed)

            # --- GENERAL MARKET / BUSINESS INFO ---
            elif "search" in tool_lower or "extract" in tool_lower:
                # General bucket for business background and market signals
                collected_data["market_data"].append({
                    "tool": tool_name,
                    "query": tool_args.get("query") or tool_args.get("url"),
                    "content": parsed
                })

        except Exception as e:
            logger.warning(f"[RESEARCHER] Data collection warning: {e}")

    def _find_restaurant_name(self, collected_data: dict, place_id: str) -> str:
        """Helper to resolve place_id to name."""
        # 1. Check Target
        if collected_data.get("target") and collected_data["target"].get("place_id") == place_id:
            return collected_data["target"].get("name", "Target")
        
        # 2. Check Competitors
        for comp in collected_data.get("competitors", []):
            if comp.get("place_id") == place_id:
                return comp.get("name", "Unknown Competitor")
        
        return f"Unknown ({place_id[:5]})"

    # --------------------------------------------------------------------------
    # ANALYTIC METHODS (LLM-POWERED)
    # --------------------------------------------------------------------------

    def _analyze_sentiment_with_llm(self, collected_data: dict) -> dict:
        """Synthesize sentiment from collected reviews."""
        reviews_map = collected_data.get("reviews", {})
        if not reviews_map:
            return {}

        prompt = "Analyze the customer sentiment for the following restaurants based on their reviews:\n\n"
        
        for name, data in reviews_map.items():
            review_list = data.get("reviews", [])
            # Take top 10 reviews for analysis to save tokens
            review_text = "\n".join([f"- {r.get('text', '')[:200]}" for r in review_list[:10] if isinstance(r, dict)])
            prompt += f"=== RESTAURANT: {name} ===\n{review_text}\n\n"

        prompt += """
        For EACH restaurant, provide a JSON output with:
        - sentiment_score (0.0 to 1.0)
        - service_quality (score 0-1, summary)
        - food_quality (score 0-1, summary)
        - atmosphere (score 0-1, summary)
        - value_perception (score 0-1, summary)
        - key_strengths (list)
        - key_concerns (list)
        
        RETURN A JSON OBJECT WITH KEY 'restaurants' containing a list of these objects.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._extract_json(response.content) or {}
        except Exception as e:
            logger.error(f"[RESEARCHER] Sentiment analysis failed: {e}")
            return {}

    def _analyze_business_background(self, collected_data: dict, target_name: str) -> dict:
        """Analyze business background from general search results."""
        market_data = collected_data.get("market_data", [])
        
        # Filter for relevant chunks (simple keyword match)
        relevant_text = []
        for item in market_data:
            content = str(item.get("content", ""))
            if any(k in content.lower() for k in ["founded", "owner", "opened", "history", "story"]):
                relevant_text.append(content[:1000])
        
        if not relevant_text:
            return {}

        prompt = f"""Extract business background for '{target_name}' from these snippets:\n\n
        {chr(10).join(relevant_text[:5])}
        
        Return JSON:
        {{
            "founding_year": "YYYY",
            "founders": ["Name"],
            "history_summary": "...",
            "expansion_strategy": "..."
        }}
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._extract_json(response.content) or {}
        except Exception:
            return {}

    def _build_menu_comparison(self, menus: list, target_name: str) -> MenuComparison:
        """Compare menus using LLM."""
        if not menus:
            return MenuComparison(pricing_summary="No menu data collected.")

        try:
            # Simple heuristic to identify target menu
            target_menu = next((m for m in menus if target_name.lower() in str(m.get("restaurant_name", "")).lower()), None)
            competitor_menus = [m for m in menus if m != target_menu]
            
            # If no explicit target match, assume first is target (if reasonable) or handle gracefully
            if not target_menu and menus:
                target_menu = menus[0]
                competitor_menus = menus[1:]

            # Prepare content for LLM
            prompt = f"""COMPARE THESE MENUS. 
            Target: {target_menu.get("restaurant_name") if target_menu else "Unknown"}
            Competitors: {[c.get("restaurant_name") for c in competitor_menus]}
            
            [Target Menu Content]: {str(target_menu)[:15000]}
            
            [Competitor Menus Content]: {str(competitor_menus)[:15000]}
            
            OUTPUT JSON conforming to MenuComparison schema:
            {{
                "item_comparisons": [
                    {{"item_name": "...", "target_price": 10.0, "competitor_prices": {{"Comp1": 9.0}}, "price_difference_avg": 1.0}}
                ],
                "pricing_summary": "...",
                "unique_offerings": ["..."]
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            data = self._extract_json(response.content)
            
            if data:
                return MenuComparison(**data)
        
        except Exception as e:
            logger.error(f"[RESEARCHER] Menu comparison failed: {e}")
        
        return MenuComparison(pricing_summary="Menu comparison could not be completed.")

    # --------------------------------------------------------------------------
    # OUTPUT CONSTRUCTION
    # --------------------------------------------------------------------------

    def _build_output(self, collected_data: dict, plan: dict, sentiment_analysis: dict) -> ResearcherOutput:
        """Construct the final Pydantic model."""
        
        # 1. Target Info
        target_raw = collected_data.get("target") or {}
        target_info = RestaurantInfo(
            name=target_raw.get("name", plan.get("target_restaurant", "Unknown")),
            address=target_raw.get("address", ""),
            place_id=target_raw.get("place_id"),
            rating=target_raw.get("rating"),
            review_count=target_raw.get("user_ratings_total"),
            price_level=str(target_raw.get("price_level", "")),
            cuisine_type=plan.get("cuisine_type", "Restaurant")
        )
        
        # Enrich target with business background
        bg_data = self._analyze_business_background(collected_data, target_info.name)
        if bg_data:
            target_info.business_background = BusinessBackground(**bg_data)

        # 2. Competitors
        competitors = []
        for c in collected_data.get("competitors", []):
            competitors.append(CompetitorInfo(
                name=c.get("name", "Unknown"),
                address=c.get("address", ""),
                rating=c.get("rating"),
                review_count=c.get("user_ratings_total"),
                price_level=str(c.get("price_level", "")),
                place_id=c.get("place_id")
            ))

        # 3. Market Signals (Simple heuristic + data)
        market_signals = MarketSignals(
            competitor_density=len(competitors),
            market_saturation="high" if len(competitors) > 10 else "moderate",
            avg_competitor_rating=sum(c.rating for c in competitors if c.rating)/len(competitors) if competitors else 0.0
        )

        # 4. Sentiment (Map LLM raw response to objects)
        sentiment_objects = []
        raw_sentiment = sentiment_analysis.get("restaurants", [])
        for r in raw_sentiment:
            sentiment_objects.append(RestaurantSentiment(
                name=r.get("name", "Unknown"),
                sentiment_score=r.get("sentiment_score", 0.5),
                summary=r.get("summary", ""),
                key_strengths=r.get("key_strengths", []),
                key_concerns=r.get("key_concerns", []),
                service_quality=SentimentQuality(**r.get("service_quality", {})),
                food_quality=SentimentQuality(**r.get("food_quality", {})),
                atmosphere=SentimentQuality(**r.get("atmosphere", {})),
                value_perception=SentimentQuality(**r.get("value_perception", {}))
            ))

        final_sentiment = SentimentAnalysis(
            restaurants=sentiment_objects,
            comparative_summary="Generated from detailed reviews."
        )

        # 5. Menu Comparison
        menu_comparison = self._build_menu_comparison(collected_data.get("menus", []), target_info.name)

        # 6. Pricing Analysis (Derived from menus + signals)
        pricing_analysis = PricingAnalysis(
            price_position=menu_comparison.pricing_summary, # Simplified mapping
            competitor_price_range={c.name: c.price_level for c in competitors}
        )

        # 7. Sources
        sources = [SourceReference(**s) for s in collected_data.get("sources", [])]

        return ResearcherOutput(
            target=target_info,
            competitors=competitors,
            menu_comparison=menu_comparison,
            pricing_analysis=pricing_analysis,
            sentiment_analysis=final_sentiment,
            market_signals=market_signals,
            raw_sources=sources,
            research_notes=["Use Audit data for details."]
        )

    def _extract_json(self, text: str) -> dict | None:
        """Robust JSON extractor."""
        try:
            # 1. Direct parsing
            return json.loads(text)
        except:
            pass
        
        try:
            # 2. Markdown block extraction
            match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip())
        except:
            pass

        return None
