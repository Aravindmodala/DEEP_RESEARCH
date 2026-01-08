"""RESEARCHER Agent - Autonomous ReAct-style research execution with state caching."""

from __future__ import annotations

import hashlib
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
from state import AgentGraphState, ResearchCache, ResearchProgress, get_cache, get_progress
from tools.google_maps import create_google_maps_tools
from tools.tavily_extract import create_tavily_extract_tools
from tools.tavily_search import create_tavily_tools
from tools.web_scraper import create_scraper_tools
from utils import create_llm


# Maximum characters per tool result to prevent token overflow
MAX_TOOL_RESULT_CHARS = 4000


def truncate_result(result: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate tool result to prevent token overflow."""
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + f"\n\n[...truncated {len(result) - max_chars} chars]"


def hash_query(query: str) -> str:
    """Create a hash key for caching search results."""
    return hashlib.md5(query.encode()).hexdigest()[:12]


class ResearcherAgent:
    """
    NODE 2 — RESEARCHER AGENT (AUTONOMOUS)
    
    Features:
    - State caching to avoid duplicate API calls
    - Progress tracking for resumable research
    - Token-safe tool result truncation
    - ReAct loop: Think → Act → Observe → Iterate
    """

    MAX_REACT_ITERATIONS = 10  # Reduced for efficiency

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

        # Tavily Extract tools
        try:
            tools.extend(create_tavily_extract_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Tavily Extract tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Tavily Extract tools unavailable: {e}")

        # Web scraper tools
        try:
            tools.extend(create_scraper_tools(self.config.tavily_api_key))
            logger.info("[RESEARCHER] Loaded Web Scraper tools")
        except Exception as e:
            logger.warning(f"[RESEARCHER] Web Scraper tools unavailable: {e}")

        return tools

    def research(
        self,
        plan: PlannerOutput,
        cache: ResearchCache | None = None,
        progress: ResearchProgress | None = None,
    ) -> tuple[ResearcherOutput, ResearchCache, ResearchProgress]:
        """
        Execute research with caching support.
        
        Args:
            plan: Structured research plan from Planner
            cache: Existing cache to resume from (optional)
            progress: Existing progress to resume from (optional)
            
        Returns:
            Tuple of (ResearcherOutput, updated_cache, updated_progress)
        """
        logger.info(f"[RESEARCHER] Starting research for {plan.target_restaurant}")
        logger.info(f"[RESEARCHER] Location: {plan.location}")
        logger.info(f"[RESEARCHER] Cuisine: {plan.cuisine_type}")

        # Initialize or use existing cache/progress
        cache = cache or ResearchCache(
            target_restaurant=None,
            competitors=[],
            menus={},
            reviews={},
            search_results={},
            market_data=[],
        )
        progress = progress or ResearchProgress(
            target_found=False,
            competitors_found=False,
            target_menu_extracted=False,
            competitor_menus_extracted=False,
            reviews_collected=False,
            market_signals_gathered=False,
            synthesis_complete=False,
        )

        # Log existing progress
        completed = [k for k, v in progress.items() if v]
        if completed:
            logger.info(f"[RESEARCHER] Resuming with completed steps: {completed}")

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize conversation - include cache summary if resuming
        cache_context = self._format_cache_context(cache, progress)
        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=self._format_plan_message(plan, cache_context)),
        ]

        # ReAct loop
        iteration = 0
        while iteration < self.MAX_REACT_ITERATIONS:
            iteration += 1
            logger.info(f"[RESEARCHER] ReAct iteration {iteration}/{self.MAX_REACT_ITERATIONS}")

            try:
                response = llm_with_tools.invoke(messages)
                messages.append(response)
            except Exception as e:
                logger.error(f"[RESEARCHER] LLM error: {e}")
                # On error, try to synthesize from what we have
                break

            # Check if LLM is done
            if not response.tool_calls:
                logger.info("[RESEARCHER] No more tool calls - synthesis phase")
                break

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call.get("id", f"call_{iteration}")

                logger.info(f"[RESEARCHER] Tool: {tool_name}")
                
                try:
                    # Check cache first for certain tools
                    cached_result = self._check_cache(cache, tool_name, tool_args)
                    
                    if cached_result is not None:
                        logger.info(f"[RESEARCHER] Using cached result for {tool_name}")
                        result = cached_result
                    else:
                        # Execute tool
                        tool = self.tool_map.get(tool_name)
                        if tool:
                            result = tool.invoke(tool_args)
                            # Update cache
                            self._update_cache(cache, progress, tool_name, tool_args, result)
                        else:
                            result = f"Tool {tool_name} not found"
                            logger.warning(result)

                    # Truncate result to prevent token overflow
                    result_str = truncate_result(str(result))
                    
                except Exception as e:
                    result_str = f"Tool error: {str(e)[:200]}"
                    logger.error(f"[RESEARCHER] {tool_name} error: {e}")

                messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))

        # Synthesize output from cache
        output = self._synthesize_from_cache(cache, progress, plan, messages)
        progress["synthesis_complete"] = True

        logger.info(f"[RESEARCHER] Complete. {len(output.competitors)} competitors found.")
        return output, cache, progress

    def _format_cache_context(self, cache: ResearchCache, progress: ResearchProgress) -> str:
        """Format cache contents for context injection."""
        if not any(progress.values()):
            return ""

        lines = ["## Previously Collected Data (use this, don't re-fetch):"]
        
        if progress.get("target_found") and cache.get("target_restaurant"):
            target = cache["target_restaurant"]
            lines.append(f"- Target restaurant found: {target.get('name', 'Unknown')}")
            lines.append(f"  Place ID: {target.get('place_id', 'N/A')}")
        
        if progress.get("competitors_found") and cache.get("competitors"):
            lines.append(f"- {len(cache['competitors'])} competitors already found")
            for c in cache["competitors"][:3]:
                lines.append(f"  • {c.get('name', 'Unknown')} ({c.get('rating', 'N/A')}★)")
        
        if progress.get("target_menu_extracted") and cache.get("menus"):
            lines.append(f"- {len(cache['menus'])} menus already extracted")
        
        if progress.get("reviews_collected") and cache.get("reviews"):
            total_reviews = sum(len(r) for r in cache["reviews"].values())
            lines.append(f"- {total_reviews} reviews already collected")

        return "\n".join(lines)

    def _format_plan_message(self, plan: PlannerOutput, cache_context: str = "") -> str:
        """Format the plan into a message for the researcher."""
        cuisine = plan.cuisine_type if plan.cuisine_type != "unknown" else "restaurant"
        
        base = f"""## Research Plan

**Target Restaurant:** {plan.target_restaurant}
**Location:** {plan.location}
**Cuisine Type:** {cuisine}
**Intent:** {plan.intent}

**Research Queries:**
1. {plan.search_queries[0]}
2. {plan.search_queries[1]}
3. {plan.search_queries[2]}
4. {plan.search_queries[3]}

{cache_context}

## Requirements:

1. **Find same-cuisine competitors** using find_competitors with cuisine_type="{cuisine}"
2. **Extract menus** for target + top 3 competitors
3. **Collect reviews** for sentiment analysis
4. **Gather market signals** for lending assessment

Begin research now. Skip any steps already completed (see above)."""
        return base

    def _check_cache(
        self, cache: ResearchCache, tool_name: str, tool_args: dict
    ) -> str | None:
        """Check if we have cached results for this tool call."""
        # Check for cached competitor search
        if tool_name == "find_competitors" and cache.get("competitors"):
            return json.dumps({
                "competitors": cache["competitors"],
                "message": "Using cached competitor data"
            })
        
        # Check for cached target restaurant
        if tool_name == "find_restaurant" and cache.get("target_restaurant"):
            return json.dumps(cache["target_restaurant"])
        
        # Check for cached search results
        if "search" in tool_name and cache.get("search_results"):
            query = tool_args.get("query", "")
            query_hash = hash_query(query)
            if query_hash in cache["search_results"]:
                return json.dumps(cache["search_results"][query_hash])
        
        return None

    def _update_cache(
        self,
        cache: ResearchCache,
        progress: ResearchProgress,
        tool_name: str,
        tool_args: dict,
        result: Any,
    ) -> None:
        """Update cache with tool results."""
        try:
            # Parse result if string
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except:
                    parsed = result
            else:
                parsed = result

            # Update based on tool type
            if tool_name == "find_restaurant":
                if isinstance(parsed, dict):
                    cache["target_restaurant"] = parsed
                    progress["target_found"] = True
                    logger.debug("[CACHE] Saved target restaurant")

            elif tool_name == "find_competitors":
                if isinstance(parsed, dict) and "competitors" in parsed:
                    cache["competitors"] = parsed["competitors"]
                    progress["competitors_found"] = True
                    logger.debug(f"[CACHE] Saved {len(parsed['competitors'])} competitors")

            elif "menu" in tool_name.lower() or "extract" in tool_name.lower():
                restaurant_name = tool_args.get("restaurant_name", "unknown")
                if isinstance(parsed, (str, dict)):
                    cache["menus"][restaurant_name] = str(parsed)[:8000]  # Limit size
                    if "target" in restaurant_name.lower() or progress.get("target_found"):
                        progress["target_menu_extracted"] = True
                    else:
                        progress["competitor_menus_extracted"] = True
                    logger.debug(f"[CACHE] Saved menu for {restaurant_name}")

            elif "review" in tool_name.lower():
                place_id = tool_args.get("place_id", "default")
                if isinstance(parsed, list):
                    cache["reviews"][place_id] = parsed[:20]  # Limit count
                    progress["reviews_collected"] = True
                    logger.debug(f"[CACHE] Saved {len(parsed)} reviews")

            elif "search" in tool_name.lower():
                query = tool_args.get("query", "")
                query_hash = hash_query(query)
                if isinstance(parsed, list):
                    cache["search_results"][query_hash] = parsed[:10]  # Limit count
                elif isinstance(parsed, dict):
                    cache["search_results"][query_hash] = parsed
                progress["market_signals_gathered"] = True
                logger.debug(f"[CACHE] Saved search results for '{query[:30]}...'")

            elif "market" in tool_name.lower() or "traffic" in tool_name.lower():
                if isinstance(parsed, (list, dict)):
                    cache["market_data"].append(parsed)
                    progress["market_signals_gathered"] = True

        except Exception as e:
            logger.warning(f"[CACHE] Error updating cache: {e}")

    def _synthesize_from_cache(
        self,
        cache: ResearchCache,
        progress: ResearchProgress,
        plan: PlannerOutput,
        messages: list,
    ) -> ResearcherOutput:
        """Synthesize output from cached data."""
        # Build competitors from cache
        competitors = []
        for c in cache.get("competitors", []):
            if isinstance(c, dict):
                try:
                    competitors.append(Competitor(
                        name=c.get("name", "Unknown"),
                        address=c.get("address", ""),
                        distance_miles=float(c.get("distance_miles", 0)),
                        rating=c.get("rating"),
                        review_count=c.get("review_count"),
                        price_level=c.get("price_level"),
                        cuisine_type=c.get("cuisine_type", plan.cuisine_type),
                        website=c.get("website"),
                        place_id=c.get("place_id"),
                    ))
                except Exception as e:
                    logger.debug(f"Skip competitor parse: {e}")

        # Build sentiment from cached reviews
        all_reviews = []
        for reviews in cache.get("reviews", {}).values():
            all_reviews.extend(reviews)
        sentiment = self._analyze_sentiment(all_reviews)

        # Build menu comparison from cache
        menu_comparison = self._build_menu_comparison(
            cache.get("menus", {}), plan.target_restaurant
        )

        # Build pricing analysis
        pricing = self._build_pricing_analysis(competitors, cache.get("menus", {}))

        # Build market signals
        market_signals = self._build_market_signals(
            competitors, cache.get("market_data", [])
        )

        # Sources
        sources = [
            SourceReference(
                source_type="google_maps",
                title="Google Places API",
                data_summary=f"Found {len(competitors)} competitors",
                accessed_at=datetime.now().isoformat(),
            ),
            SourceReference(
                source_type="tavily",
                title="Web Search",
                data_summary=f"Searched {len(cache.get('search_results', {}))} queries",
                accessed_at=datetime.now().isoformat(),
            ),
        ]

        research_notes = [
            f"Analyzed {len(competitors)} {plan.cuisine_type} competitors in {plan.location}",
            f"Extracted {len(cache.get('menus', {}))} menus",
            f"Collected {len(all_reviews)} reviews for sentiment analysis",
            f"Research intent: {plan.intent}",
        ]

        return ResearcherOutput(
            competitors=competitors[:self.config.max_competitors],
            menu_comparison=menu_comparison,
            pricing_analysis=pricing,
            sentiment_analysis=sentiment,
            market_signals=market_signals,
            raw_sources=sources,
            research_notes=research_notes,
        )

    def _analyze_sentiment(self, reviews: list) -> SentimentAnalysis:
        """Analyze sentiment from collected reviews."""
        if not reviews:
            return SentimentAnalysis(
                positive_drivers=["Insufficient review data"],
                common_complaints=["No reviews collected"],
            )

        ratings = []
        positive_themes = []
        negative_themes = []

        positive_kw = ["great", "excellent", "delicious", "amazing", "best", "love", "fantastic", "friendly", "recommend", "fresh"]
        negative_kw = ["slow", "cold", "expensive", "rude", "wait", "disappointing", "mediocre", "overpriced", "dirty", "bad"]

        for review in reviews:
            if isinstance(review, dict):
                if review.get("rating"):
                    ratings.append(float(review["rating"]))
                text = review.get("text", "").lower()
                for kw in positive_kw:
                    if kw in text:
                        positive_themes.append(kw)
                for kw in negative_kw:
                    if kw in text:
                        negative_themes.append(kw)

        avg_rating = sum(ratings) / len(ratings) if ratings else None
        overall = avg_rating / 5 if avg_rating else None

        from collections import Counter
        top_pos = [t[0] for t in Counter(positive_themes).most_common(5)] or ["Good ratings"]
        top_neg = [t[0] for t in Counter(negative_themes).most_common(5)] or ["No major complaints"]

        return SentimentAnalysis(
            target_overall_sentiment=overall,
            target_sentiment_breakdown=SentimentBreakdown(food_quality=overall, service=overall),
            positive_drivers=top_pos,
            common_complaints=top_neg,
            sample_reviews=reviews[:3],
        )

    def _build_menu_comparison(self, menus: dict[str, str], target_name: str) -> MenuComparison:
        """Build menu comparison from cached menus."""
        target_menu = None
        competitor_menus = []

        for name, content in menus.items():
            menu = RestaurantMenu(
                restaurant_name=name,
                items=[],
                signature_items=[content[:200]] if content else [],
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

    def _build_pricing_analysis(self, competitors: list[Competitor], menus: dict) -> PricingAnalysis:
        """Build pricing analysis from competitor data."""
        price_levels = []
        for c in competitors:
            if c.price_level:
                level = len(str(c.price_level).replace(" ", ""))
                price_levels.append(level)

        avg = sum(price_levels) / len(price_levels) if price_levels else 2
        position = "budget" if avg <= 1.5 else "mid-range" if avg <= 2.5 else "premium" if avg <= 3.5 else "luxury"

        return PricingAnalysis(
            price_position=position,
            competitor_price_range={c.name: {"level": c.price_level or "$$"} for c in competitors[:5]},
        )

    def _build_market_signals(self, competitors: list[Competitor], market_data: list) -> MarketSignals:
        """Build market signals from collected data."""
        count = len(competitors)
        saturation = "low" if count <= 3 else "moderate" if count <= 8 else "high" if count <= 15 else "oversaturated"

        ratings = [c.rating for c in competitors if c.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        foot_traffic = ["Data from Google Places API"]
        growth = ["Market signals collected via web search"]
        risks = []

        for data in market_data:
            if isinstance(data, dict):
                content = str(data.get("content", "")).lower()
                if "closing" in content or "decline" in content:
                    risks.append(data.get("title", "")[:50])

        return MarketSignals(
            competitor_density=count,
            market_saturation=saturation,
            avg_competitor_rating=avg_rating,
            foot_traffic_indicators=foot_traffic,
            growth_indicators=growth,
            risk_indicators=risks if risks else ["No significant risk signals"],
        )

    def __call__(self, state: AgentGraphState) -> AgentGraphState:
        """LangGraph-compatible call interface with state caching."""
        planner_output = state.get("planner_output")
        if not planner_output:
            raise ValueError("No planner_output in state")

        # Convert dict to PlannerOutput if needed
        if isinstance(planner_output, dict):
            planner_output = PlannerOutput(**planner_output)

        # Get existing cache and progress from state
        cache = get_cache(state)
        progress = get_progress(state)

        try:
            output, updated_cache, updated_progress = self.research(
                planner_output, cache, progress
            )
            
            return {
                **state,
                "researcher_output": output.model_dump(),
                "research_cache": updated_cache,
                "research_progress": updated_progress,
                "current_node": "critic",
                "iteration_count": state.get("iteration_count", 0) + 1,
            }
        except Exception as e:
            logger.error(f"[RESEARCHER] Error: {e}")
            return {
                **state,
                "research_cache": cache,  # Preserve cache even on error
                "research_progress": progress,
                "error_log": state.get("error_log", []) + [f"Researcher: {str(e)[:200]}"],
                "current_node": "error",
                "iteration_count": state.get("iteration_count", 0) + 1,  # Always increment
            }
