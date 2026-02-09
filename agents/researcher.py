"""RESEARCHER Agent - Phase-Gated ReAct-style research execution."""

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
    ResearcherOutput,
    RestaurantInfo,
    SourceReference,
)
from prompts.researcher import RESEARCHER_SYSTEM_PROMPT, PHASE_CONFIG
from tools.google_maps import create_google_maps_tools
from tools.tavily_search import create_tavily_tools
from tools.tavily_extract import create_tavily_extract_tools
from tools.web_scraper import create_scraper_tools
from utils import create_llm


class ResearcherAgent:
    """
    NODE 2 — RESEARCHER AGENT (PHASE-GATED)

    Executes research plan using a phase-gated ReAct architecture.
    Each of 6 phases has its own prompt, tool whitelist, iteration budget,
    and completion criteria.

    Returns validated ResearcherOutput with raw data (no analysis).
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config)
        self.tools = self._initialize_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.max_total_iterations = config.max_total_react_iterations

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
        Execute research based on the planner's output using 6 phases.

        Args:
            plan: Research plan dict from Planner

        Returns:
            Validated ResearcherOutput Pydantic model (raw data only)
        """
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        cuisine = plan.get("cuisine_type", "Unknown")

        logger.info(f"[RESEARCHER] Starting phase-gated research for {target} in {location}")

        # Data Store
        collected_data = {
            "target": None,
            "competitors": [],
            "reviews": {},
            "supplementary_reviews": {},
            "menus": [],
            "business_intel": {},
            "market_data": [],
            "sources": [],
        }

        # Track total iterations across all phases
        total_iterations = 0

        # Run each phase
        for phase_idx, phase_config in enumerate(PHASE_CONFIG):
            phase_name = phase_config["name"]
            phase_prompt = phase_config["prompt"]
            phase_max_iters = phase_config["max_iters"]
            phase_tool_names = phase_config["tools"]

            # Hard cap check
            if total_iterations >= self.max_total_iterations:
                logger.warning(f"[RESEARCHER] Hit total iteration cap ({self.max_total_iterations}). Stopping.")
                break

            logger.info(f"[RESEARCHER] === PHASE {phase_idx + 1}: {phase_name} ===")

            # Get phase-specific tools
            phase_tools = [t for t in self.tools if t.name in phase_tool_names]
            if not phase_tools:
                logger.warning(f"[RESEARCHER] No tools available for phase {phase_name}. Skipping.")
                continue

            # Bind phase tools to LLM
            llm_with_tools = self.llm.bind_tools(phase_tools)

            # Build context message with current data summary
            context_msg = self._build_phase_context(plan, collected_data, phase_name)

            # Phase message history
            messages = [
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                HumanMessage(content=context_msg),
                HumanMessage(content=phase_prompt),
            ]

            # Phase ReAct loop
            phase_iters = 0
            remaining_budget = min(phase_max_iters, self.max_total_iterations - total_iterations)

            while phase_iters < remaining_budget:
                phase_iters += 1
                total_iterations += 1

                logger.info(f"[RESEARCHER] Phase '{phase_name}' iteration {phase_iters}/{remaining_budget} (total: {total_iterations})")

                # THOUGHT + ACTION
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if response.content:
                    logger.info(f"[RESEARCHER] THOUGHT: {str(response.content)[:300]}...")

                # Check for tool calls
                if not response.tool_calls:
                    logger.info(f"[RESEARCHER] Phase '{phase_name}' complete (no more tool calls).")
                    break

                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call.get("id", f"call_{total_iterations}")

                    logger.info(f"[RESEARCHER] CALLING: {tool_name}")

                    try:
                        tool = self.tool_map.get(tool_name)
                        if tool:
                            result = tool.invoke(tool_args)

                            # Route data to correct bucket based on current phase
                            self._collect_data(collected_data, tool_name, tool_args, result, phase_name)

                            # Audit trail
                            collected_data["sources"].append({
                                "source_type": self._get_source_type(tool_name),
                                "title": f"Tool: {tool_name} (Phase: {phase_name})",
                                "url": tool_args.get("url") or tool_args.get("place_id") or "API",
                                "data_summary": str(result)[:200] + "...",
                                "accessed_at": datetime.now().isoformat(),
                            })
                        else:
                            result = f"Error: Tool '{tool_name}' not found."
                            logger.error(f"[RESEARCHER] {result}")
                    except Exception as e:
                        result = f"Error executing tool '{tool_name}': {str(e)}"
                        logger.error(f"[RESEARCHER] {result}")

                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

                # Check phase completion criteria
                if self._check_phase_complete(collected_data, phase_name):
                    logger.info(f"[RESEARCHER] Phase '{phase_name}' objectives met.")
                    break

            logger.info(f"[RESEARCHER] Phase '{phase_name}' finished. Iterations used: {phase_iters}")

        # Build raw data output (no analysis)
        output = self._build_output(collected_data, plan)

        logger.info(f"[RESEARCHER] Research Complete. Total iterations: {total_iterations}. "
                     f"Competitors: {len(output.competitors)}")
        return output

    def _build_phase_context(self, plan: dict[str, Any], collected_data: dict, phase_name: str) -> str:
        """Build a context message with the plan and current data summary."""
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        cuisine = plan.get("cuisine_type", "Unknown")

        # Summarize what we've collected so far
        target_info = collected_data.get("target")
        target_summary = "Not yet found"
        if target_info:
            target_summary = f"{target_info.get('name', 'Unknown')} (place_id: {target_info.get('place_id', 'N/A')}, rating: {target_info.get('rating', 'N/A')})"

        comp_count = len(collected_data.get("competitors", []))
        review_count = len(collected_data.get("reviews", {}))
        menu_count = len(collected_data.get("menus", []))
        supp_review_count = len(collected_data.get("supplementary_reviews", {}))

        # Include competitor details if available
        comp_details = ""
        for c in collected_data.get("competitors", [])[:5]:
            comp_details += f"\n  - {c.get('name', 'Unknown')} (place_id: {c.get('place_id', 'N/A')}, rating: {c.get('rating', 'N/A')}, distance: {c.get('distance_miles', 'N/A')} mi)"

        # Include target website if known
        target_website = ""
        if target_info and target_info.get("website"):
            target_website = f"\n- **Target Website:** {target_info.get('website')}"

        return f"""## RESEARCH MISSION

**TARGET**: {target}
**LOCATION**: {location}
**CUISINE**: {cuisine}

## DATA COLLECTED SO FAR

- **Target**: {target_summary}{target_website}
- **Competitors Found**: {comp_count}{comp_details}
- **Reviews Collected**: {review_count} restaurants (Google), {supp_review_count} supplementary sources
- **Menus Collected**: {menu_count}
- **Business Intel**: {"Yes" if collected_data.get("business_intel") else "Not yet"}
- **Market Data**: {len(collected_data.get("market_data", []))} data points

Now executing: **{phase_name}**"""

    def _get_source_type(self, tool_name: str) -> str:
        if "google" in tool_name or "restaurant" in tool_name:
            return "google_maps"
        if "tavily" in tool_name or "search" in tool_name:
            return "search"
        if "extract" in tool_name or "scrape" in tool_name:
            return "website"
        return "system"

    def _collect_data(
        self,
        collected_data: dict,
        tool_name: str,
        tool_args: dict,
        result: Any,
        current_phase: str,
    ) -> None:
        """Intelligent data collection router with phase awareness."""
        try:
            # Parse result if it's a JSON string
            parsed = result
            if isinstance(result, str):
                if result.strip().startswith(("{", "[")):
                    try:
                        parsed = json.loads(result)
                    except Exception:
                        pass

            tool_lower = tool_name.lower()

            # --- COMPETITOR / TARGET IDENTIFICATION ---
            if "find_competitor" in tool_lower:
                if isinstance(parsed, dict) and "competitors" in parsed:
                    found_count = len(parsed["competitors"])
                    logger.info(f"[RESEARCHER] Collected {found_count} competitors from tool.")
                    collected_data["competitors"].extend(parsed["competitors"])

            elif "find_restaurant" in tool_lower:
                if isinstance(parsed, dict) and parsed.get("name"):
                    collected_data["target"] = parsed

            # --- DETAILS (can enrich target or competitors) ---
            elif "get_restaurant_details" in tool_lower or "get_restaurant_website" in tool_lower:
                # Store website and other details on target or competitor
                if isinstance(parsed, dict):
                    place_id = tool_args.get("place_id")
                    website = parsed.get("website") if isinstance(parsed, dict) else None
                    # Try to extract website from string result
                    if not website and isinstance(result, str) and "Website for" in result:
                        parts = result.split(": ", 1)
                        if len(parts) > 1:
                            website = parts[1].strip()

                    if place_id and website:
                        # Update target
                        if collected_data.get("target") and collected_data["target"].get("place_id") == place_id:
                            collected_data["target"]["website"] = website
                        # Update competitors
                        for c in collected_data.get("competitors", []):
                            if c.get("place_id") == place_id:
                                c["website"] = website

            # --- REVIEWS ---
            elif "review" in tool_lower:
                if current_phase == "Review Deep Dive":
                    if "search_reviews" in tool_lower:
                        # Supplementary reviews from Yelp/TripAdvisor
                        restaurant_name = tool_args.get("restaurant_name", "Unknown")
                        collected_data["supplementary_reviews"][restaurant_name] = parsed
                    elif "get_restaurant_reviews" in tool_lower:
                        # Google reviews
                        place_id = tool_args.get("place_id") or "unknown"
                        restaurant_name = self._find_restaurant_name(collected_data, place_id)
                        collected_data["reviews"][restaurant_name] = {
                            "place_id": place_id,
                            "reviews": parsed if isinstance(parsed, list) else [],
                        }
                    else:
                        # Generic review data
                        collected_data["supplementary_reviews"][tool_name] = parsed
                else:
                    # Reviews found in other phases go to supplementary
                    collected_data["supplementary_reviews"][tool_name] = parsed

            # --- MENUS ---
            elif "menu" in tool_lower:
                # Ensure we only store dicts in menus
                if isinstance(parsed, dict):
                    if "restaurant_name" not in parsed:
                        parsed["restaurant_name"] = tool_args.get("restaurant_name", "Unknown")
                    if "source_url" not in parsed:
                        parsed["source_url"] = tool_args.get("url", "Unknown")
                    collected_data["menus"].append(parsed)
                elif isinstance(parsed, list):
                    # Tool returned a list of results - wrap as a menu entry
                    restaurant_name = tool_args.get("restaurant_name") or tool_args.get("query", "Unknown")
                    collected_data["menus"].append({
                        "restaurant_name": restaurant_name,
                        "source_url": tool_args.get("url", "search"),
                        "search_results": parsed,
                    })
                elif isinstance(parsed, str):
                    # Try to extract useful data from string result
                    restaurant_name = tool_args.get("restaurant_name") or tool_args.get("query", "Unknown")
                    collected_data["menus"].append({
                        "restaurant_name": restaurant_name,
                        "source_url": tool_args.get("url", "search"),
                        "raw_text": parsed[:5000],
                    })

            # --- BUSINESS INTELLIGENCE ---
            elif current_phase == "Business Intelligence":
                collected_data["business_intel"][tool_name + "_" + str(len(collected_data["business_intel"]))] = {
                    "tool": tool_name,
                    "query": tool_args.get("query") or tool_args.get("url"),
                    "content": parsed,
                }

            # --- MARKET DATA ---
            elif current_phase == "Market Context":
                collected_data["market_data"].append({
                    "tool": tool_name,
                    "query": tool_args.get("query") or tool_args.get("url") or tool_args.get("location"),
                    "content": parsed,
                })

            # --- GENERAL (from search/extract in non-specific phases) ---
            elif "search" in tool_lower or "extract" in tool_lower:
                collected_data["market_data"].append({
                    "tool": tool_name,
                    "query": tool_args.get("query") or tool_args.get("url"),
                    "content": parsed,
                })

        except Exception as e:
            logger.warning(f"[RESEARCHER] Data collection warning: {e}")

    def _find_restaurant_name(self, collected_data: dict, place_id: str) -> str:
        """Helper to resolve place_id to name."""
        if collected_data.get("target") and collected_data["target"].get("place_id") == place_id:
            return collected_data["target"].get("name", "Target")

        for comp in collected_data.get("competitors", []):
            if comp.get("place_id") == place_id:
                return comp.get("name", "Unknown Competitor")

        return f"Unknown ({place_id[:8]})"

    def _check_phase_complete(self, collected_data: dict, phase_name: str) -> bool:
        """Check if phase objectives are met."""
        if phase_name == "Target Discovery":
            target = collected_data.get("target")
            return bool(target and target.get("name") and target.get("place_id") and target.get("rating") is not None)

        elif phase_name == "Competitor Discovery":
            competitors = collected_data.get("competitors", [])
            valid = [c for c in competitors if c.get("name") and c.get("place_id") and c.get("rating") is not None]
            return len(valid) >= 3

        elif phase_name == "Review Deep Dive":
            reviews = collected_data.get("reviews", {})
            return len(reviews) >= 2  # At least target + 1 competitor

        elif phase_name == "Menu Intelligence":
            menus = collected_data.get("menus", [])
            return len(menus) >= 2  # At least target + 1 competitor

        elif phase_name == "Business Intelligence":
            return bool(collected_data.get("business_intel"))

        elif phase_name == "Market Context":
            return len(collected_data.get("market_data", [])) >= 1

        return False

    # --------------------------------------------------------------------------
    # OUTPUT CONSTRUCTION (Raw Data Only)
    # --------------------------------------------------------------------------

    def _build_output(self, collected_data: dict, plan: dict) -> ResearcherOutput:
        """Construct the final raw-data Pydantic model. No analysis."""

        # 1. Target Info
        target_raw = collected_data.get("target") or {}
        target_info = RestaurantInfo(
            name=target_raw.get("name", plan.get("target_restaurant", "Unknown")),
            address=target_raw.get("address", ""),
            place_id=target_raw.get("place_id"),
            rating=target_raw.get("rating"),
            review_count=target_raw.get("user_ratings_total"),
            price_level=str(target_raw.get("price_level", "")),
            cuisine_type=plan.get("cuisine_type", "Restaurant"),
            website=target_raw.get("website"),
            editorial_summary=target_raw.get("editorial_summary", ""),
            types=target_raw.get("types", []),
        )

        # 2. Competitors
        competitors = []
        for c in collected_data.get("competitors", []):
            competitors.append(CompetitorInfo(
                name=c.get("name", "Unknown"),
                address=c.get("address", ""),
                rating=c.get("rating"),
                review_count=c.get("user_ratings_total") or c.get("review_count"),
                price_level=str(c.get("price_level", "")),
                place_id=c.get("place_id"),
                distance_miles=c.get("distance_miles", 0.0),
                website=c.get("website"),
                cuisine_type=c.get("cuisine_type", "restaurant"),
            ))

        # 3. Sources
        sources = [SourceReference(**s) for s in collected_data.get("sources", [])]

        return ResearcherOutput(
            target=target_info,
            competitors=competitors,
            raw_reviews=collected_data.get("reviews", {}),
            supplementary_reviews=collected_data.get("supplementary_reviews", {}),
            raw_menus=collected_data.get("menus", []),
            business_intel=collected_data.get("business_intel", {}),
            market_data=collected_data.get("market_data", []),
            raw_sources=sources,
            research_notes=[
                f"Phases completed: {len(PHASE_CONFIG)}",
                f"Total sources: {len(sources)}",
            ],
        )
