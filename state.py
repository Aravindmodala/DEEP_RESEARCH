"""State schema for the Deep Research Agent graph.

This module defines the shared state that flows between all nodes
in the LangGraph workflow, including cached intermediate results
for efficient resumption and avoiding duplicate API calls.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchCache(TypedDict, total=False):
    """Cached intermediate research results to avoid duplicate API calls."""
    
    # Target restaurant data (from Google Maps)
    target_restaurant: dict[str, Any] | None
    
    # Competitor data (from Google Maps)
    competitors: list[dict[str, Any]]
    
    # Extracted menus: {restaurant_name: menu_content}
    menus: dict[str, str]
    
    # Reviews: {place_id: [reviews]}
    reviews: dict[str, list[dict[str, Any]]]
    
    # Search results: {query_hash: results}
    search_results: dict[str, list[dict[str, Any]]]
    
    # Market data from web searches
    market_data: list[dict[str, Any]]


class ResearchProgress(TypedDict, total=False):
    """Tracks which research steps have been completed."""
    
    target_found: bool
    competitors_found: bool
    target_menu_extracted: bool
    competitor_menus_extracted: bool
    reviews_collected: bool
    market_signals_gathered: bool
    synthesis_complete: bool


class AgentGraphState(TypedDict, total=False):
    """State schema for the agent graph.
    
    This TypedDict defines the shared state that flows between
    all nodes in the LangGraph workflow.
    """

    # =========================================================================
    # INPUT
    # =========================================================================
    user_query: str

    # =========================================================================
    # NODE OUTPUTS (final structured outputs from each agent)
    # =========================================================================
    planner_output: dict[str, Any] | None
    researcher_output: dict[str, Any] | None
    critic_output: dict[str, Any] | None
    report_output: dict[str, Any] | None

    # =========================================================================
    # CACHED INTERMEDIATE RESULTS (avoid duplicate API calls)
    # =========================================================================
    research_cache: ResearchCache | None
    research_progress: ResearchProgress | None

    # =========================================================================
    # CONTROL FLOW
    # =========================================================================
    iteration_count: int
    max_iterations: int
    current_node: str
    error_log: list[str]

    # =========================================================================
    # METADATA
    # =========================================================================
    session_id: str
    started_at: str
    completed_at: str | None


def create_initial_state(
    user_query: str,
    session_id: str,
    max_iterations: int = 2,
) -> AgentGraphState:
    """
    Create a properly initialized state for the research workflow.
    
    Args:
        user_query: The user's research query
        session_id: Unique session identifier
        max_iterations: Maximum research iterations before forcing completion
        
    Returns:
        Initialized AgentGraphState
    """
    from datetime import datetime
    
    return AgentGraphState(
        user_query=user_query,
        planner_output=None,
        researcher_output=None,
        critic_output=None,
        report_output=None,
        research_cache=ResearchCache(
            target_restaurant=None,
            competitors=[],
            menus={},
            reviews={},
            search_results={},
            market_data=[],
        ),
        research_progress=ResearchProgress(
            target_found=False,
            competitors_found=False,
            target_menu_extracted=False,
            competitor_menus_extracted=False,
            reviews_collected=False,
            market_signals_gathered=False,
            synthesis_complete=False,
        ),
        iteration_count=0,
        max_iterations=max_iterations,
        current_node="planner",
        error_log=[],
        session_id=session_id,
        started_at=datetime.now().isoformat(),
        completed_at=None,
    )


def get_cache(state: AgentGraphState) -> ResearchCache:
    """Get or create the research cache from state."""
    if state.get("research_cache") is None:
        return ResearchCache(
            target_restaurant=None,
            competitors=[],
            menus={},
            reviews={},
            search_results={},
            market_data=[],
        )
    return state["research_cache"]


def get_progress(state: AgentGraphState) -> ResearchProgress:
    """Get or create the research progress from state."""
    if state.get("research_progress") is None:
        return ResearchProgress(
            target_found=False,
            competitors_found=False,
            target_menu_extracted=False,
            competitor_menus_extracted=False,
            reviews_collected=False,
            market_signals_gathered=False,
            synthesis_complete=False,
        )
    return state["research_progress"]
