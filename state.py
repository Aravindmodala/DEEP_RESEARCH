"""State schema for the Deep Research Agent graph."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentGraphState(TypedDict, total=False):
    """State schema for the agent graph.

    This TypedDict defines the shared state that flows between
    all nodes in the LangGraph workflow.
    """

    # Input
    user_query: str

    # Node outputs (as dicts for serialization)
    planner_output: dict[str, Any] | None
    researcher_output: dict[str, Any] | None
    analyst_output: dict[str, Any] | None
    critic_output: dict[str, Any] | None
    report_output: dict[str, Any] | None

    # Control flow
    iteration_count: int
    max_iterations: int
    current_node: str
    error_log: list[str]

    # Metadata
    session_id: str
    started_at: str
    completed_at: str | None
