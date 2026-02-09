"""LangGraph workflow builder for the Deep Research Agent.

This module provides the graph building logic for the multi-agent workflow.
The workflow follows: PLANNER → RESEARCHER → ANALYST → CRITIC → REPORT
with conditional routing for REJECT cycles.
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END, StateGraph

from state import AgentGraphState


def build_research_graph(
    planner_node: Callable[[AgentGraphState], AgentGraphState],
    researcher_node: Callable[[AgentGraphState], AgentGraphState],
    analyst_node: Callable[[AgentGraphState], AgentGraphState],
    critic_node: Callable[[AgentGraphState], AgentGraphState],
    report_node: Callable[[AgentGraphState], AgentGraphState],
    error_node: Callable[[AgentGraphState], AgentGraphState],
    route_critic_decision: Callable[[AgentGraphState], str],
) -> StateGraph:
    """
    Build the LangGraph workflow for deep research.

    Args:
        planner_node: Function to execute the Planner agent
        researcher_node: Function to execute the Researcher agent
        analyst_node: Function to execute the Analyst agent
        critic_node: Function to execute the Critic agent
        report_node: Function to execute the Report agent
        error_node: Function to handle errors
        route_critic_decision: Function to route based on critic decision

    Returns:
        Configured StateGraph ready to be compiled

    Flow:
        PLANNER → RESEARCHER → ANALYST → CRITIC
                                           ↓
                              ┌─────────────┴─────────────┐
                              ↓                           ↓
                           ACCEPT                      REJECT
                              ↓                           ↓
                           REPORT              RESEARCHER (retry)
                              ↓                           ↓
                             END                  (max iterations)
                                                          ↓
                                                       REPORT
    """
    # Create state graph
    workflow = StateGraph(AgentGraphState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("report", report_node)
    workflow.add_node("error", error_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Add edges: planner → researcher → analyst → critic
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "critic")

    # Conditional edge from critic
    workflow.add_conditional_edges(
        "critic",
        route_critic_decision,
        {
            "researcher": "researcher",
            "report": "report",
            "error": "error",
        },
    )

    # Report and error go to END
    workflow.add_edge("report", END)
    workflow.add_edge("error", END)

    return workflow
