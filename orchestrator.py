"""
LangGraph Orchestrator for the Deep Research Agent System.

This module orchestrates the multi-agent workflow using LangGraph.
The workflow follows: PLANNER → RESEARCHER → ANALYST → CRITIC → REPORT
with conditional routing for REJECT cycles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.report import ReportAgent
from agents.researcher import ResearcherAgent
from config import AgentConfig, get_config
from graph import build_research_graph
from state import AgentGraphState


class DeepResearchOrchestrator:
    """
    Orchestrates the multi-agent research workflow.

    Flow:
    1. PLANNER: Parse user query → research plan
    2. RESEARCHER: Execute phased research with ReAct loops
    3. ANALYST: Multi-pass LLM analysis of raw data
    4. CRITIC: Evaluate quality
       - If ACCEPT → REPORT
       - If REJECT → back to RESEARCHER (up to max_iterations)
    5. REPORT: Generate section-by-section final report
    """

    def __init__(self, config: AgentConfig | None = None, skip_confirmation: bool = False):
        self.config = config or get_config()
        self.skip_confirmation = skip_confirmation

        # Initialize agents
        self.planner = PlannerAgent(self.config)
        self.researcher = ResearcherAgent(self.config)
        self.analyst = AnalystAgent(self.config)
        self.critic = CriticAgent(self.config)
        self.reporter = ReportAgent(self.config)

        # Build the graph using the graph builder
        self.graph = build_research_graph(
            planner_node=self._planner_node,
            researcher_node=self._researcher_node,
            analyst_node=self._analyst_node,
            critic_node=self._critic_node,
            report_node=self._report_node,
            error_node=self._error_node,
            route_critic_decision=self._route_critic_decision,
        )
        self.app = self.graph.compile()

    def _planner_node(self, state: AgentGraphState) -> AgentGraphState:
        """Execute the Planner agent with human confirmation."""
        logger.info("=" * 60)
        logger.info("NODE: PLANNER")
        logger.info("=" * 60)

        user_query = state.get("user_query", "")
        if not user_query:
            return {
                **state,
                "error_log": state.get("error_log", []) + ["No user query provided"],
                "current_node": "error",
            }

        try:
            if self.skip_confirmation:
                output = self.planner.plan(user_query)
            else:
                output = self.planner.plan_with_confirmation(user_query)
            return {
                **state,
                "planner_output": output.model_dump(),
                "current_node": "researcher",
            }
        except KeyboardInterrupt:
            logger.info("Research cancelled by user")
            return {
                **state,
                "error_log": state.get("error_log", []) + ["Research cancelled by user"],
                "current_node": "error",
            }
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Planner error: {str(e)}"],
                "current_node": "error",
            }

    def _researcher_node(self, state: AgentGraphState) -> AgentGraphState:
        """Execute the Researcher agent (phase-gated)."""
        logger.info("=" * 60)
        logger.info(f"NODE: RESEARCHER (Iteration {state.get('iteration_count', 0) + 1})")
        logger.info("=" * 60)

        planner_output = state.get("planner_output")
        if not planner_output:
            return {
                **state,
                "error_log": state.get("error_log", []) + ["No planner output"],
                "current_node": "error",
            }

        try:
            # If this is a retry, include critic feedback
            critic_output = state.get("critic_output")
            if critic_output and isinstance(critic_output, dict):
                feedback = critic_output.get("required_fixes", [])
                if feedback:
                    logger.info(f"[RESEARCHER] Addressing critic feedback: {feedback}")

            output = self.researcher.research(planner_output)

            return {
                **state,
                "researcher_output": output.model_dump(),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "current_node": "analyst",
            }
        except Exception as e:
            logger.error(f"Researcher error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Researcher error: {str(e)}"],
                "current_node": "error",
            }

    def _analyst_node(self, state: AgentGraphState) -> AgentGraphState:
        """Execute the Analyst agent (multi-pass analysis)."""
        logger.info("=" * 60)
        logger.info("NODE: ANALYST")
        logger.info("=" * 60)

        researcher_output = state.get("researcher_output")
        if not researcher_output:
            return {
                **state,
                "error_log": state.get("error_log", []) + ["No researcher output for analyst"],
                "current_node": "error",
            }

        try:
            output = self.analyst.analyze(researcher_output)

            return {
                **state,
                "analyst_output": output.model_dump(),
                "current_node": "critic",
            }
        except Exception as e:
            logger.error(f"Analyst error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Analyst error: {str(e)}"],
                "current_node": "error",
            }

    def _critic_node(self, state: AgentGraphState) -> AgentGraphState:
        """Execute the Critic agent."""
        logger.info("=" * 60)
        logger.info("NODE: CRITIC")
        logger.info("=" * 60)

        researcher_output = state.get("researcher_output")
        analyst_output = state.get("analyst_output")
        if not researcher_output:
            return {
                **state,
                "error_log": state.get("error_log", []) + ["No researcher output"],
                "current_node": "error",
            }

        try:
            user_query = state.get("user_query", "")
            planner_output = state.get("planner_output") or {}

            output = self.critic.evaluate(
                user_query, planner_output, researcher_output, analyst_output
            )

            return {
                **state,
                "critic_output": output.model_dump(),
                "current_node": "report" if output.decision == "ACCEPT" else "researcher",
            }
        except Exception as e:
            logger.error(f"Critic error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Critic error: {str(e)}"],
                "current_node": "error",
            }

    def _report_node(self, state: AgentGraphState) -> AgentGraphState:
        """Execute the Report agent (section-by-section)."""
        logger.info("=" * 60)
        logger.info("NODE: REPORT")
        logger.info("=" * 60)

        try:
            report = self.reporter.generate(
                plan=state["planner_output"],
                research=state["researcher_output"],
                critique=state["critic_output"],
                analyst=state.get("analyst_output"),
            )

            return {
                **state,
                "report_output": report,
                "current_node": "complete",
                "completed_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Report error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Report error: {str(e)}"],
                "current_node": "error",
            }

    def _error_node(self, state: AgentGraphState) -> AgentGraphState:
        """Handle errors in the workflow."""
        logger.error("=" * 60)
        logger.error("NODE: ERROR")
        logger.error(f"Errors: {state.get('error_log', [])}")
        logger.error("=" * 60)
        return {
            **state,
            "current_node": "error",
            "completed_at": datetime.now().isoformat(),
        }

    def _route_critic_decision(self, state: AgentGraphState) -> str:
        """Route based on critic decision and iteration count."""
        critic_output = state.get("critic_output")

        if critic_output is None:
            logger.error("[ROUTER] No critic output - routing to ERROR")
            return "error"

        decision = critic_output.get("decision", "REJECT")
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", self.config.max_research_iterations)

        if decision == "ACCEPT":
            logger.info("[ROUTER] Critic ACCEPTED - routing to REPORT")
            return "report"
        elif iteration >= max_iter:
            logger.warning(f"[ROUTER] Max iterations ({max_iter}) reached - forcing to REPORT")
            return "report"
        else:
            logger.info(f"[ROUTER] Critic REJECTED - routing back to RESEARCHER (iteration {iteration + 1})")
            return "researcher"

    def run(
        self,
        user_query: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete research workflow.

        Args:
            user_query: Natural language research request
            session_id: Optional session identifier

        Returns:
            Final state with all agent outputs
        """
        logger.info("=" * 80)
        logger.info("DEEP RESEARCH AGENT v2 - STARTING")
        logger.info(f"Query: {user_query}")
        logger.info("=" * 80)

        initial_state: AgentGraphState = {
            "user_query": user_query,
            "planner_output": None,
            "researcher_output": None,
            "analyst_output": None,
            "critic_output": None,
            "report_output": None,
            "iteration_count": 0,
            "max_iterations": self.config.max_research_iterations,
            "current_node": "planner",
            "error_log": [],
            "session_id": session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
        }

        final_state = self.app.invoke(initial_state)

        logger.info("=" * 80)
        logger.info("DEEP RESEARCH AGENT v2 - COMPLETE")
        logger.info(f"Final node: {final_state.get('current_node')}")
        logger.info(f"Iterations: {final_state.get('iteration_count')}")
        logger.info("=" * 80)

        return final_state

    def run_streaming(
        self,
        user_query: str,
        session_id: str | None = None,
    ):
        """
        Run the workflow with streaming updates.

        Yields state updates after each node execution.
        """
        initial_state: AgentGraphState = {
            "user_query": user_query,
            "planner_output": None,
            "researcher_output": None,
            "analyst_output": None,
            "critic_output": None,
            "report_output": None,
            "iteration_count": 0,
            "max_iterations": self.config.max_research_iterations,
            "current_node": "planner",
            "error_log": [],
            "session_id": session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
        }

        for event in self.app.stream(initial_state):
            yield event


def format_report_as_markdown(report: str) -> str:
    """
    Format a report as a readable Markdown document.

    Args:
        report: Markdown report text from Report agent

    Returns:
        Formatted Markdown string (returns as-is since it's already markdown)
    """
    return report
