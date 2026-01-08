"""
LangGraph Orchestrator for the Deep Research Agent System.

This module orchestrates the multi-agent workflow using LangGraph.
The workflow follows: PLANNER → RESEARCHER → CRITIC → REPORT
with conditional routing for REJECT cycles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.report import ReportAgent
from agents.researcher import ResearcherAgent
from config import AgentConfig, get_config
from graph import build_research_graph
from models.schemas import (
    CriticOutput,
    PlannerOutput,
    ReportOutput,
    ResearcherOutput,
)
from state import AgentGraphState


class DeepResearchOrchestrator:
    """
    Orchestrates the multi-agent research workflow.
    
    Flow:
    1. PLANNER: Parse user query → research plan
    2. RESEARCHER: Execute research with ReAct loop
    3. CRITIC: Evaluate quality
       - If ACCEPT → REPORT
       - If REJECT → back to RESEARCHER (up to max_iterations)
    4. REPORT: Generate final report
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or get_config()
        
        # Initialize agents
        self.planner = PlannerAgent(self.config)
        self.researcher = ResearcherAgent(self.config)
        self.critic = CriticAgent(self.config)
        self.reporter = ReportAgent(self.config)
        
        # Build the graph using the graph builder
        self.graph = build_research_graph(
            planner_node=self._planner_node,
            researcher_node=self._researcher_node,
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
            # Use plan_with_confirmation for human-in-the-loop
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
        """Execute the Researcher agent."""
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
            plan = PlannerOutput(**planner_output)
            
            # If this is a retry, include critic feedback
            critic_output = state.get("critic_output")
            if critic_output and isinstance(critic_output, dict):
                feedback = critic_output.get("required_fixes", [])
                if feedback:
                    logger.info(f"[RESEARCHER] Addressing critic feedback: {feedback}")
            
            output = self.researcher.research(plan)
            
            return {
                **state,
                "researcher_output": output.model_dump(),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "current_node": "critic",
            }
        except Exception as e:
            logger.error(f"Researcher error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Researcher error: {str(e)}"],
                "current_node": "error",
            }

    def _critic_node(self, state: AgentGraphState) -> AgentGraphState:
        """Execute the Critic agent."""
        logger.info("=" * 60)
        logger.info("NODE: CRITIC")
        logger.info("=" * 60)

        researcher_output = state.get("researcher_output")
        if not researcher_output:
            return {
                **state,
                "error_log": state.get("error_log", []) + ["No researcher output"],
                "current_node": "error",
            }

        try:
            research = ResearcherOutput(**researcher_output)
            output = self.critic.evaluate(research)
            
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
        """Execute the Report agent."""
        logger.info("=" * 60)
        logger.info("NODE: REPORT")
        logger.info("=" * 60)

        try:
            plan = PlannerOutput(**state["planner_output"])
            research = ResearcherOutput(**state["researcher_output"])
            critique = CriticOutput(**state["critic_output"])

            report = self.reporter.generate(plan, research, critique)
            
            return {
                **state,
                "report_output": report.model_dump(),
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
        critic_output = state.get("critic_output", {})
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
        logger.info("DEEP RESEARCH AGENT - STARTING")
        logger.info(f"Query: {user_query}")
        logger.info("=" * 80)

        # Initialize state
        initial_state: AgentGraphState = {
            "user_query": user_query,
            "planner_output": None,
            "researcher_output": None,
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

        # Run the graph
        final_state = self.app.invoke(initial_state)

        logger.info("=" * 80)
        logger.info("DEEP RESEARCH AGENT - COMPLETE")
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


def format_report_as_markdown(report: ReportOutput | dict) -> str:
    """
    Format a ReportOutput as a readable Markdown document.
    
    Args:
        report: ReportOutput or dict representation
        
    Returns:
        Formatted Markdown string
    """
    if isinstance(report, dict):
        report = ReportOutput(**report)

    md = []

    # Title
    md.append(f"# {report.report_title}")
    md.append(f"\n**Generated:** {report.generated_at}")
    md.append(f"**Location:** {report.location}")
    md.append("")

    # Executive Summary
    md.append("## Executive Summary")
    md.append("")
    md.append(report.executive_summary.overview)
    md.append("")
    md.append("### Key Findings")
    for finding in report.executive_summary.key_findings:
        md.append(f"- {finding}")
    md.append("")
    md.append(f"**Lending Implications:** {report.executive_summary.lending_implications}")
    md.append("")
    md.append(f"**Expansion Implications:** {report.executive_summary.expansion_implications}")
    md.append("")

    # Competitive Landscape
    md.append("## Competitive Landscape")
    md.append("")
    md.append(report.competitive_landscape.summary)
    md.append("")
    md.append(f"- **Competitors:** {report.competitive_landscape.competitor_count}")
    md.append(f"- **Market Saturation:** {report.competitive_landscape.market_saturation_level}")
    md.append("")
    
    if report.competitive_landscape.top_competitors:
        md.append("### Top Competitors")
        md.append("")
        md.append("| Name | Rating | Distance | Price |")
        md.append("|------|--------|----------|-------|")
        for c in report.competitive_landscape.top_competitors[:5]:
            name = c.get("name", "N/A")
            rating = c.get("rating", "N/A")
            distance = c.get("distance", "N/A")
            price = c.get("price_level", "N/A")
            md.append(f"| {name} | {rating} | {distance} | {price} |")
        md.append("")

    md.append("### Competitive Advantages")
    for adv in report.competitive_landscape.competitive_advantages:
        md.append(f"- {adv}")
    md.append("")

    md.append("### Competitive Disadvantages")
    for dis in report.competitive_landscape.competitive_disadvantages:
        md.append(f"- {dis}")
    md.append("")

    # Menu & Pricing
    md.append("## Menu & Pricing Position")
    md.append("")
    md.append(report.menu_pricing.summary)
    md.append("")
    md.append(f"**Price Position:** {report.menu_pricing.price_position}")
    md.append("")
    md.append(report.menu_pricing.price_comparison_narrative)
    md.append("")
    md.append(f"**Menu Differentiation:** {report.menu_pricing.menu_differentiation}")
    md.append("")

    # Customer Sentiment
    md.append("## Customer Sentiment")
    md.append("")
    md.append(report.customer_sentiment.summary)
    md.append("")
    md.append(f"**Overall Sentiment:** {report.customer_sentiment.overall_sentiment_rating}")
    md.append("")
    
    md.append("### Strengths")
    for s in report.customer_sentiment.key_strengths:
        md.append(f"- {s}")
    md.append("")

    md.append("### Concerns")
    for c in report.customer_sentiment.key_concerns:
        md.append(f"- {c}")
    md.append("")

    md.append(f"**Reputation Risk:** {report.customer_sentiment.reputation_risk_assessment}")
    md.append("")

    # Banking Relevance
    md.append("## Commercial Banking Relevance")
    md.append("")
    
    md.append("### Revenue Stability Indicators")
    for ind in report.banking_relevance.revenue_stability_indicators:
        md.append(f"- {ind}")
    md.append("")

    md.append(f"**Expansion Viability:** {report.banking_relevance.expansion_viability_assessment}")
    md.append("")

    md.append("### Risk Considerations")
    for risk in report.banking_relevance.risk_considerations:
        md.append(f"- {risk}")
    md.append("")

    # Final Recommendation
    md.append("## Final Recommendation")
    md.append("")
    md.append(f"**Outlook:** {report.final_recommendation.outlook.upper()}")
    md.append(f"**Confidence:** {report.final_recommendation.confidence_level.upper()}")
    md.append("")
    md.append(f"### Recommendation")
    md.append(report.final_recommendation.primary_recommendation)
    md.append("")

    md.append("### Supporting Rationale")
    for r in report.final_recommendation.supporting_rationale:
        md.append(f"- {r}")
    md.append("")

    md.append("### Risk Mitigations")
    for m in report.final_recommendation.risk_mitigations:
        md.append(f"- {m}")
    md.append("")

    md.append("### Next Steps")
    for s in report.final_recommendation.next_steps:
        md.append(f"1. {s}")
    md.append("")

    # Disclaimers
    md.append("---")
    md.append("")
    md.append("## Disclaimers")
    for d in report.disclaimers:
        md.append(f"- {d}")
    md.append("")

    return "\n".join(md)
