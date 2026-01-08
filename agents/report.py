"""REPORT Agent - Generates client-ready market research reports."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import AgentConfig
from models.schemas import (
    CriticOutput,
    PlannerOutput,
    ReportOutput,
    ResearcherOutput,
)
from prompts import REPORT_SYSTEM_PROMPT
from utils import create_llm


class ReportAgent:
    """
    NODE 4 — REPORT AGENT
    
    Responsibility: Generate a client-ready Market Research Report
    for Commercial Banking using LLM with structured output.
    
    Uses `with_structured_output()` to let the LLM generate the full
    report directly, eliminating manual fallback logic.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        # Use structured output to enforce ReportOutput schema
        base_llm = create_llm(config, temperature=0.3)
        self.llm = base_llm.with_structured_output(ReportOutput)

    def generate(
        self,
        plan: PlannerOutput,
        research: ResearcherOutput,
        critique: CriticOutput,
    ) -> ReportOutput:
        """
        Generate a comprehensive market research report.
        
        The LLM receives all research data and generates a structured
        ReportOutput directly, leveraging its natural language capabilities.
        
        Args:
            plan: Original research plan from Planner
            research: Research findings from Researcher
            critique: Quality assessment from Critic
            
        Returns:
            ReportOutput with full structured report
        """
        logger.info(
            f"[REPORT] Generating report for {plan.target_restaurant} in {plan.location}"
        )

        # Prepare comprehensive context for LLM
        context = self._prepare_context(plan, research, critique)

        messages = [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        # LLM generates structured output directly
        report: ReportOutput = self.llm.invoke(messages)

        # Override metadata fields with actual values
        report.report_title = f"Market Research Report: {plan.target_restaurant}"
        report.generated_at = datetime.now().isoformat()
        report.target_restaurant = plan.target_restaurant
        report.location = plan.location
        report.appendix_sources = research.raw_sources

        logger.info("[REPORT] Report generation complete")
        return report

    def _prepare_context(
        self,
        plan: PlannerOutput,
        research: ResearcherOutput,
        critique: CriticOutput,
    ) -> str:
        """
        Prepare comprehensive context for report generation.
        
        Formats all research data in a clear structure for the LLM to synthesize.
        """
        # Format competitors summary
        competitor_summary = [
            {
                "name": c.name,
                "distance_miles": c.distance_miles,
                "rating": c.rating,
                "review_count": c.review_count,
                "price_level": c.price_level,
                "cuisine_type": c.cuisine_type,
            }
            for c in research.competitors[:10]
        ]

        # Format sentiment data
        sentiment_data = None
        if research.sentiment_analysis:
            sentiment_data = {
                "overall_sentiment": research.sentiment_analysis.target_overall_sentiment,
                "positive_drivers": research.sentiment_analysis.positive_drivers,
                "common_complaints": research.sentiment_analysis.common_complaints,
                "review_trend": research.sentiment_analysis.review_volume_trend,
                "sample_reviews": research.sentiment_analysis.sample_reviews[:3],
            }

        # Format market signals
        market_data = None
        if research.market_signals:
            market_data = {
                "competitor_density": research.market_signals.competitor_density,
                "market_saturation": research.market_signals.market_saturation,
                "avg_competitor_rating": research.market_signals.avg_competitor_rating,
                "foot_traffic_indicators": research.market_signals.foot_traffic_indicators,
                "growth_indicators": research.market_signals.growth_indicators,
                "risk_indicators": research.market_signals.risk_indicators,
                "local_economic_context": research.market_signals.local_economic_context,
            }

        # Format pricing data
        pricing_data = None
        if research.pricing_analysis:
            pricing_data = {
                "target_avg_price": research.pricing_analysis.target_avg_price,
                "market_avg_price": research.pricing_analysis.market_avg_price,
                "price_position": research.pricing_analysis.price_position,
                "price_percentile": research.pricing_analysis.price_percentile,
            }

        # Format menu data
        menu_data = None
        if research.menu_comparison:
            menu_data = {
                "has_target_menu": research.menu_comparison.target_menu is not None,
                "competitor_menus_count": len(research.menu_comparison.competitor_menus),
                "unique_offerings": research.menu_comparison.unique_offerings,
                "menu_breadth_comparison": research.menu_comparison.menu_breadth_comparison,
            }

        return f"""## RESEARCH DATA FOR REPORT GENERATION

### Target Restaurant
- **Name:** {plan.target_restaurant}
- **Location:** {plan.location}
- **Cuisine Type:** {plan.cuisine_type}
- **Research Intent:** {plan.intent}

### Competitors ({len(research.competitors)} found)
```json
{json.dumps(competitor_summary, indent=2)}
```

### Pricing Analysis
```json
{json.dumps(pricing_data, indent=2)}
```

### Menu Intelligence
```json
{json.dumps(menu_data, indent=2)}
```

### Customer Sentiment
```json
{json.dumps(sentiment_data, indent=2)}
```

### Market Signals
```json
{json.dumps(market_data, indent=2)}
```

### Quality Assessment (from Critic Agent)
- **Decision:** {critique.decision}
- **Quality Score:** {critique.overall_quality_score:.2f}/1.00
- **Strengths Identified:** {critique.strengths}
- **Issues Found:** {len(critique.issues_found)} issues
- **Banking Suitability:** {critique.banking_suitability_assessment}

### Research Notes
{research.research_notes}

### Sources Used
{len(research.raw_sources)} sources referenced

---

Generate a comprehensive, professional Market Research Report based on ALL the data above.
Synthesize insights, don't just repeat data. Think like a commercial banker evaluating this restaurant."""

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        planner_output = state.get("planner_output")
        researcher_output = state.get("researcher_output")
        critic_output = state.get("critic_output")

        if not all([planner_output, researcher_output, critic_output]):
            raise ValueError("Missing required outputs from previous agents")

        # Convert dicts to Pydantic models if needed
        if isinstance(planner_output, dict):
            planner_output = PlannerOutput(**planner_output)
        if isinstance(researcher_output, dict):
            researcher_output = ResearcherOutput(**researcher_output)
        if isinstance(critic_output, dict):
            critic_output = CriticOutput(**critic_output)

        try:
            report = self.generate(planner_output, researcher_output, critic_output)
            return {
                **state,
                "report_output": report,
                "current_node": "complete",
                "completed_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"[REPORT] Error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Report error: {str(e)}"],
                "current_node": "error",
            }
