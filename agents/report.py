"""REPORT Agent - Generates client-ready market research reports."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import AgentConfig
from models.schemas import ReportOutput, SourceReference
from prompts import REPORT_SYSTEM_PROMPT
from utils import create_llm


class ReportAgent:
    """
    NODE 4 — REPORT AGENT
    
    Generates a client-ready Market Research Report using LLM with structured output.
    
    This is the ONLY agent that uses Pydantic validation because the final
    report needs a guaranteed structure for UI rendering.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        base_llm = create_llm(config, temperature=0.3)
        self.llm = base_llm.with_structured_output(ReportOutput)

    def generate(
        self,
        plan: dict[str, Any],
        research: dict[str, Any],
        critique: dict[str, Any],
    ) -> ReportOutput:
        """
        Generate a comprehensive market research report.
        
        All inputs are plain dicts - only the output is validated.
        
        Args:
            plan: Research plan dict from Planner
            research: Research findings dict from Researcher
            critique: Quality assessment dict from Critic
            
        Returns:
            ReportOutput (validated Pydantic model for UI rendering)
        """
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        
        logger.info(f"[REPORT] Generating report for {target} in {location}")

        context = self._prepare_context(plan, research, critique)

        messages = [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        # LLM generates structured output directly
        report: ReportOutput = self.llm.invoke(messages)

        # Override metadata fields with actual values
        report.report_title = f"Market Research Report: {target}"
        report.generated_at = datetime.now().isoformat()
        report.target_restaurant = target
        report.location = location
        
        # Convert raw_sources to SourceReference objects
        raw_sources = research.get("raw_sources", [])
        report.appendix_sources = [
            SourceReference(**s) if isinstance(s, dict) else s
            for s in raw_sources
        ]

        logger.info("[REPORT] Report generation complete")
        return report

    def _prepare_context(
        self,
        plan: dict[str, Any],
        research: dict[str, Any],
        critique: dict[str, Any],
    ) -> str:
        """Prepare comprehensive context for report generation."""
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        cuisine_type = plan.get("cuisine_type", "unknown")
        intent = plan.get("intent", "")
        
        competitors = research.get("competitors", [])
        pricing = research.get("pricing_analysis", {})
        sentiment = research.get("sentiment_analysis", {})
        market_signals = research.get("market_signals", {})
        menu = research.get("menu_comparison", {})
        raw_sources = research.get("raw_sources", [])
        research_notes = research.get("research_notes", [])
        
        decision = critique.get("decision", "UNKNOWN")
        quality_score = critique.get("overall_quality_score", 0)
        strengths = critique.get("strengths", [])
        banking_assessment = critique.get("banking_suitability_assessment", "")

        # Format competitors summary
        competitor_summary = [
            {
                "name": c.get("name"),
                "distance_miles": c.get("distance_miles"),
                "rating": c.get("rating"),
                "review_count": c.get("review_count"),
                "price_level": c.get("price_level"),
                "cuisine_type": c.get("cuisine_type"),
            }
            for c in competitors[:10]
        ]

        return f"""## RESEARCH DATA FOR REPORT GENERATION

### Target Restaurant
- **Name:** {target}
- **Location:** {location}
- **Cuisine Type:** {cuisine_type}
- **Research Intent:** {intent}

### Competitors ({len(competitors)} found)
```json
{json.dumps(competitor_summary, indent=2, default=str)}
```

### Pricing Analysis
```json
{json.dumps(pricing, indent=2, default=str)}
```

### Menu Intelligence
```json
{json.dumps(menu, indent=2, default=str)}
```

### Customer Sentiment
```json
{json.dumps(sentiment, indent=2, default=str)}
```

### Market Signals
```json
{json.dumps(market_signals, indent=2, default=str)}
```

### Quality Assessment (from Critic Agent)
- **Decision:** {decision}
- **Quality Score:** {quality_score:.2f}/1.00
- **Strengths Identified:** {strengths}
- **Banking Suitability:** {banking_assessment}

### Research Notes
{research_notes}

### Sources Used
{len(raw_sources)} sources referenced

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

        # All inputs are dicts - no conversion needed

        try:
            report = self.generate(planner_output, researcher_output, critic_output)
            return {
                **state,
                "report_output": report.model_dump(),  # Convert to dict for state
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
