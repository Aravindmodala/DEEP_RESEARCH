"""CRITIC Agent - Bank-grade quality and risk reviewer."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import AgentConfig
from models.schemas import CriticOutput, ResearchIssue
from prompts import CRITIC_SYSTEM_PROMPT
from utils import create_llm


class CriticAgent:
    """
    NODE 3 — CRITIC AGENT
    
    Evaluates research quality using rule-based checks + LLM evaluation.
    Returns validated CriticOutput Pydantic model.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config, temperature=0.0)

    def evaluate(
        self,
        user_query: str,
        plan: dict[str, Any],
        research: dict[str, Any],
    ) -> CriticOutput:
        """
        Evaluate research quality and decide ACCEPT or REJECT.
        
        Args:
            user_query: Original user request
            plan: Planner output dict
            research: Research output dict from Researcher
            
        Returns:
            Validated CriticOutput Pydantic model
        """
        logger.info("[CRITIC] Evaluating research quality...")

        # LLM evaluation (keep the prompt compact; do not dump entire state)
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=self._format_research_for_review(user_query, plan, research)),
        ]

        response = self.llm.invoke(messages)
        llm_evaluation = self._parse_llm_evaluation(response.content)

        # Use only LLM issues (less strict, avoids false negatives from deterministic rules)
        all_issues = llm_evaluation.get("issues", [])

        # Calculate quality score
        quality_score = self._calculate_quality_score(research, all_issues)

        # Determine decision (less strict)
        # Only REJECT if core research is missing (hard blocker)
        competitors = research.get("competitors", []) if isinstance(research, dict) else []
        target = research.get("target") if isinstance(research, dict) else None
        core_ok = bool(target) and isinstance(competitors, list) and len(competitors) >= 3

        decision: Literal["ACCEPT", "REJECT"] = "ACCEPT" if core_ok else "REJECT"
        required_fixes = []
        if decision == "REJECT":
            required_fixes = [
                "Provide target restaurant identification (name/place_id/address) and at least 3 competitors with ratings/distances."
            ]

        # Identify strengths
        strengths = self._identify_strengths(research)

        # Banking suitability assessment
        banking_assessment = self._assess_banking_suitability(research, all_issues)

        # Convert issues to ResearchIssue models
        issues_models = []
        for issue in all_issues:
            severity = issue.get("severity", "minor")
            # Ensure severity is valid
            if severity not in ("critical", "major", "minor"):
                severity = "minor"
            issues_models.append(ResearchIssue(
                severity=severity,
                category=issue.get("category", "general"),
                description=issue.get("description", ""),
                affected_section=issue.get("affected_section"),
            ))

        output = CriticOutput(
            decision=decision,
            overall_quality_score=quality_score,
            issues_found=issues_models,
            required_fixes=required_fixes,
            strengths=strengths,
            banking_suitability_assessment=banking_assessment,
        )

        logger.info(f"[CRITIC] Decision: {decision}")
        logger.info(f"[CRITIC] Quality Score: {quality_score:.2f}")
        logger.info(f"[CRITIC] Issues found: {len(issues_models)}")

        return output

    def _compact_research_for_review(self, research: dict[str, Any]) -> dict[str, Any]:
        """Create a compact representation of research for Critic prompting."""
        competitors = research.get("competitors", []) if isinstance(research, dict) else []
        menu = research.get("menu_comparison", {}) if isinstance(research, dict) else {}
        pricing = research.get("pricing_analysis", {}) if isinstance(research, dict) else {}
        market = research.get("market_signals", {}) if isinstance(research, dict) else {}
        sentiment = research.get("sentiment_analysis", {}) if isinstance(research, dict) else {}
        sources = research.get("raw_sources", []) if isinstance(research, dict) else []

        # Keep only a few competitors (selected set, not "found")
        competitors_compact = []
        if isinstance(competitors, list):
            for c in competitors[:5]:
                if isinstance(c, dict):
                    competitors_compact.append(
                        {
                            "name": c.get("name"),
                            "distance_miles": c.get("distance_miles"),
                            "rating": c.get("rating"),
                            "review_count": c.get("review_count"),
                            "price_level": c.get("price_level"),
                            "website": c.get("website"),
                            "place_id": c.get("place_id"),
                        }
                    )

        # Summarize menu extraction without dumping raw page content
        menu_compact: dict[str, Any] = {"target_menu_present": False, "competitor_menu_count": 0}
        if isinstance(menu, dict):
            target_menu = menu.get("target_menu")
            competitor_menus = menu.get("competitor_menus", [])
            menu_compact["target_menu_present"] = bool(target_menu)
            if isinstance(competitor_menus, list):
                menu_compact["competitor_menu_count"] = len(competitor_menus)
                menu_compact["competitor_menus_sample"] = [
                    {
                        "restaurant_name": m.get("restaurant_name"),
                        "source_url": m.get("source_url"),
                        "status": m.get("status"),
                        "content_length": m.get("content_length"),
                    }
                    for m in competitor_menus[:3]
                    if isinstance(m, dict)
                ]

        # Strip raw reviews to reduce payload
        sentiment_compact = {}
        if isinstance(sentiment, dict):
            sentiment_compact = {
                "comparative_summary": sentiment.get("comparative_summary", ""),
                "restaurants": sentiment.get("restaurants", [])[:5] if isinstance(sentiment.get("restaurants"), list) else [],
            }

        # Keep sources minimal
        sources_compact = []
        if isinstance(sources, list):
            for s in sources[:5]:
                if isinstance(s, dict):
                    sources_compact.append(
                        {
                            "source_type": s.get("source_type"),
                            "title": s.get("title"),
                            "url": s.get("url"),
                            "accessed_at": s.get("accessed_at"),
                        }
                    )

        return {
            "target": research.get("target"),
            "selected_competitors": competitors_compact,
            "menu_summary": menu_compact,
            "pricing_analysis": pricing,
            "market_signals": market,
            "sentiment_analysis": sentiment_compact,
            "sources": sources_compact,
        }

    def _format_research_for_review(self, user_query: str, plan: dict[str, Any], research: dict[str, Any]) -> str:
        """Format a compact critique prompt for the Critic LLM."""
        plan_summary = {
            "target_restaurant": plan.get("target_restaurant"),
            "location": plan.get("location"),
            "cuisine_type": plan.get("cuisine_type"),
            "intent": plan.get("intent"),
        }
        compact = self._compact_research_for_review(research)

        return f"""## User Request
{user_query}

## Research Plan (high level)
```json
{json.dumps(plan_summary, indent=2, default=str)}
```

## Research Output (compact)
```json
{json.dumps(compact, indent=2, default=str)}
```

## Task
Provide a constructive critique of the research output:
- What are the top gaps or risks?
- What should be improved next to strengthen the final report?
- Point out any obvious inconsistencies (if any).

Output your evaluation as JSON using the required schema."""

    def _parse_llm_evaluation(self, response_text: str) -> dict:
        """Parse LLM evaluation response."""
        try:
            json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
            match = re.search(json_pattern, response_text)
            if match:
                data = json.loads(match.group(1))
            else:
                brace_start = response_text.find("{")
                brace_end = response_text.rfind("}") + 1
                if brace_start != -1 and brace_end > brace_start:
                    data = json.loads(response_text[brace_start:brace_end])
                else:
                    data = {}

            # Parse issues from LLM response
            issues = []
            for issue_data in data.get("issues_found", []):
                if isinstance(issue_data, dict):
                    issues.append({
                        "severity": issue_data.get("severity", "minor"),
                        "category": issue_data.get("category", "logic"),
                        "description": issue_data.get("description", ""),
                        "affected_section": issue_data.get("affected_section"),
                    })

            return {"issues": issues, "raw": data}

        except Exception as e:
            logger.warning(f"[CRITIC] Error parsing LLM evaluation: {e}")
            return {"issues": [], "raw": {}}

    def _calculate_quality_score(self, research: dict[str, Any], issues: list[dict]) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0

        for issue in issues:
            severity = issue.get("severity", "minor")
            if severity == "critical":
                score -= 0.3
            elif severity == "major":
                score -= 0.15
            else:
                score -= 0.05

        # Bonus for completeness
        competitors = research.get("competitors", [])
        menu_comparison = research.get("menu_comparison")
        sentiment_analysis = research.get("sentiment_analysis")
        market_signals = research.get("market_signals")
        raw_sources = research.get("raw_sources", [])

        completeness_bonus = 0.0
        if competitors and len(competitors) >= 5:
            completeness_bonus += 0.05
        if menu_comparison and menu_comparison.get("target_menu"):
            completeness_bonus += 0.05
        if sentiment_analysis and sentiment_analysis.get("restaurants"):
            completeness_bonus += 0.05
        if market_signals and market_signals.get("competitor_density"):
            completeness_bonus += 0.05
        if len(raw_sources) >= 5:
            completeness_bonus += 0.05

        score += completeness_bonus
        return max(0.0, min(1.0, score))

    def _identify_strengths(self, research: dict[str, Any]) -> list[str]:
        """Identify research strengths."""
        strengths = []
        
        competitors = research.get("competitors", [])
        menu_comparison = research.get("menu_comparison")
        sentiment_analysis = research.get("sentiment_analysis")
        market_signals = research.get("market_signals")
        pricing_analysis = research.get("pricing_analysis")
        raw_sources = research.get("raw_sources", [])

        if competitors and len(competitors) >= 5:
            strengths.append(f"Comprehensive competitor analysis ({len(competitors)} competitors)")

        if menu_comparison and menu_comparison.get("target_menu"):
            strengths.append("Target restaurant menu data extracted")

        if sentiment_analysis and sentiment_analysis.get("restaurants"):
            strengths.append(f"LLM-analyzed sentiment for {len(sentiment_analysis['restaurants'])} restaurants")

        if market_signals and market_signals.get("market_saturation"):
            strengths.append("Market saturation assessment provided")

        if raw_sources and len(raw_sources) >= 5:
            strengths.append(f"Well-sourced research ({len(raw_sources)} sources)")

        if pricing_analysis and pricing_analysis.get("price_position"):
            strengths.append(f"Clear pricing position identified: {pricing_analysis['price_position']}")

        return strengths if strengths else ["Research conducted with available data"]

    def _assess_banking_suitability(self, research: dict[str, Any], issues: list[dict]) -> str:
        """Assess suitability for banking decisions."""
        critical_count = len([i for i in issues if i.get("severity") == "critical"])
        major_count = len([i for i in issues if i.get("severity") == "major"])

        if critical_count > 0:
            return (
                f"NOT SUITABLE for banking decisions. {critical_count} critical issues "
                "require immediate resolution before this research can support lending decisions."
            )
        elif major_count >= 3:
            return (
                f"MARGINALLY SUITABLE. {major_count} major issues identified. "
                "Research provides baseline intelligence but gaps exist. "
                "Additional due diligence recommended."
            )
        elif major_count > 0:
            return (
                f"SUITABLE WITH CAVEATS. {major_count} major issue(s) noted. "
                "Research provides adequate foundation for initial credit assessment. "
                "Standard verification procedures should be followed."
            )
        else:
            return (
                "FULLY SUITABLE for commercial banking decisions. "
                "Research is comprehensive, well-sourced, and provides actionable intelligence "
                "for lending and expansion analysis."
            )

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        researcher_output = state.get("researcher_output")
        if not researcher_output:
            raise ValueError("No researcher_output found in state")

        try:
            output = self.evaluate(researcher_output)  # researcher_output is already a dict

            if output.decision == "ACCEPT":
                next_node = "report"
            else:
                iteration = state.get("iteration_count", 0)
                max_iter = state.get("max_iterations", self.config.max_research_iterations)
                if iteration >= max_iter:
                    logger.warning(f"[CRITIC] Max iterations ({max_iter}) reached. Forcing accept.")
                    output.decision = "ACCEPT"
                    output.required_fixes = []
                    next_node = "report"
                else:
                    next_node = "researcher"

            return {
                **state,
                "critic_output": output.model_dump(),  # Convert Pydantic to dict for state
                "current_node": next_node,
            }
        except Exception as e:
            logger.error(f"[CRITIC] Error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Critic error: {str(e)}"],
                "current_node": "error",
            }
