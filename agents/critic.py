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
    NODE 4 — CRITIC AGENT

    Evaluates research + analysis quality using rule-based checks + LLM evaluation.
    Now also evaluates the Analyst output for depth of analysis.
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
        analyst: dict[str, Any] | None = None,
    ) -> CriticOutput:
        """
        Evaluate research + analysis quality and decide ACCEPT or REJECT.

        Args:
            user_query: Original user request
            plan: Planner output dict
            research: Research output dict from Researcher
            analyst: Analyst output dict (optional, may be None on first pass)

        Returns:
            Validated CriticOutput Pydantic model
        """
        logger.info("[CRITIC] Evaluating research and analysis quality...")

        # LLM evaluation
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=self._format_research_for_review(user_query, plan, research, analyst)),
        ]

        response = self.llm.invoke(messages)
        llm_evaluation = self._parse_llm_evaluation(response.content)

        all_issues = llm_evaluation.get("issues", [])

        # Calculate quality score
        quality_score = self._calculate_quality_score(research, analyst, all_issues)

        # Determine decision
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
        strengths = self._identify_strengths(research, analyst)

        # Banking suitability assessment
        banking_assessment = self._assess_banking_suitability(research, all_issues)

        # Convert issues to ResearchIssue models
        issues_models = []
        for issue in all_issues:
            severity = issue.get("severity", "minor")
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

    def _compact_research_for_review(self, research: dict[str, Any], analyst: dict[str, Any] | None) -> dict[str, Any]:
        """Create a compact representation of research + analysis for Critic prompting."""
        competitors = research.get("competitors", []) if isinstance(research, dict) else []
        raw_reviews = research.get("raw_reviews", {}) if isinstance(research, dict) else {}
        raw_menus = research.get("raw_menus", []) if isinstance(research, dict) else []
        business_intel = research.get("business_intel", {}) if isinstance(research, dict) else {}
        market_data = research.get("market_data", []) if isinstance(research, dict) else []
        sources = research.get("raw_sources", []) if isinstance(research, dict) else []

        # Compact competitors
        competitors_compact = []
        if isinstance(competitors, list):
            for c in competitors[:5]:
                if isinstance(c, dict):
                    competitors_compact.append({
                        "name": c.get("name"),
                        "distance_miles": c.get("distance_miles"),
                        "rating": c.get("rating"),
                        "review_count": c.get("review_count"),
                        "price_level": c.get("price_level"),
                        "website": c.get("website"),
                        "place_id": c.get("place_id"),
                    })

        # Compact analysis summary
        analysis_summary = {}
        if analyst and isinstance(analyst, dict):
            swot = analyst.get("swot_analyses", [])
            sentiment = analyst.get("sentiment_analysis", {})
            parsed_menus = analyst.get("parsed_menus", [])
            menu_comparison = analyst.get("menu_comparison", {})
            keywords = analyst.get("keyword_analysis", {})
            metrics = analyst.get("performance_metrics", [])
            strategy = analyst.get("strategic_recommendations", {})

            analysis_summary = {
                "swot_count": len(swot) if isinstance(swot, list) else 0,
                "swot_has_content": any(s.get("strengths") for s in swot) if isinstance(swot, list) else False,
                "sentiment_restaurant_count": len(sentiment.get("restaurants", [])) if isinstance(sentiment, dict) else 0,
                "sentiment_has_dimensions": any(
                    r.get("service_quality", {}).get("score") for r in sentiment.get("restaurants", [])
                ) if isinstance(sentiment, dict) else False,
                "parsed_menus_count": len(parsed_menus) if isinstance(parsed_menus, list) else 0,
                "menu_comparison_items": len(menu_comparison.get("item_comparisons", [])) if isinstance(menu_comparison, dict) else 0,
                "keywords_analyzed": bool(keywords.get("positive_keywords")) if isinstance(keywords, dict) else False,
                "metrics_count": len(metrics) if isinstance(metrics, list) else 0,
                "strategy_pillars": len(strategy.get("pillars", [])) if isinstance(strategy, dict) else 0,
            }

        return {
            "target": research.get("target"),
            "selected_competitors": competitors_compact,
            "raw_reviews_count": len(raw_reviews) if isinstance(raw_reviews, dict) else 0,
            "raw_menus_count": len(raw_menus) if isinstance(raw_menus, list) else 0,
            "business_intel_present": bool(business_intel),
            "market_data_count": len(market_data) if isinstance(market_data, list) else 0,
            "sources_count": len(sources) if isinstance(sources, list) else 0,
            "analysis_summary": analysis_summary,
        }

    def _format_research_for_review(
        self, user_query: str, plan: dict[str, Any],
        research: dict[str, Any], analyst: dict[str, Any] | None
    ) -> str:
        """Format a compact critique prompt for the Critic LLM."""
        plan_summary = {
            "target_restaurant": plan.get("target_restaurant"),
            "location": plan.get("location"),
            "cuisine_type": plan.get("cuisine_type"),
            "intent": plan.get("intent"),
        }
        compact = self._compact_research_for_review(research, analyst)

        return f"""## User Request
{user_query}

## Research Plan (high level)
```json
{json.dumps(plan_summary, indent=2, default=str)}
```

## Research + Analysis Output (compact)
```json
{json.dumps(compact, indent=2, default=str)}
```

## Task
Provide a constructive critique of the research and analysis output:
- What are the top gaps or risks?
- Is the analysis depth sufficient (SWOT, sentiment dimensions, menu parsing, keywords)?
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

    def _calculate_quality_score(
        self, research: dict[str, Any], analyst: dict[str, Any] | None, issues: list[dict]
    ) -> float:
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
        raw_reviews = research.get("raw_reviews", {})
        raw_menus = research.get("raw_menus", [])
        raw_sources = research.get("raw_sources", [])

        completeness_bonus = 0.0
        if competitors and len(competitors) >= 5:
            completeness_bonus += 0.05
        if raw_reviews and len(raw_reviews) >= 2:
            completeness_bonus += 0.05
        if raw_menus and len(raw_menus) >= 2:
            completeness_bonus += 0.05
        if len(raw_sources) >= 5:
            completeness_bonus += 0.05

        # Analysis depth bonus
        if analyst and isinstance(analyst, dict):
            swot = analyst.get("swot_analyses", [])
            sentiment = analyst.get("sentiment_analysis", {})
            parsed_menus = analyst.get("parsed_menus", [])
            keywords = analyst.get("keyword_analysis", {})
            strategy = analyst.get("strategic_recommendations", {})

            if isinstance(swot, list) and len(swot) >= 2:
                completeness_bonus += 0.05
            if isinstance(sentiment, dict) and len(sentiment.get("restaurants", [])) >= 2:
                completeness_bonus += 0.05
            if isinstance(parsed_menus, list) and len(parsed_menus) >= 2:
                completeness_bonus += 0.05
            if isinstance(keywords, dict) and keywords.get("positive_keywords"):
                completeness_bonus += 0.03
            if isinstance(strategy, dict) and len(strategy.get("pillars", [])) >= 3:
                completeness_bonus += 0.02

        score += completeness_bonus
        return max(0.0, min(1.0, score))

    def _identify_strengths(self, research: dict[str, Any], analyst: dict[str, Any] | None) -> list[str]:
        """Identify research and analysis strengths."""
        strengths = []

        competitors = research.get("competitors", [])
        raw_reviews = research.get("raw_reviews", {})
        raw_menus = research.get("raw_menus", [])
        raw_sources = research.get("raw_sources", [])
        business_intel = research.get("business_intel", {})

        if competitors and len(competitors) >= 5:
            strengths.append(f"Comprehensive competitor discovery ({len(competitors)} competitors)")

        if raw_reviews and len(raw_reviews) >= 3:
            strengths.append(f"Multi-source review collection ({len(raw_reviews)} restaurants)")

        if raw_menus and len(raw_menus) >= 2:
            strengths.append(f"Menu data collected for {len(raw_menus)} restaurants")

        if raw_sources and len(raw_sources) >= 10:
            strengths.append(f"Well-sourced research ({len(raw_sources)} sources)")

        if business_intel:
            strengths.append("Business intelligence gathered")

        # Analysis strengths
        if analyst and isinstance(analyst, dict):
            swot = analyst.get("swot_analyses", [])
            sentiment = analyst.get("sentiment_analysis", {})
            parsed_menus = analyst.get("parsed_menus", [])
            keywords = analyst.get("keyword_analysis", {})
            strategy = analyst.get("strategic_recommendations", {})

            if isinstance(swot, list) and len(swot) >= 2:
                strengths.append(f"SWOT analysis for {len(swot)} restaurants")
            if isinstance(sentiment, dict) and sentiment.get("restaurants"):
                strengths.append(f"Multi-dimensional sentiment analysis for {len(sentiment['restaurants'])} restaurants")
            if isinstance(parsed_menus, list) and len(parsed_menus) >= 2:
                strengths.append(f"Structured menus parsed for {len(parsed_menus)} restaurants")
            if isinstance(keywords, dict) and keywords.get("positive_keywords"):
                strengths.append("Review keyword frequency analysis completed")
            if isinstance(strategy, dict) and strategy.get("pillars"):
                strengths.append(f"Strategic recommendations with {len(strategy['pillars'])} pillars")

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

        user_query = state.get("user_query", "")
        planner_output = state.get("planner_output", {})
        analyst_output = state.get("analyst_output")

        try:
            output = self.evaluate(user_query, planner_output, researcher_output, analyst_output)

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
                "critic_output": output.model_dump(),
                "current_node": next_node,
            }
        except Exception as e:
            logger.error(f"[CRITIC] Error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Critic error: {str(e)}"],
                "current_node": "error",
            }
