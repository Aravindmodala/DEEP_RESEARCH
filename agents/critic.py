"""CRITIC Agent - Bank-grade quality and risk reviewer."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import ValidationError

from config import AgentConfig
from models.schemas import CriticIssue, CriticOutput, ResearcherOutput
from prompts import CRITIC_SYSTEM_PROMPT
from utils import create_llm


class CriticAgent:
    """
    NODE 3 — CRITIC AGENT
    
    Responsibility: Act as a bank-grade quality & risk reviewer.
    
    Evaluates:
    - Completeness of research
    - Logical competitor selection
    - Grounding in real sources
    - Absence of hallucinated data
    - Suitability for commercial banking decisions
    
    Decisions:
    - ACCEPT: Pass to Report Agent
    - REJECT: Send back to Researcher with required fixes
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        # Critic uses temperature=0.0 for deterministic evaluation
        self.llm = create_llm(config, temperature=0.0)

    def evaluate(self, research: ResearcherOutput) -> CriticOutput:
        """
        Evaluate research quality and decide ACCEPT or REJECT.
        
        Args:
            research: Output from the Researcher agent
            
        Returns:
            CriticOutput with decision and detailed evaluation
        """
        logger.info("[CRITIC] Evaluating research quality...")

        # First, perform rule-based checks
        rule_issues = self._rule_based_evaluation(research)

        # Then, get LLM evaluation
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=self._format_research_for_review(research)),
        ]

        response = self.llm.invoke(messages)
        llm_evaluation = self._parse_llm_evaluation(response.content)

        # Combine rule-based and LLM issues
        all_issues = rule_issues + llm_evaluation.get("issues", [])

        # Calculate final quality score
        quality_score = self._calculate_quality_score(research, all_issues)

        # Determine decision
        critical_issues = [i for i in all_issues if i.severity == "critical"]
        major_issues = [i for i in all_issues if i.severity == "major"]

        if critical_issues:
            decision = "REJECT"
            required_fixes = [f"CRITICAL: {i.description}" for i in critical_issues]
        elif quality_score < self.config.min_quality_score:
            decision = "REJECT"
            required_fixes = [f"Quality score {quality_score:.2f} below threshold {self.config.min_quality_score}"]
            required_fixes.extend([f"MAJOR: {i.description}" for i in major_issues])
        else:
            decision = "ACCEPT"
            required_fixes = []

        # Identify strengths
        strengths = self._identify_strengths(research)

        # Banking suitability assessment
        banking_assessment = self._assess_banking_suitability(research, all_issues)

        output = CriticOutput(
            decision=decision,
            overall_quality_score=quality_score,
            issues_found=all_issues,
            required_fixes=required_fixes,
            strengths=strengths,
            banking_suitability_assessment=banking_assessment,
        )

        logger.info(f"[CRITIC] Decision: {decision}")
        logger.info(f"[CRITIC] Quality Score: {quality_score:.2f}")
        logger.info(f"[CRITIC] Issues found: {len(all_issues)}")

        return output

    def _rule_based_evaluation(self, research: ResearcherOutput) -> list[CriticIssue]:
        """Perform deterministic rule-based checks."""
        issues = []

        # Check competitor data
        if not research.competitors:
            issues.append(
                CriticIssue(
                    severity="critical",
                    category="completeness",
                    description="No competitor data found",
                    affected_section="competitors",
                )
            )
        elif len(research.competitors) < 3:
            issues.append(
                CriticIssue(
                    severity="major",
                    category="completeness",
                    description=f"Only {len(research.competitors)} competitors found (minimum 3 expected)",
                    affected_section="competitors",
                )
            )

        # Check for competitor ratings
        competitors_with_ratings = [c for c in research.competitors if c.rating]
        if research.competitors and len(competitors_with_ratings) < len(research.competitors) * 0.5:
            issues.append(
                CriticIssue(
                    severity="minor",
                    category="completeness",
                    description="Less than 50% of competitors have ratings data",
                    affected_section="competitors",
                )
            )

        # Check menu data (minor issue since some sites block extraction)
        if self.config.require_menu_data:
            if not research.menu_comparison or not research.menu_comparison.target_menu:
                issues.append(
                    CriticIssue(
                        severity="minor",  # Reduced from major - extraction often fails
                        category="completeness",
                        description="Target restaurant menu data not extracted (site may block scraping)",
                        affected_section="menu_comparison",
                    )
                )

        # Check sentiment data
        if self.config.require_sentiment_data:
            if not research.sentiment_analysis:
                issues.append(
                    CriticIssue(
                        severity="major",
                        category="completeness",
                        description="Sentiment analysis missing",
                        affected_section="sentiment_analysis",
                    )
                )
            elif not research.sentiment_analysis.positive_drivers and not research.sentiment_analysis.common_complaints:
                issues.append(
                    CriticIssue(
                        severity="minor",
                        category="completeness",
                        description="Sentiment analysis lacks specific themes",
                        affected_section="sentiment_analysis",
                    )
                )

        # Check market signals
        if not research.market_signals:
            issues.append(
                CriticIssue(
                    severity="major",
                    category="completeness",
                    description="Market signals analysis missing",
                    affected_section="market_signals",
                )
            )

        # Check pricing analysis
        if not research.pricing_analysis or not research.pricing_analysis.price_position:
            issues.append(
                CriticIssue(
                    severity="minor",
                    category="completeness",
                    description="Price positioning not determined",
                    affected_section="pricing_analysis",
                )
            )

        # Check source grounding
        if not research.raw_sources:
            issues.append(
                CriticIssue(
                    severity="major",
                    category="grounding",
                    description="No source references provided",
                    affected_section="raw_sources",
                )
            )
        elif len(research.raw_sources) < 3:
            issues.append(
                CriticIssue(
                    severity="minor",
                    category="grounding",
                    description=f"Only {len(research.raw_sources)} sources cited (minimum 3 expected)",
                    affected_section="raw_sources",
                )
            )

        return issues

    def _format_research_for_review(self, research: ResearcherOutput) -> str:
        """Format research output for LLM review."""
        return f"""## Research Output for Review

### Competitors ({len(research.competitors)} found)
```json
{json.dumps([c.model_dump() for c in research.competitors[:5]], indent=2)}
```

### Menu Comparison
```json
{research.menu_comparison.model_dump_json(indent=2) if research.menu_comparison else "null"}
```

### Pricing Analysis
```json
{research.pricing_analysis.model_dump_json(indent=2) if research.pricing_analysis else "null"}
```

### Sentiment Analysis
```json
{research.sentiment_analysis.model_dump_json(indent=2) if research.sentiment_analysis else "null"}
```

### Market Signals
```json
{research.market_signals.model_dump_json(indent=2) if research.market_signals else "null"}
```

### Sources ({len(research.raw_sources)} total)
```json
{json.dumps([s.model_dump() for s in research.raw_sources[:5]], indent=2)}
```

### Research Notes
{research.research_notes}

---
Evaluate this research for commercial banking suitability. Output your evaluation as JSON."""

    def _parse_llm_evaluation(self, response_text: str) -> dict:
        """Parse LLM evaluation response."""
        try:
            # Try to extract JSON
            import re

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
                    try:
                        issues.append(
                            CriticIssue(
                                severity=issue_data.get("severity", "minor"),
                                category=issue_data.get("category", "logic"),
                                description=issue_data.get("description", ""),
                                affected_section=issue_data.get("affected_section"),
                            )
                        )
                    except:
                        pass

            return {"issues": issues, "raw": data}

        except Exception as e:
            logger.warning(f"[CRITIC] Error parsing LLM evaluation: {e}")
            return {"issues": [], "raw": {}}

    def _calculate_quality_score(
        self, research: ResearcherOutput, issues: list[CriticIssue]
    ) -> float:
        """Calculate overall quality score (0-1)."""
        # Start with base score
        score = 1.0

        # Deduct for issues
        for issue in issues:
            if issue.severity == "critical":
                score -= 0.3
            elif issue.severity == "major":
                score -= 0.15
            else:
                score -= 0.05

        # Bonus for completeness
        completeness_bonus = 0.0
        if research.competitors and len(research.competitors) >= 5:
            completeness_bonus += 0.05
        if research.menu_comparison and research.menu_comparison.target_menu:
            completeness_bonus += 0.05
        if research.sentiment_analysis and research.sentiment_analysis.target_overall_sentiment:
            completeness_bonus += 0.05
        if research.market_signals and research.market_signals.competitor_density:
            completeness_bonus += 0.05
        if len(research.raw_sources) >= 5:
            completeness_bonus += 0.05

        score += completeness_bonus

        # Clamp to 0-1
        return max(0.0, min(1.0, score))

    def _identify_strengths(self, research: ResearcherOutput) -> list[str]:
        """Identify research strengths."""
        strengths = []

        if research.competitors and len(research.competitors) >= 5:
            strengths.append(f"Comprehensive competitor analysis ({len(research.competitors)} competitors)")

        if research.menu_comparison and research.menu_comparison.target_menu:
            strengths.append("Target restaurant menu data extracted")

        if research.sentiment_analysis and research.sentiment_analysis.sample_reviews:
            strengths.append("Customer sentiment backed by actual reviews")

        if research.market_signals and research.market_signals.market_saturation:
            strengths.append("Market saturation assessment provided")

        if research.raw_sources and len(research.raw_sources) >= 5:
            strengths.append(f"Well-sourced research ({len(research.raw_sources)} sources)")

        if research.pricing_analysis and research.pricing_analysis.price_position:
            strengths.append(f"Clear pricing position identified: {research.pricing_analysis.price_position}")

        return strengths if strengths else ["Research conducted with available data"]

    def _assess_banking_suitability(
        self, research: ResearcherOutput, issues: list[CriticIssue]
    ) -> str:
        """Assess suitability for banking decisions."""
        critical_count = len([i for i in issues if i.severity == "critical"])
        major_count = len([i for i in issues if i.severity == "major"])

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

        # Convert dict to ResearcherOutput if needed
        if isinstance(researcher_output, dict):
            researcher_output = ResearcherOutput(**researcher_output)

        try:
            output = self.evaluate(researcher_output)

            if output.decision == "ACCEPT":
                next_node = "report"
            else:
                iteration = state.get("iteration_count", 0)
                max_iter = state.get("max_iterations", self.config.max_research_iterations)
                if iteration >= max_iter:
                    logger.warning(
                        f"[CRITIC] Max iterations ({max_iter}) reached. Forcing accept."
                    )
                    output.decision = "ACCEPT"
                    output.required_fixes = []
                    next_node = "report"
                else:
                    next_node = "researcher"

            return {
                **state,
                "critic_output": output,
                "current_node": next_node,
            }
        except Exception as e:
            logger.error(f"[CRITIC] Error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Critic error: {str(e)}"],
                "current_node": "error",
            }

