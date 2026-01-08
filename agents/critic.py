"""CRITIC Agent - Bank-grade quality and risk reviewer."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import AgentConfig
from prompts import CRITIC_SYSTEM_PROMPT
from utils import create_llm


class CriticAgent:
    """
    NODE 3 — CRITIC AGENT
    
    Evaluates research quality using rule-based checks + LLM evaluation.
    Returns plain dict (no strict validation - Report LLM can handle it).
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config, temperature=0.0)

    def evaluate(self, research: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate research quality and decide ACCEPT or REJECT.
        
        Args:
            research: Research output dict from Researcher
            
        Returns:
            Plain dict with evaluation results
        """
        logger.info("[CRITIC] Evaluating research quality...")

        # Rule-based checks on dict keys
        rule_issues = self._rule_based_evaluation(research)

        # LLM evaluation
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=self._format_research_for_review(research)),
        ]

        response = self.llm.invoke(messages)
        llm_evaluation = self._parse_llm_evaluation(response.content)

        # Combine issues
        all_issues = rule_issues + llm_evaluation.get("issues", [])

        # Calculate quality score
        quality_score = self._calculate_quality_score(research, all_issues)

        # Determine decision
        critical_issues = [i for i in all_issues if i.get("severity") == "critical"]
        major_issues = [i for i in all_issues if i.get("severity") == "major"]

        if critical_issues:
            decision = "REJECT"
            required_fixes = [f"CRITICAL: {i.get('description', '')}" for i in critical_issues]
        elif quality_score < self.config.min_quality_score:
            decision = "REJECT"
            required_fixes = [f"Quality score {quality_score:.2f} below threshold {self.config.min_quality_score}"]
            required_fixes.extend([f"MAJOR: {i.get('description', '')}" for i in major_issues])
        else:
            decision = "ACCEPT"
            required_fixes = []

        # Identify strengths
        strengths = self._identify_strengths(research)

        # Banking suitability assessment
        banking_assessment = self._assess_banking_suitability(research, all_issues)

        output = {
            "decision": decision,
            "overall_quality_score": quality_score,
            "issues_found": all_issues,
            "required_fixes": required_fixes,
            "strengths": strengths,
            "banking_suitability_assessment": banking_assessment,
        }

        logger.info(f"[CRITIC] Decision: {decision}")
        logger.info(f"[CRITIC] Quality Score: {quality_score:.2f}")
        logger.info(f"[CRITIC] Issues found: {len(all_issues)}")

        return output

    def _rule_based_evaluation(self, research: dict[str, Any]) -> list[dict]:
        """Perform deterministic rule-based checks on research dict."""
        issues = []
        
        competitors = research.get("competitors", [])
        menu_comparison = research.get("menu_comparison")
        sentiment_analysis = research.get("sentiment_analysis")
        market_signals = research.get("market_signals")
        pricing_analysis = research.get("pricing_analysis")
        raw_sources = research.get("raw_sources", [])

        # Check competitor data
        if not competitors:
            issues.append({
                "severity": "critical",
                "category": "completeness",
                "description": "No competitor data found",
                "affected_section": "competitors",
            })
        elif len(competitors) < 3:
            issues.append({
                "severity": "major",
                "category": "completeness",
                "description": f"Only {len(competitors)} competitors found (minimum 3 expected)",
                "affected_section": "competitors",
            })

        # Check for competitor ratings
        if competitors:
            competitors_with_ratings = [c for c in competitors if c.get("rating")]
            if len(competitors_with_ratings) < len(competitors) * 0.5:
                issues.append({
                    "severity": "minor",
                    "category": "completeness",
                    "description": "Less than 50% of competitors have ratings data",
                    "affected_section": "competitors",
                })

        # Check menu data
        if self.config.require_menu_data:
            if not menu_comparison or not menu_comparison.get("target_menu"):
                issues.append({
                    "severity": "major",
                    "category": "completeness",
                    "description": "Target restaurant menu data not extracted",
                    "affected_section": "menu_comparison",
                })

        # Check sentiment data
        if self.config.require_sentiment_data:
            if not sentiment_analysis:
                issues.append({
                    "severity": "major",
                    "category": "completeness",
                    "description": "Sentiment analysis missing",
                    "affected_section": "sentiment_analysis",
                })
            elif not sentiment_analysis.get("restaurants") and not sentiment_analysis.get("raw_reviews"):
                issues.append({
                    "severity": "minor",
                    "category": "completeness",
                    "description": "Sentiment analysis lacks restaurant reviews",
                    "affected_section": "sentiment_analysis",
                })

        # Check market signals
        if not market_signals:
            issues.append({
                "severity": "major",
                "category": "completeness",
                "description": "Market signals analysis missing",
                "affected_section": "market_signals",
            })

        # Check pricing analysis
        if not pricing_analysis or not pricing_analysis.get("price_position"):
            issues.append({
                "severity": "minor",
                "category": "completeness",
                "description": "Price positioning not determined",
                "affected_section": "pricing_analysis",
            })

        # Check source grounding
        if not raw_sources:
            issues.append({
                "severity": "major",
                "category": "grounding",
                "description": "No source references provided",
                "affected_section": "raw_sources",
            })
        elif len(raw_sources) < 3:
            issues.append({
                "severity": "minor",
                "category": "grounding",
                "description": f"Only {len(raw_sources)} sources cited (minimum 3 expected)",
                "affected_section": "raw_sources",
            })

        return issues

    def _format_research_for_review(self, research: dict[str, Any]) -> str:
        """Format research output for LLM review."""
        competitors = research.get("competitors", [])
        
        return f"""## Research Output for Review

### Competitors ({len(competitors)} found)
```json
{json.dumps(competitors[:5], indent=2, default=str)}
```

### Menu Comparison
```json
{json.dumps(research.get("menu_comparison"), indent=2, default=str)}
```

### Pricing Analysis
```json
{json.dumps(research.get("pricing_analysis"), indent=2, default=str)}
```

### Sentiment Analysis
```json
{json.dumps(research.get("sentiment_analysis"), indent=2, default=str)}
```

### Market Signals
```json
{json.dumps(research.get("market_signals"), indent=2, default=str)}
```

### Sources ({len(research.get("raw_sources", []))} total)
```json
{json.dumps(research.get("raw_sources", [])[:5], indent=2, default=str)}
```

### Research Notes
{research.get("research_notes", [])}

---
Evaluate this research for commercial banking suitability. Output your evaluation as JSON."""

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

            if output["decision"] == "ACCEPT":
                next_node = "report"
            else:
                iteration = state.get("iteration_count", 0)
                max_iter = state.get("max_iterations", self.config.max_research_iterations)
                if iteration >= max_iter:
                    logger.warning(f"[CRITIC] Max iterations ({max_iter}) reached. Forcing accept.")
                    output["decision"] = "ACCEPT"
                    output["required_fixes"] = []
                    next_node = "report"
                else:
                    next_node = "researcher"

            return {
                **state,
                "critic_output": output,  # Plain dict
                "current_node": next_node,
            }
        except Exception as e:
            logger.error(f"[CRITIC] Error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Critic error: {str(e)}"],
                "current_node": "error",
            }
