"""REPORT Agent - Generates client-ready market research reports."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import ValidationError

from config import AgentConfig
from models.schemas import (
    BankingRelevanceSection,
    CompetitiveLandscapeSection,
    CriticOutput,
    ExecutiveSummary,
    FinalRecommendation,
    MenuPricingSection,
    PlannerOutput,
    ReportOutput,
    ResearcherOutput,
    SentimentSection,
)
from prompts import REPORT_SYSTEM_PROMPT
from utils import create_llm


class ReportAgent:
    """
    NODE 4 — REPORT AGENT
    
    Responsibility: Generate a client-ready Market Research Report
    for Commercial Banking.
    
    This is decision-grade intelligence, not a summary.
    
    Report Sections:
    1. Executive Summary
    2. Competitive Landscape
    3. Menu & Pricing Position
    4. Customer Sentiment Insights
    5. Commercial Banking Relevance
    6. Final Recommendation
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        # Report uses temperature=0.2 for slightly creative writing
        self.llm = create_llm(config, temperature=0.2)

    def generate(
        self,
        plan: PlannerOutput,
        research: ResearcherOutput,
        critique: CriticOutput,
    ) -> ReportOutput:
        """
        Generate a comprehensive market research report.
        
        Args:
            plan: Original research plan
            research: Research findings
            critique: Quality assessment
            
        Returns:
            ReportOutput with full structured report
        """
        logger.info(
            f"[REPORT] Generating report for {plan.target_restaurant} in {plan.location}"
        )

        # Prepare context for LLM
        context = self._prepare_context(plan, research, critique)

        # Generate report via LLM
        messages = [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        response = self.llm.invoke(messages)
        
        # Parse LLM response into structured report
        report = self._parse_report(
            response.content, plan, research, critique
        )

        logger.info("[REPORT] Report generation complete")
        return report

    def _prepare_context(
        self,
        plan: PlannerOutput,
        research: ResearcherOutput,
        critique: CriticOutput,
    ) -> str:
        """Prepare comprehensive context for report generation."""
        # Format competitors for context
        competitor_summary = []
        for c in research.competitors[:10]:
            competitor_summary.append({
                "name": c.name,
                "distance": f"{c.distance_miles} miles",
                "rating": c.rating,
                "reviews": c.review_count,
                "price": c.price_level,
            })

        # Format sentiment
        sentiment_data = {}
        if research.sentiment_analysis:
            sentiment_data = {
                "overall": research.sentiment_analysis.target_overall_sentiment,
                "positives": research.sentiment_analysis.positive_drivers,
                "negatives": research.sentiment_analysis.common_complaints,
                "reviews": research.sentiment_analysis.sample_reviews[:3],
            }

        # Format market signals
        market_data = {}
        if research.market_signals:
            market_data = {
                "competitor_density": research.market_signals.competitor_density,
                "saturation": research.market_signals.market_saturation,
                "avg_rating": research.market_signals.avg_competitor_rating,
                "foot_traffic": research.market_signals.foot_traffic_indicators,
                "growth": research.market_signals.growth_indicators,
                "risks": research.market_signals.risk_indicators,
            }

        return f"""## Research Input for Report Generation

### Target Restaurant
- **Name:** {plan.target_restaurant}
- **Location:** {plan.location}
- **Research Intent:** {plan.intent}

### Competitive Analysis
**{len(research.competitors)} competitors identified**
```json
{json.dumps(competitor_summary, indent=2)}
```

### Pricing Position
- **Position:** {research.pricing_analysis.price_position if research.pricing_analysis else "Unknown"}
- **Target Avg Price:** {research.pricing_analysis.target_avg_price if research.pricing_analysis else "N/A"}
- **Market Avg Price:** {research.pricing_analysis.market_avg_price if research.pricing_analysis else "N/A"}

### Menu Intelligence
- **Target Menu Items:** {len(research.menu_comparison.target_menu.items) if research.menu_comparison and research.menu_comparison.target_menu else 0}
- **Competitor Menus Analyzed:** {len(research.menu_comparison.competitor_menus) if research.menu_comparison else 0}

### Customer Sentiment
```json
{json.dumps(sentiment_data, indent=2)}
```

### Market Signals
```json
{json.dumps(market_data, indent=2)}
```

### Quality Assessment (Critic)
- **Decision:** {critique.decision}
- **Quality Score:** {critique.overall_quality_score:.2f}
- **Strengths:** {critique.strengths}
- **Banking Suitability:** {critique.banking_suitability_assessment}

### Sources Used
{len(research.raw_sources)} sources referenced

### Research Notes
{research.research_notes}

---
Generate a comprehensive Market Research Report suitable for commercial banking decisions.
The report should be professional, neutral, and data-backed.
Output as structured JSON matching the ReportOutput schema."""

    def _parse_report(
        self,
        response_text: str,
        plan: PlannerOutput,
        research: ResearcherOutput,
        critique: CriticOutput,
    ) -> ReportOutput:
        """Parse LLM response into structured ReportOutput."""
        try:
            # Try to extract JSON from response
            json_data = self._extract_json(response_text)
        except Exception as e:
            logger.warning(f"[REPORT] Error parsing LLM JSON: {e}")
            json_data = {}

        # Build report with fallbacks
        now = datetime.now().isoformat()

        # Executive Summary
        exec_summary = self._build_executive_summary(json_data, plan, research, critique)

        # Competitive Landscape
        competitive = self._build_competitive_section(json_data, research)

        # Menu & Pricing
        menu_pricing = self._build_menu_pricing_section(json_data, research)

        # Customer Sentiment
        sentiment = self._build_sentiment_section(json_data, research)

        # Banking Relevance
        banking = self._build_banking_section(json_data, research, critique)

        # Final Recommendation
        recommendation = self._build_recommendation(json_data, research, critique)

        # Disclaimers
        disclaimers = [
            "This report is based on publicly available data and automated analysis.",
            "Findings should be verified through standard due diligence procedures.",
            "Market conditions may have changed since data collection.",
            "This analysis does not constitute financial advice.",
        ]

        return ReportOutput(
            report_title=f"Market Research Report: {plan.target_restaurant}",
            generated_at=now,
            target_restaurant=plan.target_restaurant,
            location=plan.location,
            executive_summary=exec_summary,
            competitive_landscape=competitive,
            menu_pricing=menu_pricing,
            customer_sentiment=sentiment,
            banking_relevance=banking,
            final_recommendation=recommendation,
            appendix_sources=research.raw_sources,
            disclaimers=disclaimers,
        )

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from response text."""
        import re

        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass

        # Find JSON block
        json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        # Find raw JSON
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end])
            except:
                pass

        return {}

    def _build_executive_summary(
        self,
        json_data: dict,
        plan: PlannerOutput,
        research: ResearcherOutput,
        critique: CriticOutput,
    ) -> ExecutiveSummary:
        """Build executive summary section."""
        # Try to use LLM-generated content
        llm_summary = json_data.get("executive_summary", {})

        # Generate key findings
        key_findings = llm_summary.get("key_findings", [])
        if not key_findings:
            key_findings = []
            if research.competitors:
                key_findings.append(
                    f"{len(research.competitors)} direct competitors identified within the target area"
                )
            if research.market_signals and research.market_signals.market_saturation:
                key_findings.append(
                    f"Market saturation level: {research.market_signals.market_saturation}"
                )
            if research.pricing_analysis and research.pricing_analysis.price_position:
                key_findings.append(
                    f"Target restaurant positioned in the {research.pricing_analysis.price_position} segment"
                )
            if research.sentiment_analysis and research.sentiment_analysis.target_overall_sentiment:
                sentiment_pct = research.sentiment_analysis.target_overall_sentiment * 100
                key_findings.append(
                    f"Customer sentiment score: {sentiment_pct:.0f}%"
                )
            if research.market_signals and research.market_signals.avg_competitor_rating:
                key_findings.append(
                    f"Average competitor rating: {research.market_signals.avg_competitor_rating:.1f}/5.0"
                )

        # Generate lending implications
        lending = llm_summary.get("lending_implications", "")
        if not lending:
            if critique.overall_quality_score >= 0.8:
                lending = "Strong market position supports favorable lending consideration."
            elif critique.overall_quality_score >= 0.6:
                lending = "Moderate market position warrants standard underwriting scrutiny."
            else:
                lending = "Elevated market risk factors require enhanced due diligence."

        # Generate expansion implications
        expansion = llm_summary.get("expansion_implications", "")
        if not expansion:
            if research.market_signals:
                sat = research.market_signals.market_saturation
                if sat == "low":
                    expansion = "Low market saturation suggests favorable conditions for expansion."
                elif sat == "moderate":
                    expansion = "Moderate competition indicates selective expansion opportunities."
                else:
                    expansion = "High market saturation may limit expansion viability."
            else:
                expansion = "Expansion assessment requires additional market analysis."

        overview = llm_summary.get(
            "overview",
            f"This report presents a comprehensive market analysis of {plan.target_restaurant} "
            f"located in {plan.location}, conducted to support {plan.intent}."
        )

        return ExecutiveSummary(
            overview=overview,
            key_findings=key_findings[:5],
            lending_implications=lending,
            expansion_implications=expansion,
        )

    def _build_competitive_section(
        self, json_data: dict, research: ResearcherOutput
    ) -> CompetitiveLandscapeSection:
        """Build competitive landscape section."""
        llm_section = json_data.get("competitive_landscape", {})

        # Top competitors
        top_competitors = []
        for c in research.competitors[:5]:
            top_competitors.append({
                "name": c.name,
                "rating": c.rating,
                "distance": f"{c.distance_miles} miles",
                "price_level": c.price_level,
                "review_count": c.review_count,
            })

        # Determine saturation level
        saturation = "unknown"
        if research.market_signals and research.market_signals.market_saturation:
            saturation = research.market_signals.market_saturation

        # Generate summary
        summary = llm_section.get(
            "summary",
            f"Analysis identified {len(research.competitors)} competing restaurants in the target market area. "
            f"The competitive density is classified as {saturation}."
        )

        # Advantages and disadvantages
        advantages = llm_section.get("competitive_advantages", [])
        if not advantages:
            if research.market_signals and research.market_signals.avg_competitor_rating:
                advantages.append("Established market presence")
            if research.pricing_analysis and research.pricing_analysis.price_position == "mid-range":
                advantages.append("Competitive pricing strategy")

        disadvantages = llm_section.get("competitive_disadvantages", [])
        if not disadvantages:
            if saturation in ["high", "oversaturated"]:
                disadvantages.append("Significant competitive pressure")
            if len(research.competitors) > 10:
                disadvantages.append("Crowded market segment")

        return CompetitiveLandscapeSection(
            summary=summary,
            competitor_count=len(research.competitors),
            market_saturation_level=saturation,
            top_competitors=top_competitors,
            competitive_advantages=advantages if advantages else ["Market presence established"],
            competitive_disadvantages=disadvantages if disadvantages else ["Standard competitive risks"],
        )

    def _build_menu_pricing_section(
        self, json_data: dict, research: ResearcherOutput
    ) -> MenuPricingSection:
        """Build menu and pricing section."""
        llm_section = json_data.get("menu_pricing", {})

        # Price position
        price_position = "unknown"
        if research.pricing_analysis and research.pricing_analysis.price_position:
            price_position = research.pricing_analysis.price_position

        # Summary
        summary = llm_section.get(
            "summary",
            f"The target restaurant operates in the {price_position} price segment. "
            "Pricing strategy appears aligned with local market expectations."
        )

        # Price comparison narrative
        comparison = llm_section.get("price_comparison_narrative", "")
        if not comparison:
            if research.pricing_analysis:
                if research.pricing_analysis.target_avg_price and research.pricing_analysis.market_avg_price:
                    diff = research.pricing_analysis.target_avg_price - research.pricing_analysis.market_avg_price
                    if diff > 0:
                        comparison = f"Target prices are approximately ${diff:.2f} above market average."
                    elif diff < 0:
                        comparison = f"Target prices are approximately ${abs(diff):.2f} below market average."
                    else:
                        comparison = "Target prices are aligned with market average."
                else:
                    comparison = "Detailed price comparison requires additional menu data."
            else:
                comparison = "Price comparison data not available."

        # Menu differentiation
        differentiation = llm_section.get("menu_differentiation", "")
        if not differentiation:
            if research.menu_comparison and research.menu_comparison.unique_offerings:
                offerings = ", ".join(research.menu_comparison.unique_offerings[:3])
                differentiation = f"Distinctive offerings include: {offerings}"
            else:
                differentiation = "Menu differentiation analysis requires additional data."

        # Pricing recommendations
        recommendations = llm_section.get("pricing_recommendations", [])
        if not recommendations:
            recommendations = [
                "Verify menu pricing against direct competitors",
                "Consider seasonal pricing adjustments",
                "Monitor competitor pricing changes",
            ]

        return MenuPricingSection(
            summary=summary,
            price_position=price_position,
            price_comparison_narrative=comparison,
            menu_differentiation=differentiation,
            pricing_recommendations=recommendations,
        )

    def _build_sentiment_section(
        self, json_data: dict, research: ResearcherOutput
    ) -> SentimentSection:
        """Build customer sentiment section."""
        llm_section = json_data.get("customer_sentiment", {})

        # Overall sentiment rating
        rating = "neutral"
        if research.sentiment_analysis and research.sentiment_analysis.target_overall_sentiment:
            score = research.sentiment_analysis.target_overall_sentiment
            if score >= 0.8:
                rating = "highly positive"
            elif score >= 0.6:
                rating = "positive"
            elif score >= 0.4:
                rating = "mixed"
            else:
                rating = "concerning"

        # Summary
        summary = llm_section.get(
            "summary",
            f"Customer sentiment analysis indicates {rating} perception of the restaurant. "
            "Review analysis reveals consistent themes in customer feedback."
        )

        # Key strengths
        strengths = llm_section.get("key_strengths", [])
        if not strengths and research.sentiment_analysis:
            strengths = research.sentiment_analysis.positive_drivers[:5]

        # Key concerns
        concerns = llm_section.get("key_concerns", [])
        if not concerns and research.sentiment_analysis:
            concerns = research.sentiment_analysis.common_complaints[:5]

        # Reputation risk assessment
        risk_assessment = llm_section.get("reputation_risk_assessment", "")
        if not risk_assessment:
            if rating in ["highly positive", "positive"]:
                risk_assessment = "Low reputation risk. Customer satisfaction levels support stable operations."
            elif rating == "mixed":
                risk_assessment = "Moderate reputation risk. Monitor review trends and address recurring complaints."
            else:
                risk_assessment = "Elevated reputation risk. Negative sentiment may impact revenue stability."

        return SentimentSection(
            summary=summary,
            overall_sentiment_rating=rating,
            key_strengths=strengths if strengths else ["Customer feedback analysis pending"],
            key_concerns=concerns if concerns else ["No significant concerns identified"],
            reputation_risk_assessment=risk_assessment,
        )

    def _build_banking_section(
        self,
        json_data: dict,
        research: ResearcherOutput,
        critique: CriticOutput,
    ) -> BankingRelevanceSection:
        """Build commercial banking relevance section."""
        llm_section = json_data.get("banking_relevance", {})

        # Revenue stability indicators
        indicators = llm_section.get("revenue_stability_indicators", [])
        if not indicators:
            indicators = []
            if research.sentiment_analysis and research.sentiment_analysis.target_overall_sentiment:
                if research.sentiment_analysis.target_overall_sentiment >= 0.6:
                    indicators.append("Positive customer sentiment supports repeat business")
            if research.market_signals:
                if research.market_signals.market_saturation == "low":
                    indicators.append("Low competition reduces revenue volatility risk")
                if research.market_signals.avg_competitor_rating:
                    indicators.append(f"Market average rating: {research.market_signals.avg_competitor_rating:.1f}/5.0")
            if research.competitors:
                indicators.append(f"Established market with {len(research.competitors)} active competitors")

        # Expansion viability
        expansion = llm_section.get("expansion_viability_assessment", "")
        if not expansion:
            if research.market_signals:
                sat = research.market_signals.market_saturation
                if sat == "low":
                    expansion = "Favorable conditions for expansion. Limited competition suggests growth opportunity."
                elif sat == "moderate":
                    expansion = "Selective expansion may be viable. Market can support additional capacity with differentiation."
                else:
                    expansion = "Expansion carries elevated risk. Market saturation limits growth potential."
            else:
                expansion = "Expansion viability requires additional market analysis."

        # Risk considerations
        risks = llm_section.get("risk_considerations", [])
        if not risks:
            risks = []
            if research.market_signals and research.market_signals.risk_indicators:
                risks.extend(research.market_signals.risk_indicators[:3])
            if not risks:
                risks = [
                    "Standard industry risks apply",
                    "Monitor local economic conditions",
                    "Track competitive dynamics",
                ]

        # Loan term factors
        loan_factors = llm_section.get("recommended_loan_terms_factors", [])
        if not loan_factors:
            loan_factors = [
                "Consider seasonal revenue fluctuations",
                "Require updated financial statements",
                "Standard restaurant industry covenants",
            ]

        return BankingRelevanceSection(
            revenue_stability_indicators=indicators if indicators else ["Standard revenue indicators apply"],
            expansion_viability_assessment=expansion,
            risk_considerations=risks,
            collateral_considerations="Standard restaurant equipment and lease considerations",
            recommended_loan_terms_factors=loan_factors,
        )

    def _build_recommendation(
        self,
        json_data: dict,
        research: ResearcherOutput,
        critique: CriticOutput,
    ) -> FinalRecommendation:
        """Build final recommendation section."""
        llm_rec = json_data.get("final_recommendation", {})

        # Determine outlook
        quality = critique.overall_quality_score
        sentiment = 0.5
        if research.sentiment_analysis and research.sentiment_analysis.target_overall_sentiment:
            sentiment = research.sentiment_analysis.target_overall_sentiment

        saturation_score = 0.5
        if research.market_signals and research.market_signals.market_saturation:
            sat_map = {"low": 0.8, "moderate": 0.5, "high": 0.3, "oversaturated": 0.2}
            saturation_score = sat_map.get(research.market_signals.market_saturation, 0.5)

        combined = (quality + sentiment + saturation_score) / 3

        if combined >= 0.7:
            outlook = "moderate"
            confidence = "medium"
        elif combined >= 0.5:
            outlook = "conservative"
            confidence = "medium"
        else:
            outlook = "conservative"
            confidence = "low"

        outlook = llm_rec.get("outlook", outlook)
        confidence = llm_rec.get("confidence_level", confidence)

        # Primary recommendation
        primary = llm_rec.get("primary_recommendation", "")
        if not primary:
            if outlook == "aggressive":
                primary = "Proceed with favorable terms. Market conditions support growth."
            elif outlook == "moderate":
                primary = "Standard underwriting recommended. Market position is stable."
            else:
                primary = "Enhanced due diligence recommended. Conservative approach warranted."

        # Supporting rationale
        rationale = llm_rec.get("supporting_rationale", [])
        if not rationale:
            rationale = []
            if research.competitors:
                rationale.append(f"Competitive analysis covers {len(research.competitors)} market participants")
            if research.sentiment_analysis:
                rationale.append("Customer sentiment data supports stability assessment")
            if critique.overall_quality_score >= 0.7:
                rationale.append("Research quality meets banking standards")

        # Risk mitigations
        mitigations = llm_rec.get("risk_mitigations", [])
        if not mitigations:
            mitigations = [
                "Verify financials with recent tax returns",
                "Confirm lease terms and landlord relationship",
                "Review historical revenue patterns",
            ]

        # Next steps
        next_steps = llm_rec.get("next_steps", [])
        if not next_steps:
            next_steps = [
                "Request detailed financial statements",
                "Schedule site visit",
                "Complete standard underwriting process",
            ]

        return FinalRecommendation(
            outlook=outlook,
            confidence_level=confidence,
            primary_recommendation=primary,
            supporting_rationale=rationale if rationale else ["Based on available market data"],
            risk_mitigations=mitigations,
            next_steps=next_steps,
        )

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        planner_output = state.get("planner_output")
        researcher_output = state.get("researcher_output")
        critic_output = state.get("critic_output")

        if not all([planner_output, researcher_output, critic_output]):
            raise ValueError("Missing required outputs from previous agents")

        # Convert dicts if needed
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

