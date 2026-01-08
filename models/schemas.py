"""Pydantic schemas - Only for final report output that needs guaranteed structure."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# SOURCE REFERENCE (used in report appendix)
# =============================================================================


class SourceReference(BaseModel):
    """Reference to a data source used in research."""

    source_type: str = "unknown"
    url: str | None = None
    title: str | None = None
    accessed_at: str | None = None
    data_summary: str | None = None


# =============================================================================
# REPORT OUTPUT (Final output - MUST be validated for UI rendering)
# =============================================================================


class ExecutiveSummary(BaseModel):
    """Executive summary section of the report."""

    overview: str = ""
    key_findings: list[str] = Field(default_factory=list)
    lending_implications: str = ""
    expansion_implications: str = ""


class CompetitiveLandscapeSection(BaseModel):
    """Competitive landscape section."""

    summary: str = ""
    competitor_count: int = 0
    market_saturation_level: str = "unknown"
    top_competitors: list[dict[str, Any]] = Field(default_factory=list)
    competitive_advantages: list[str] = Field(default_factory=list)
    competitive_disadvantages: list[str] = Field(default_factory=list)


class MenuPricingSection(BaseModel):
    """Menu and pricing section."""

    summary: str = ""
    price_position: str = "unknown"
    price_comparison_narrative: str = ""
    menu_differentiation: str = ""
    pricing_recommendations: list[str] = Field(default_factory=list)


class SentimentSection(BaseModel):
    """Customer sentiment section."""

    summary: str = ""
    overall_sentiment_rating: str = "unknown"
    key_strengths: list[str] = Field(default_factory=list)
    key_concerns: list[str] = Field(default_factory=list)
    reputation_risk_assessment: str = ""


class BankingRelevanceSection(BaseModel):
    """Commercial banking relevance section."""

    revenue_stability_indicators: list[str] = Field(default_factory=list)
    expansion_viability_assessment: str = ""
    risk_considerations: list[str] = Field(default_factory=list)
    collateral_considerations: str | None = None
    recommended_loan_terms_factors: list[str] = Field(default_factory=list)


class FinalRecommendation(BaseModel):
    """Final recommendation section."""

    outlook: Literal["conservative", "moderate", "aggressive"] = "conservative"
    confidence_level: Literal["low", "medium", "high"] = "medium"
    primary_recommendation: str = ""
    supporting_rationale: list[str] = Field(default_factory=list)
    risk_mitigations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class ReportOutput(BaseModel):
    """
    Final structured report from the Report Agent.
    
    This is the ONLY model that needs strict validation because:
    1. It gets rendered into a document/UI
    2. It's the final deliverable to the user
    3. It needs guaranteed structure for downstream processing
    """

    report_title: str = ""
    generated_at: str = ""
    target_restaurant: str = ""
    location: str = ""
    executive_summary: ExecutiveSummary = Field(default_factory=ExecutiveSummary)
    competitive_landscape: CompetitiveLandscapeSection = Field(default_factory=CompetitiveLandscapeSection)
    menu_pricing: MenuPricingSection = Field(default_factory=MenuPricingSection)
    customer_sentiment: SentimentSection = Field(default_factory=SentimentSection)
    banking_relevance: BankingRelevanceSection = Field(default_factory=BankingRelevanceSection)
    final_recommendation: FinalRecommendation = Field(default_factory=FinalRecommendation)
    appendix_sources: list[SourceReference] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
