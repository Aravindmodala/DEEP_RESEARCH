"""Pydantic schemas for all agent outputs with guaranteed structure."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# SOURCE REFERENCE (shared across agents)
# =============================================================================


class SourceReference(BaseModel):
    """Reference to a data source used in research."""

    source_type: str = "unknown"
    url: str | None = None
    title: str | None = None
    accessed_at: str | None = None
    data_summary: str | None = None


# =============================================================================
# PLANNER OUTPUT
# =============================================================================


class PlannerOutput(BaseModel):
    """
    Structured output from the Planner Agent.
    
    Converts user's natural-language request into a research plan.
    """

    target_restaurant: str = "Unknown"
    location: str = "Unknown"
    cuisine_type: str = "unknown"
    intent: str = ""
    search_queries: list[str] = Field(default_factory=list)
    research_focus: list[str] = Field(default_factory=list)
    priority_areas: list[str] = Field(default_factory=list)


# =============================================================================
# RESEARCHER OUTPUT
# =============================================================================


# =============================================================================
# RESEARCHER OUTPUT
# =============================================================================


class BusinessBackground(BaseModel):
    """Business history and expansion details."""
    
    founding_year: str | None = None
    founders: list[str] = Field(default_factory=list)
    history_summary: str = ""
    expansion_strategy: str = ""
    recent_openings: list[str] = Field(default_factory=list)


class RestaurantInfo(BaseModel):
    """Basic restaurant information."""

    name: str = "Unknown"
    address: str = ""
    place_id: str | None = None
    rating: float | None = None
    review_count: int | None = None
    price_level: str | None = None
    cuisine_type: str = "restaurant"
    website: str | None = None
    business_background: BusinessBackground = Field(default_factory=BusinessBackground)


class CompetitorInfo(BaseModel):
    """Competitor restaurant information."""

    name: str = "Unknown"
    address: str = ""
    distance_miles: float = 0.0
    rating: float | None = None
    review_count: int | None = None
    price_level: str | None = None
    cuisine_type: str = "restaurant"
    cuisine_subtype: str | None = None  # e.g. "South Indian" vs "Indian"
    website: str | None = None
    place_id: str | None = None


# =============================================================================
# MENU PARSING SCHEMAS (NEW)
# =============================================================================


class MenuItem(BaseModel):
    """A single menu item with name, price, and category."""
    
    name: str = ""
    price: float | None = None
    category: str = "Other"
    description: str = ""


class ParsedMenu(BaseModel):
    """Structured menu for a restaurant."""
    
    restaurant_name: str = ""
    source_url: str = ""
    items: list[MenuItem] = Field(default_factory=list)
    total_items: int = 0
    categories: list[str] = Field(default_factory=list)


class MenuItemComparison(BaseModel):
    """Comparison of a single item across restaurants."""
    
    item_name: str = ""
    category: str = ""
    target_price: float | None = None
    competitor_prices: dict[str, float | None] = Field(default_factory=dict)  # {competitor_name: price}
    price_difference_avg: float | None = None  # vs avg competitor price
    notes: str = ""


class MenuComparison(BaseModel):
    """Menu comparison data with structured items."""

    target_menu: ParsedMenu | None = None
    competitor_menus: list[ParsedMenu] = Field(default_factory=list)
    item_comparisons: list[MenuItemComparison] = Field(default_factory=list)
    unique_offerings: list[str] = Field(default_factory=list)
    pricing_summary: str = ""


class PricingAnalysis(BaseModel):
    """Pricing analysis data."""

    price_position: str = "unknown"
    target_avg_price: float | None = None
    market_avg_price: float | None = None
    competitor_price_range: dict[str, Any] = Field(default_factory=dict)


class SentimentQuality(BaseModel):
    """Detailed sentiment for a specific aspect (Service, Food, etc)."""
    
    score: float = 0.5  # 0.0 to 1.0
    positive_themes: list[str] = Field(default_factory=list)
    negative_themes: list[str] = Field(default_factory=list)
    summary: str = ""


class RestaurantSentiment(BaseModel):
    """Sentiment analysis for a single restaurant."""

    name: str = ""
    sentiment_score: float = 0.5
    
    # Detailed aspects
    service_quality: SentimentQuality = Field(default_factory=SentimentQuality)
    food_quality: SentimentQuality = Field(default_factory=SentimentQuality)
    atmosphere: SentimentQuality = Field(default_factory=SentimentQuality)
    value_perception: SentimentQuality = Field(default_factory=SentimentQuality)
    
    key_strengths: list[str] = Field(default_factory=list)
    key_concerns: list[str] = Field(default_factory=list)
    summary: str = ""


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""

    restaurants: list[RestaurantSentiment] = Field(default_factory=list)
    comparative_summary: str = ""
    raw_reviews: dict[str, Any] = Field(default_factory=dict)


class MarketSignals(BaseModel):
    """Market signals and indicators."""

    competitor_density: int = 0
    market_saturation: str = "unknown"
    avg_competitor_rating: float | None = None
    foot_traffic_indicators: list[str] = Field(default_factory=list)
    growth_indicators: list[str] = Field(default_factory=list)
    risk_indicators: list[str] = Field(default_factory=list)


class ResearcherOutput(BaseModel):
    """
    Structured output from the Researcher Agent.
    
    Contains all research findings from the autonomous ReAct loop.
    """

    target: RestaurantInfo | None = None
    competitors: list[CompetitorInfo] = Field(default_factory=list)
    menu_comparison: MenuComparison = Field(default_factory=MenuComparison)
    pricing_analysis: PricingAnalysis = Field(default_factory=PricingAnalysis)
    sentiment_analysis: SentimentAnalysis = Field(default_factory=SentimentAnalysis)
    market_signals: MarketSignals = Field(default_factory=MarketSignals)
    raw_sources: list[SourceReference] = Field(default_factory=list)
    research_notes: list[str] = Field(default_factory=list)


# =============================================================================
# CRITIC OUTPUT
# =============================================================================


class ResearchIssue(BaseModel):
    """An issue found during research evaluation."""

    severity: Literal["critical", "major", "minor"] = "minor"
    category: str = "general"
    description: str = ""
    affected_section: str | None = None


class CriticOutput(BaseModel):
    """
    Structured output from the Critic Agent.
    
    Evaluates research quality and determines ACCEPT/REJECT decision.
    """

    decision: Literal["ACCEPT", "REJECT"] = "REJECT"
    overall_quality_score: float = 0.0
    issues_found: list[ResearchIssue] = Field(default_factory=list)
    required_fixes: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    banking_suitability_assessment: str = ""


# =============================================================================
# REPORT OUTPUT (Final output - MUST be validated for UI rendering)
# =============================================================================


class ExecutiveSummary(BaseModel):
    """Executive summary section of the report."""

    overview: str = ""
    key_findings: list[str] = Field(default_factory=list)
    lending_implications: str = ""
    expansion_implications: str = ""


class TopCompetitor(BaseModel):
    """Structured competitor item for report rendering / structured output."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    distance_miles: float | None = None
    rating: float | None = None
    review_count: int | None = None
    price_level: str | None = None
    cuisine_type: str | None = None
    website: str | None = None
    place_id: str | None = None


class CompetitiveLandscapeSection(BaseModel):
    """Competitive landscape section."""

    summary: str = ""
    competitor_count: int = 0
    market_saturation_level: str = "unknown"

    # NOTE: We must avoid `dict[str, Any]` here because OpenAI structured output
    # requires JSON schemas with `additionalProperties: false` for object items.
    # Using a concrete Pydantic model with `extra="forbid"` produces a valid schema.
    top_competitors: list[TopCompetitor] = Field(default_factory=list)
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
    
    This model is rendered into a document/UI and needs guaranteed structure.
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
