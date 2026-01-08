"""Pydantic schemas for all agent node inputs/outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# NODE 1 — PLANNER OUTPUT
# =============================================================================


class PlannerOutput(BaseModel):
    """Structured output from the Planner Agent."""

    target_restaurant: str = Field(
        ..., description="Name of the target restaurant to analyze"
    )
    location: str = Field(
        ..., description="City, State of the restaurant location"
    )
    cuisine_type: str = Field(
        default="unknown",
        description="Cuisine type detected from web search (e.g., Indian, Italian, American)",
    )
    intent: str = Field(
        ...,
        description="Inferred user intent (e.g., lending risk, expansion viability)",
    )
    search_queries: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Exactly 4 research queries covering: competitive landscape, menu/pricing, sentiment, market demand",
    )


# =============================================================================
# NODE 2 — RESEARCHER OUTPUT
# =============================================================================


class Competitor(BaseModel):
    """Individual competitor restaurant data."""

    name: str = Field(..., description="Competitor restaurant name")
    address: str = Field(..., description="Full address")
    distance_miles: float = Field(..., description="Distance from target in miles")
    rating: float | None = Field(None, description="Google rating (1-5)")
    review_count: int | None = Field(None, description="Total number of reviews")
    price_level: str | None = Field(
        None, description="Price level: $, $$, $$$, $$$$"
    )
    cuisine_type: str = Field(..., description="Primary cuisine category")
    website: str | None = Field(None, description="Restaurant website URL")
    place_id: str | None = Field(None, description="Google Place ID for reference")


class MenuItem(BaseModel):
    """Individual menu item."""

    name: str = Field(..., description="Item name")
    price: float | None = Field(None, description="Price in USD")
    category: str = Field(..., description="Menu category (appetizers, mains, etc.)")
    description: str | None = Field(None, description="Item description if available")


class RestaurantMenu(BaseModel):
    """Menu data for a single restaurant."""

    restaurant_name: str
    items: list[MenuItem] = Field(default_factory=list)
    avg_appetizer_price: float | None = None
    avg_entree_price: float | None = None
    avg_dessert_price: float | None = None
    signature_items: list[str] = Field(default_factory=list)


class MenuComparison(BaseModel):
    """Comparative menu analysis across target and competitors."""

    target_menu: RestaurantMenu | None = None
    competitor_menus: list[RestaurantMenu] = Field(default_factory=list)
    menu_breadth_comparison: dict[str, int] = Field(
        default_factory=dict,
        description="Restaurant name -> total menu item count",
    )
    unique_offerings: list[str] = Field(
        default_factory=list,
        description="Unique items or categories the target offers",
    )


class PricingAnalysis(BaseModel):
    """Pricing position analysis."""

    target_avg_price: float | None = Field(
        None, description="Target restaurant average item price"
    )
    market_avg_price: float | None = Field(
        None, description="Market average across competitors"
    )
    price_position: Literal["budget", "mid-range", "premium", "luxury"] | None = None
    price_percentile: int | None = Field(
        None, description="Target's percentile in local market (0-100)"
    )
    competitor_price_range: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Competitor name -> price level info",
    )


class SentimentBreakdown(BaseModel):
    """Sentiment breakdown by category."""

    food_quality: float | None = Field(None, description="Sentiment score 0-1")
    service: float | None = Field(None, description="Sentiment score 0-1")
    ambiance: float | None = Field(None, description="Sentiment score 0-1")
    value_for_money: float | None = Field(None, description="Sentiment score 0-1")
    cleanliness: float | None = Field(None, description="Sentiment score 0-1")


class SentimentAnalysis(BaseModel):
    """Customer sentiment analysis from reviews."""

    target_overall_sentiment: float | None = Field(
        None, description="Overall sentiment score 0-1"
    )
    target_sentiment_breakdown: SentimentBreakdown | None = None
    positive_drivers: list[str] = Field(
        default_factory=list, description="Key positive themes from reviews"
    )
    common_complaints: list[str] = Field(
        default_factory=list, description="Recurring negative themes"
    )
    review_volume_trend: Literal["increasing", "stable", "decreasing"] | None = None
    sample_reviews: list[dict[str, Any]] = Field(
        default_factory=list, description="Representative review samples"
    )


class MarketSignals(BaseModel):
    """Market intelligence signals for banking decisions."""

    competitor_density: int = Field(
        ..., description="Number of direct competitors in radius"
    )
    market_saturation: Literal["low", "moderate", "high", "oversaturated"] | None = None
    avg_competitor_rating: float | None = None
    foot_traffic_indicators: list[str] = Field(
        default_factory=list, description="Signals of demand/traffic"
    )
    growth_indicators: list[str] = Field(
        default_factory=list, description="Market growth signals"
    )
    risk_indicators: list[str] = Field(
        default_factory=list, description="Market risk signals"
    )
    local_economic_context: str | None = Field(
        None, description="Brief local economic context"
    )


class SourceReference(BaseModel):
    """Reference to a data source used in research."""

    source_type: Literal["google_maps", "tavily", "website_scrape", "api"]
    url: str | None = None
    title: str | None = None
    accessed_at: str | None = None
    data_summary: str | None = None


class ResearcherOutput(BaseModel):
    """Structured output from the Researcher Agent."""

    competitors: list[Competitor] = Field(default_factory=list)
    menu_comparison: MenuComparison | None = None
    pricing_analysis: PricingAnalysis | None = None
    sentiment_analysis: SentimentAnalysis | None = None
    market_signals: MarketSignals | None = None
    raw_sources: list[SourceReference] = Field(default_factory=list)
    research_notes: list[str] = Field(
        default_factory=list, description="Researcher's observations and notes"
    )


# =============================================================================
# NODE 3 — CRITIC OUTPUT
# =============================================================================


class CriticIssue(BaseModel):
    """Individual issue found by the Critic."""

    severity: Literal["critical", "major", "minor"]
    category: Literal[
        "completeness",
        "accuracy",
        "grounding",
        "logic",
        "banking_relevance",
    ]
    description: str
    affected_section: str | None = None


class CriticOutput(BaseModel):
    """Structured output from the Critic Agent."""

    decision: Literal["ACCEPT", "REJECT"] = Field(
        ..., description="Whether research passes quality review"
    )
    overall_quality_score: float = Field(
        ..., ge=0, le=1, description="Quality score 0-1"
    )
    issues_found: list[CriticIssue] = Field(default_factory=list)
    required_fixes: list[str] = Field(
        default_factory=list, description="Specific actions needed if REJECT"
    )
    strengths: list[str] = Field(
        default_factory=list, description="Research strengths noted"
    )
    banking_suitability_assessment: str | None = Field(
        None, description="Assessment of suitability for banking decisions"
    )


# =============================================================================
# NODE 4 — REPORT OUTPUT
# =============================================================================


class ExecutiveSummary(BaseModel):
    """Executive summary section of the report."""

    overview: str
    key_findings: list[str]
    lending_implications: str
    expansion_implications: str


class CompetitiveLandscapeSection(BaseModel):
    """Competitive landscape section."""

    summary: str
    competitor_count: int
    market_saturation_level: str
    top_competitors: list[dict[str, Any]]
    competitive_advantages: list[str]
    competitive_disadvantages: list[str]


class MenuPricingSection(BaseModel):
    """Menu and pricing section."""

    summary: str
    price_position: str
    price_comparison_narrative: str
    menu_differentiation: str
    pricing_recommendations: list[str]


class SentimentSection(BaseModel):
    """Customer sentiment section."""

    summary: str
    overall_sentiment_rating: str
    key_strengths: list[str]
    key_concerns: list[str]
    reputation_risk_assessment: str


class BankingRelevanceSection(BaseModel):
    """Commercial banking relevance section."""

    revenue_stability_indicators: list[str]
    expansion_viability_assessment: str
    risk_considerations: list[str]
    collateral_considerations: str | None = None
    recommended_loan_terms_factors: list[str] = Field(default_factory=list)


class FinalRecommendation(BaseModel):
    """Final recommendation section."""

    outlook: Literal["conservative", "moderate", "aggressive"]
    confidence_level: Literal["low", "medium", "high"]
    primary_recommendation: str
    supporting_rationale: list[str]
    risk_mitigations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class ReportOutput(BaseModel):
    """Final structured report from the Report Agent."""

    report_title: str
    generated_at: str
    target_restaurant: str
    location: str
    executive_summary: ExecutiveSummary
    competitive_landscape: CompetitiveLandscapeSection
    menu_pricing: MenuPricingSection
    customer_sentiment: SentimentSection
    banking_relevance: BankingRelevanceSection
    final_recommendation: FinalRecommendation
    appendix_sources: list[SourceReference] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)


# =============================================================================
# AGENT STATE (for LangGraph orchestration)
# =============================================================================


class AgentState(BaseModel):
    """Shared state passed between agent nodes."""

    # Input
    user_query: str = Field(..., description="Original user query")

    # Node outputs
    planner_output: PlannerOutput | None = None
    researcher_output: ResearcherOutput | None = None
    critic_output: CriticOutput | None = None
    report_output: ReportOutput | None = None

    # Control flow
    iteration_count: int = Field(default=0, description="Research iteration count")
    max_iterations: int = Field(default=3, description="Max research attempts")
    current_node: str = Field(default="planner", description="Current active node")
    error_log: list[str] = Field(default_factory=list)

    # Metadata
    session_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None

