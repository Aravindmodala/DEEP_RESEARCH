"""Pydantic models for all agent outputs with validated structure."""

from .schemas import (
    # Shared
    SourceReference,
    # Planner
    PlannerOutput,
    # Researcher
    RestaurantInfo,
    CompetitorInfo,
    MenuComparison,
    PricingAnalysis,
    RestaurantSentiment,
    SentimentAnalysis,
    MarketSignals,
    ResearcherOutput,
    # Critic
    ResearchIssue,
    CriticOutput,
    # Report
    ExecutiveSummary,
    CompetitiveLandscapeSection,
    MenuPricingSection,
    SentimentSection,
    BankingRelevanceSection,
    FinalRecommendation,
    ReportOutput,
)

__all__ = [
    # Shared
    "SourceReference",
    # Planner
    "PlannerOutput",
    # Researcher
    "RestaurantInfo",
    "CompetitorInfo",
    "MenuComparison",
    "PricingAnalysis",
    "RestaurantSentiment",
    "SentimentAnalysis",
    "MarketSignals",
    "ResearcherOutput",
    # Critic
    "ResearchIssue",
    "CriticOutput",
    # Report
    "ExecutiveSummary",
    "CompetitiveLandscapeSection",
    "MenuPricingSection",
    "SentimentSection",
    "BankingRelevanceSection",
    "FinalRecommendation",
    "ReportOutput",
]
