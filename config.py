"""Configuration for the Deep Research Agent system."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


# Default report sections (all 13)
DEFAULT_REPORT_SECTIONS = [
    "executive_summary",
    "history_founding",
    "dining_concept_menu",
    "allergy_dietary",
    "expansion_locations",
    "leadership_ownership",
    "community_engagement",
    "awards_recognition",
    "comparison_analysis",
    "sentiment_analysis",
    "menu_comparison",
    "operational_analysis",
    "performance_strategy",
]


class AgentConfig(BaseSettings):
    """Configuration for all agent nodes."""

    # =========================================================================
    # LLM Settings
    # =========================================================================
    llm_provider: Literal["openai", "anthropic", "vertex", "gemini", "ollama"] = Field(
        default="openai",
        description="LLM provider to use: openai, anthropic, vertex, gemini, ollama",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model name for the LLM (e.g., gpt-4o, claude-3-sonnet, gemini-1.5-pro)",
    )
    llm_temperature: float = Field(
        default=0.1,
        description="Temperature for LLM generation",
    )

    # =========================================================================
    # Provider API Keys
    # =========================================================================

    # OpenAI
    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
    )

    # Anthropic
    anthropic_api_key: str | None = Field(
        default=None,
        alias="ANTHROPIC_API_KEY",
    )

    # Google Vertex AI / Gemini
    gcp_project_id: str | None = Field(
        default=None,
        alias="GCP_PROJECT_ID",
        description="Google Cloud project ID for Vertex AI",
    )
    gcp_location: str = Field(
        default="us-central1",
        alias="GCP_LOCATION",
        description="Google Cloud region for Vertex AI",
    )

    # Ollama (Local)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
        description="Base URL for Ollama server",
    )

    # =========================================================================
    # Research Tool API Keys
    # =========================================================================
    google_maps_api_key: str | None = Field(
        default=None,
        alias="GOOGLE_MAPS_API_KEY",
    )
    tavily_api_key: str | None = Field(
        default=None,
        alias="TAVILY_API_KEY",
    )

    # =========================================================================
    # Research Settings
    # =========================================================================
    max_research_iterations: int = Field(
        default=3,
        description="Maximum research loop iterations before forcing accept",
    )
    competitor_search_radius_miles: float = Field(
        default=2.0,
        description="Default radius for competitor search",
    )
    max_competitors: int = Field(
        default=15,
        description="Maximum number of competitors to analyze",
    )
    max_menu_items_per_restaurant: int = Field(
        default=50,
        description="Maximum menu items to extract per restaurant",
    )

    # =========================================================================
    # Phase-Gated Researcher Settings
    # =========================================================================
    max_react_iterations_per_phase: int = Field(
        default=8,
        description="Maximum ReAct iterations per research phase",
    )
    max_total_react_iterations: int = Field(
        default=35,
        description="Hard cap on total ReAct iterations across all phases",
    )

    # =========================================================================
    # Analyst Settings
    # =========================================================================
    analyst_max_llm_calls: int = Field(
        default=15,
        description="Maximum LLM calls for the Analyst agent",
    )

    # =========================================================================
    # Report Settings
    # =========================================================================
    report_sections_enabled: list[str] = Field(
        default_factory=lambda: DEFAULT_REPORT_SECTIONS.copy(),
        description="List of report section keys to generate",
    )

    # =========================================================================
    # Quality Thresholds
    # =========================================================================
    min_quality_score: float = Field(
        default=0.7,
        description="Minimum quality score for Critic acceptance",
    )
    require_menu_data: bool = Field(
        default=True,
        description="Whether menu data is required for acceptance",
    )
    require_sentiment_data: bool = Field(
        default=True,
        description="Whether sentiment data is required for acceptance",
    )

    # =========================================================================
    # Rate Limiting
    # =========================================================================
    google_maps_rate_limit: int = Field(
        default=10,
        description="Max Google Maps API calls per second",
    )
    tavily_rate_limit: int = Field(
        default=5,
        description="Max Tavily API calls per second",
    )

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_to_file: bool = Field(
        default=True,
        description="Whether to log to file",
    )
    log_file_path: str = Field(
        default="logs/agent.log",
        description="Path to log file",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_config() -> AgentConfig:
    """Get the agent configuration, loading from environment."""
    return AgentConfig()
