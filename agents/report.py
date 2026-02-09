"""REPORT Agent - Section-by-section market research report generation."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import AgentConfig
from prompts.report import REPORT_BASE_INSTRUCTIONS, REPORT_SECTION_PROMPTS
from utils import create_llm


# Section display titles (in order)
SECTION_TITLES = {
    "executive_summary": "Executive Summary",
    "history_founding": "History and Founding",
    "dining_concept_menu": "Dining Concept and Menu",
    "allergy_dietary": "Allergy-Friendly / Dietary Commitment",
    "expansion_locations": "Expansion and Locations",
    "leadership_ownership": "Leadership and Ownership",
    "community_engagement": "Community Engagement",
    "awards_recognition": "Awards and Recognition",
    "comparison_analysis": "Comparison Analysis",
    "sentiment_analysis": "Sentiment Analysis",
    "menu_comparison": "Menu Comparison",
    "operational_analysis": "Operational Analysis",
    "performance_strategy": "Performance Strategy",
}


class ReportAgent:
    """
    NODE 5 — REPORT AGENT (Section-by-Section)

    Generates a client-ready Market Research Report using 13 separate
    LLM calls, one per section. Includes inline citations.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config, temperature=0.3)
        self.enabled_sections = config.report_sections_enabled

    def generate(
        self,
        plan: dict[str, Any],
        research: dict[str, Any],
        critique: dict[str, Any],
        analyst: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a comprehensive market research report section by section.

        Args:
            plan: Research plan dict from Planner
            research: Research findings dict from Researcher
            critique: Quality assessment dict from Critic
            analyst: Analysis results dict from Analyst

        Returns:
            Markdown report text with all sections and source appendix
        """
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")

        logger.info(f"[REPORT] Generating section-by-section report for {target} in {location}")

        # Build source registry for citations
        source_registry = self._build_source_registry(research)
        citation_instructions = self._build_citation_instructions(source_registry)

        # Prepare data context
        data_ctx = self._prepare_data_context(plan, research, analyst)

        # Generate each section
        sections = []
        report_header = f"# Market Research Report: {target}\n\n"
        report_header += f"**Location:** {location}\n"
        report_header += f"**Cuisine:** {plan.get('cuisine_type', 'N/A')}\n"
        report_header += f"**Generated:** {datetime.now().strftime('%B %d, %Y')}\n"
        report_header += f"**Quality Score:** {critique.get('overall_quality_score', 0):.2f}/1.00\n\n"
        report_header += "---\n\n"

        sections.append(report_header)

        section_number = 1
        for section_key in self.enabled_sections:
            if section_key not in REPORT_SECTION_PROMPTS:
                logger.warning(f"[REPORT] Unknown section key: {section_key}. Skipping.")
                continue

            title = SECTION_TITLES.get(section_key, section_key.replace("_", " ").title())
            logger.info(f"[REPORT] Generating section {section_number}: {title}")

            try:
                section_content = self._generate_section(
                    section_key, target, location, data_ctx, citation_instructions
                )
                sections.append(f"## {section_number}. {title}\n\n{section_content}\n\n---\n\n")
            except Exception as e:
                logger.error(f"[REPORT] Error generating section '{section_key}': {e}")
                sections.append(f"## {section_number}. {title}\n\n*Section could not be generated: {str(e)}*\n\n---\n\n")

            section_number += 1

        # Append Sources appendix
        sources_appendix = self._build_sources_appendix(source_registry)
        sections.append(sources_appendix)

        # Append disclaimers
        sections.append(self._build_disclaimers())

        report = "".join(sections)
        logger.info(f"[REPORT] Report generation complete. {section_number - 1} sections generated.")
        return report

    def _generate_section(
        self,
        section_key: str,
        target: str,
        location: str,
        data_ctx: dict[str, str],
        citation_instructions: str,
    ) -> str:
        """Generate a single section using an LLM call."""
        prompt_template = REPORT_SECTION_PROMPTS[section_key]

        # Build format kwargs from data context
        format_kwargs = {
            "target": target,
            "location": location,
            "citation_instructions": citation_instructions,
        }

        # Add all data context values
        for key, value in data_ctx.items():
            format_kwargs[key] = value

        # Format the prompt (use safe formatting)
        try:
            prompt = prompt_template.format(**format_kwargs)
        except KeyError as e:
            # If a key is missing, use a placeholder
            logger.warning(f"[REPORT] Missing format key {e} for section {section_key}")
            prompt = prompt_template
            for k, v in format_kwargs.items():
                prompt = prompt.replace(f"{{{k}}}", v)

        messages = [
            SystemMessage(content=REPORT_BASE_INSTRUCTIONS),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def _prepare_data_context(
        self,
        plan: dict[str, Any],
        research: dict[str, Any],
        analyst: dict[str, Any] | None,
    ) -> dict[str, str]:
        """Prepare all data context values for section prompts."""
        target_info = research.get("target") or {}
        competitors = research.get("competitors", [])
        raw_reviews = research.get("raw_reviews", {})
        raw_menus = research.get("raw_menus", [])
        business_intel = research.get("business_intel", {})
        market_data = research.get("market_data", [])

        # Analyst data (if available)
        swot_analyses = []
        sentiment_data = {}
        parsed_menus = []
        menu_comparison = {}
        keyword_data = {}
        metrics_data = []
        strategy_data = {}
        positioning_data = {}

        if analyst:
            swot_analyses = analyst.get("swot_analyses", [])
            sentiment_data = analyst.get("sentiment_analysis", {})
            parsed_menus = analyst.get("parsed_menus", [])
            menu_comparison = analyst.get("menu_comparison", {})
            keyword_data = analyst.get("keyword_analysis", {})
            metrics_data = analyst.get("performance_metrics", [])
            strategy_data = analyst.get("strategic_recommendations", {})
            positioning_data = analyst.get("competitive_positioning", {})

        # Business background
        bg = target_info.get("business_background", {})
        business_background = json.dumps(bg, default=str, indent=2)[:3000] if bg else "No business background data available."

        # Competitors summary
        comp_summary = []
        for c in competitors[:10]:
            comp_summary.append({
                "name": c.get("name"),
                "rating": c.get("rating"),
                "distance": c.get("distance_miles"),
                "price_level": c.get("price_level"),
            })
        competitors_summary = json.dumps(comp_summary, default=str)[:3000]

        # Sentiment summary
        sentiment_restaurants = sentiment_data.get("restaurants", []) if isinstance(sentiment_data, dict) else []
        sentiment_summary = json.dumps(sentiment_restaurants[:4], default=str)[:5000] if sentiment_restaurants else "No sentiment data available."

        # Market position
        market_position = json.dumps(positioning_data, default=str)[:2000] if positioning_data else "No positioning data available."

        # Business intel
        business_intel_str = json.dumps(business_intel, default=str)[:5000] if business_intel else "No business intelligence data available."

        # Target menu
        target_menu_data = next(
            (m for m in parsed_menus if isinstance(m, dict)),
            None
        )
        target_menu = json.dumps(target_menu_data, default=str)[:8000] if target_menu_data else "No menu data available."

        # Review summary
        review_texts = []
        for name, data in raw_reviews.items():
            reviews = data.get("reviews", []) if isinstance(data, dict) else []
            for r in reviews[:3]:
                text = r.get("text", "")[:200] if isinstance(r, dict) else str(r)[:200]
                review_texts.append(f"[{name}]: {text}")
        review_summary = "\n".join(review_texts[:15]) if review_texts else "No review data available."

        return {
            "cuisine_type": plan.get("cuisine_type", "N/A"),
            "business_background": business_background,
            "competitors_summary": competitors_summary,
            "competitors_data": json.dumps(competitors[:10], default=str)[:5000],
            "sentiment_summary": sentiment_summary,
            "sentiment_data": json.dumps(sentiment_data, default=str)[:8000] if sentiment_data else "No sentiment data.",
            "market_position": market_position,
            "business_intel": business_intel_str,
            "target_menu": target_menu,
            "unique_offerings": json.dumps(menu_comparison.get("unique_offerings", []), default=str) if isinstance(menu_comparison, dict) else "[]",
            "review_summary": review_summary,
            "swot_data": json.dumps(swot_analyses, default=str)[:8000] if swot_analyses else "No SWOT data available.",
            "swot_summary": json.dumps(swot_analyses[:2], default=str)[:3000] if swot_analyses else "No SWOT data.",
            "positioning_data": json.dumps(positioning_data, default=str)[:3000] if positioning_data else "No positioning data.",
            "keyword_data": json.dumps(keyword_data, default=str)[:5000] if keyword_data else "No keyword data available.",
            "menu_comparison_data": json.dumps(menu_comparison, default=str)[:8000] if menu_comparison else "No menu comparison data.",
            "parsed_menus_summary": json.dumps([{"name": m.get("restaurant_name"), "items": m.get("total_items")} for m in parsed_menus if isinstance(m, dict)], default=str) if parsed_menus else "No parsed menus.",
            "metrics_data": json.dumps(metrics_data, default=str)[:5000] if metrics_data else "No metrics data available.",
            "market_data": json.dumps(market_data[:3], default=str)[:3000] if market_data else "No market data available.",
            "strategy_data": json.dumps(strategy_data, default=str)[:8000] if strategy_data else "No strategic data available.",
        }

    def _build_source_registry(self, research: dict[str, Any]) -> list[dict[str, str]]:
        """Build a numbered source registry from raw_sources."""
        sources = research.get("raw_sources", [])
        registry = []
        seen_urls = set()

        for s in sources:
            url = s.get("url", "API")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            registry.append({
                "number": len(registry) + 1,
                "type": s.get("source_type", "unknown"),
                "title": s.get("title", ""),
                "url": url,
                "accessed": s.get("accessed_at", ""),
            })

        return registry

    def _build_citation_instructions(self, source_registry: list[dict]) -> str:
        """Build citation instructions with the source registry."""
        if not source_registry:
            return "No sources available for citation."

        source_list = "\n".join([
            f"[Source {s['number']}]: {s['title']} ({s['url']})"
            for s in source_registry[:30]  # Cap at 30
        ])

        return f"""CITATION INSTRUCTIONS:
Use inline citations [Source N] to reference data. Available sources:
{source_list}

When making factual claims, include the relevant [Source N] citation."""

    def _build_sources_appendix(self, source_registry: list[dict]) -> str:
        """Build the Sources appendix section."""
        if not source_registry:
            return "## Sources\n\nNo sources recorded.\n\n"

        lines = ["## Sources\n\n"]
        for s in source_registry:
            lines.append(f"**[Source {s['number']}]** {s['title']}  \n")
            lines.append(f"Type: {s['type']} | URL: {s['url']}  \n")
            if s.get("accessed"):
                lines.append(f"Accessed: {s['accessed']}  \n")
            lines.append("\n")

        return "".join(lines)

    def _build_disclaimers(self) -> str:
        """Build standard disclaimers section."""
        return """## Disclaimers

- This report is generated using AI-powered research tools and should be used as a supplementary resource, not as the sole basis for financial decisions.
- Data is sourced from publicly available information including Google Maps, review platforms, and web searches. It may not reflect the most current state of the business.
- Menu prices and competitor data are subject to change and should be verified independently.
- Sentiment analysis is based on a sample of publicly available reviews and may not represent the full customer experience.
- This report does not constitute financial advice. Professional due diligence is recommended before making lending or investment decisions.
"""

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        planner_output = state.get("planner_output")
        researcher_output = state.get("researcher_output")
        critic_output = state.get("critic_output")
        analyst_output = state.get("analyst_output")

        if not all([planner_output, researcher_output, critic_output]):
            raise ValueError("Missing required outputs from previous agents")

        try:
            report_text = self.generate(
                planner_output, researcher_output, critic_output, analyst_output
            )
            return {
                **state,
                "report_output": report_text,
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
