"""ANALYST Agent - Multi-pass LLM analysis of raw research data."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import AgentConfig
from models.schemas import (
    AnalystOutput,
    CompetitivePositioning,
    MenuComparison,
    MenuItemComparison,
    ParsedMenu,
    MenuItem,
    PerformanceMetrics,
    ReviewKeywordAnalysis,
    RestaurantSentiment,
    SentimentAnalysis,
    SentimentQuality,
    StrategicPillar,
    StrategicRecommendations,
    SWOTAnalysis,
)
from prompts.analyst import (
    ANALYST_SYSTEM_PROMPT,
    KEYWORD_ANALYSIS_PROMPT,
    MENU_COMPARISON_PROMPT,
    MENU_PARSING_PROMPT,
    PERFORMANCE_METRICS_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
    STRATEGIC_RECOMMENDATIONS_PROMPT,
    SWOT_ANALYSIS_PROMPT,
)
from utils import create_llm


class AnalystAgent:
    """
    NODE 3 — ANALYST AGENT (NEW)

    Takes raw ResearcherOutput and performs 7 multi-pass LLM analyses:
    1. SWOT Analysis (per restaurant)
    2. Sentiment Analysis (per restaurant)
    3. Menu Parsing (per menu)
    4. Menu Comparison
    5. Review Keyword Analysis
    6. Performance Metrics
    7. Strategic Recommendations
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config, temperature=0.2)
        self.max_calls = config.analyst_max_llm_calls

    def analyze(self, research: dict[str, Any]) -> AnalystOutput:
        """
        Run all 7 analysis passes on the raw research data.

        Args:
            research: Raw research output dict from Researcher

        Returns:
            AnalystOutput with all analysis products
        """
        logger.info("[ANALYST] Starting multi-pass analysis...")
        call_count = 0

        target = research.get("target") or {}
        competitors = research.get("competitors", [])
        raw_reviews = research.get("raw_reviews", {})
        supplementary_reviews = research.get("supplementary_reviews", {})
        raw_menus = research.get("raw_menus", [])
        business_intel = research.get("business_intel", {})
        market_data = research.get("market_data", [])

        target_name = target.get("name", "Unknown")

        # 1. SWOT Analysis (1 call per restaurant)
        logger.info("[ANALYST] Pass 1: SWOT Analysis")
        swot_analyses = []

        # SWOT for target
        if call_count < self.max_calls:
            swot = self._analyze_swot(target_name, target, raw_reviews.get(target_name, {}), market_data)
            if swot:
                swot_analyses.append(swot)
            call_count += 1

        # SWOT for top competitors
        for comp in competitors[:3]:
            if call_count >= self.max_calls:
                break
            comp_name = comp.get("name", "Unknown")
            comp_reviews = raw_reviews.get(comp_name, {})
            swot = self._analyze_swot(comp_name, comp, comp_reviews, market_data)
            if swot:
                swot_analyses.append(swot)
            call_count += 1

        # 2. Sentiment Analysis (1 call per restaurant)
        logger.info("[ANALYST] Pass 2: Sentiment Analysis")
        sentiment_restaurants = []

        # Sentiment for target
        if call_count < self.max_calls:
            sentiment = self._analyze_sentiment(
                target_name, target, raw_reviews.get(target_name, {}),
                supplementary_reviews.get(target_name, {})
            )
            if sentiment:
                sentiment_restaurants.append(sentiment)
            call_count += 1

        # Sentiment for competitors
        for comp in competitors[:3]:
            if call_count >= self.max_calls:
                break
            comp_name = comp.get("name", "Unknown")
            sentiment = self._analyze_sentiment(
                comp_name, comp, raw_reviews.get(comp_name, {}),
                supplementary_reviews.get(comp_name, {})
            )
            if sentiment:
                sentiment_restaurants.append(sentiment)
            call_count += 1

        sentiment_analysis = SentimentAnalysis(
            restaurants=sentiment_restaurants,
            comparative_summary=f"Analyzed sentiment for {len(sentiment_restaurants)} restaurants.",
            raw_reviews=raw_reviews,
        )

        # 3. Menu Parsing (1 call per menu)
        logger.info("[ANALYST] Pass 3: Menu Parsing")
        parsed_menus = []
        for menu_data in raw_menus:
            if call_count >= self.max_calls:
                break
            parsed = self._parse_menu(menu_data)
            if parsed:
                parsed_menus.append(parsed)
            call_count += 1

        # 4. Menu Comparison (1 call)
        logger.info("[ANALYST] Pass 4: Menu Comparison")
        menu_comparison = MenuComparison(pricing_summary="Insufficient menu data.")
        if call_count < self.max_calls and len(parsed_menus) >= 2:
            menu_comparison = self._compare_menus(target_name, parsed_menus)
            call_count += 1

        # 5. Keyword Analysis (1 call)
        logger.info("[ANALYST] Pass 5: Review Keyword Analysis")
        keyword_analysis = ReviewKeywordAnalysis()
        if call_count < self.max_calls and raw_reviews:
            keyword_analysis = self._analyze_keywords(raw_reviews, supplementary_reviews)
            call_count += 1

        # 6. Performance Metrics (1 call)
        logger.info("[ANALYST] Pass 6: Performance Metrics")
        performance_metrics = []
        if call_count < self.max_calls:
            performance_metrics = self._calculate_metrics(
                target, competitors, sentiment_restaurants
            )
            call_count += 1

        # 7. Strategic Recommendations (1 call)
        logger.info("[ANALYST] Pass 7: Strategic Recommendations")
        strategic_recs = StrategicRecommendations()
        if call_count < self.max_calls:
            strategic_recs = self._generate_recommendations(
                swot_analyses, sentiment_analysis, menu_comparison,
                performance_metrics, market_data
            )
            call_count += 1

        # Build competitive positioning from SWOT data
        competitive_positioning = self._build_positioning(target_name, swot_analyses, competitors)

        output = AnalystOutput(
            swot_analyses=swot_analyses,
            sentiment_analysis=sentiment_analysis,
            parsed_menus=parsed_menus,
            menu_comparison=menu_comparison,
            keyword_analysis=keyword_analysis,
            performance_metrics=performance_metrics,
            strategic_recommendations=strategic_recs,
            competitive_positioning=competitive_positioning,
        )

        logger.info(f"[ANALYST] Analysis complete. LLM calls used: {call_count}/{self.max_calls}")
        return output

    # --------------------------------------------------------------------------
    # ANALYSIS METHODS
    # --------------------------------------------------------------------------

    def _analyze_swot(
        self, name: str, info: dict, reviews: dict, market_data: list
    ) -> SWOTAnalysis | None:
        """SWOT analysis for a single restaurant."""
        try:
            review_text = self._format_reviews(reviews)
            market_text = json.dumps(market_data[:3], default=str)[:3000]

            prompt = SWOT_ANALYSIS_PROMPT.format(
                restaurant_name=name,
                restaurant_info=json.dumps(info, default=str)[:3000],
                reviews=review_text[:5000],
                market_context=market_text,
            )

            data = self._llm_json_call(prompt)
            if data:
                return SWOTAnalysis(**data)
        except Exception as e:
            logger.error(f"[ANALYST] SWOT analysis failed for {name}: {e}")
        return None

    def _analyze_sentiment(
        self, name: str, info: dict, google_reviews: dict, supp_reviews: Any
    ) -> RestaurantSentiment | None:
        """Multi-dimensional sentiment for a single restaurant."""
        try:
            google_text = self._format_reviews(google_reviews)
            supp_text = str(supp_reviews)[:3000] if supp_reviews else "No supplementary reviews."

            rating = info.get("rating", 0)
            review_count = info.get("review_count") or info.get("user_ratings_total") or 0
            address = info.get("address", "")

            prompt = SENTIMENT_ANALYSIS_PROMPT.format(
                restaurant_name=name,
                google_reviews=google_text[:5000],
                supplementary_reviews=supp_text,
                rating=rating,
                review_count=review_count,
                address=address,
                rating_val=rating or 0,
                review_count_val=review_count or 0,
            )

            data = self._llm_json_call(prompt)
            if data:
                # Handle nested dicts for SentimentQuality
                for key in ["service_quality", "food_quality", "atmosphere", "value_perception"]:
                    if isinstance(data.get(key), dict):
                        data[key] = SentimentQuality(**data[key])
                return RestaurantSentiment(**data)
        except Exception as e:
            logger.error(f"[ANALYST] Sentiment analysis failed for {name}: {e}")
        return None

    def _parse_menu(self, menu_data: dict) -> ParsedMenu | None:
        """Parse raw menu content into structured MenuItem objects."""
        try:
            restaurant_name = menu_data.get("restaurant_name", "Unknown")
            source_url = menu_data.get("source_url", "")
            raw_content = menu_data.get("raw_content") or menu_data.get("content") or str(menu_data)

            if not raw_content or len(str(raw_content)) < 50:
                return None

            prompt = MENU_PARSING_PROMPT.format(
                restaurant_name=restaurant_name,
                raw_menu=str(raw_content)[:15000],
                source_url=source_url,
            )

            data = self._llm_json_call(prompt)
            if data:
                items = []
                for item in data.get("items", []):
                    items.append(MenuItem(**item))
                return ParsedMenu(
                    restaurant_name=data.get("restaurant_name", restaurant_name),
                    source_url=data.get("source_url", source_url),
                    items=items,
                    total_items=data.get("total_items", len(items)),
                    categories=data.get("categories", []),
                )
        except Exception as e:
            logger.error(f"[ANALYST] Menu parsing failed: {e}")
        return None

    def _compare_menus(self, target_name: str, parsed_menus: list[ParsedMenu]) -> MenuComparison:
        """Build item-by-item price comparison."""
        try:
            target_menu = next((m for m in parsed_menus if target_name.lower() in m.restaurant_name.lower()), None)
            if not target_menu and parsed_menus:
                target_menu = parsed_menus[0]

            competitor_menus = [m for m in parsed_menus if m != target_menu]

            prompt = MENU_COMPARISON_PROMPT.format(
                target_name=target_name,
                target_menu=target_menu.model_dump_json()[:8000] if target_menu else "No target menu",
                competitor_menus=json.dumps([m.model_dump() for m in competitor_menus[:3]], default=str)[:10000],
            )

            data = self._llm_json_call(prompt)
            if data:
                comparisons = [MenuItemComparison(**c) for c in data.get("item_comparisons", [])]
                return MenuComparison(
                    target_menu=target_menu,
                    competitor_menus=competitor_menus[:3],
                    item_comparisons=comparisons,
                    unique_offerings=data.get("unique_offerings", []),
                    pricing_summary=data.get("pricing_summary", ""),
                )
        except Exception as e:
            logger.error(f"[ANALYST] Menu comparison failed: {e}")
        return MenuComparison(pricing_summary="Menu comparison could not be completed.")

    def _analyze_keywords(self, raw_reviews: dict, supp_reviews: dict) -> ReviewKeywordAnalysis:
        """Keyword frequency extraction across all reviews."""
        try:
            all_reviews_text = ""
            for name, data in raw_reviews.items():
                reviews = data.get("reviews", []) if isinstance(data, dict) else []
                for r in reviews[:10]:
                    text = r.get("text", "") if isinstance(r, dict) else str(r)
                    all_reviews_text += f"\n[{name}]: {text}"

            for name, data in supp_reviews.items():
                if isinstance(data, list):
                    for item in data[:5]:
                        content = item.get("content", "") if isinstance(item, dict) else str(item)
                        all_reviews_text += f"\n[{name} - supplementary]: {content[:500]}"
                elif isinstance(data, str):
                    all_reviews_text += f"\n[{name} - supplementary]: {data[:500]}"

            if not all_reviews_text.strip():
                return ReviewKeywordAnalysis()

            prompt = KEYWORD_ANALYSIS_PROMPT.format(
                all_reviews=all_reviews_text[:12000],
            )

            data = self._llm_json_call(prompt)
            if data:
                return ReviewKeywordAnalysis(**data)
        except Exception as e:
            logger.error(f"[ANALYST] Keyword analysis failed: {e}")
        return ReviewKeywordAnalysis()

    def _calculate_metrics(
        self, target: dict, competitors: list, sentiments: list[RestaurantSentiment]
    ) -> list[PerformanceMetrics]:
        """Calculate performance metrics for all restaurants."""
        try:
            # Build restaurants data
            restaurants = []

            # Target
            target_sentiment = next((s for s in sentiments if s.name == target.get("name")), None)
            restaurants.append({
                "name": target.get("name", "Unknown"),
                "rating": target.get("rating"),
                "review_count": target.get("review_count") or target.get("user_ratings_total"),
                "sentiment_score": target_sentiment.sentiment_score if target_sentiment else None,
                "is_target": True,
            })

            # Competitors
            for comp in competitors[:5]:
                comp_sentiment = next((s for s in sentiments if s.name == comp.get("name")), None)
                restaurants.append({
                    "name": comp.get("name", "Unknown"),
                    "rating": comp.get("rating"),
                    "review_count": comp.get("review_count") or comp.get("user_ratings_total"),
                    "sentiment_score": comp_sentiment.sentiment_score if comp_sentiment else None,
                    "is_target": False,
                })

            prompt = PERFORMANCE_METRICS_PROMPT.format(
                restaurants_data=json.dumps(restaurants, default=str),
            )

            data = self._llm_json_call(prompt)
            if data and isinstance(data, list):
                return [PerformanceMetrics(**m) for m in data]
        except Exception as e:
            logger.error(f"[ANALYST] Performance metrics failed: {e}")
        return []

    def _generate_recommendations(
        self,
        swot_analyses: list[SWOTAnalysis],
        sentiment: SentimentAnalysis,
        menu_comparison: MenuComparison,
        metrics: list[PerformanceMetrics],
        market_data: list,
    ) -> StrategicRecommendations:
        """Generate 7-pillar strategic recommendations."""
        try:
            swot_text = json.dumps([s.model_dump() for s in swot_analyses], default=str)[:5000]
            sentiment_text = sentiment.comparative_summary + "\n" + json.dumps(
                [r.model_dump() for r in sentiment.restaurants[:3]], default=str
            )[:5000]
            menu_text = menu_comparison.pricing_summary[:2000]
            metrics_text = json.dumps([m.model_dump() for m in metrics], default=str)[:3000]
            market_text = json.dumps(market_data[:3], default=str)[:3000]

            prompt = STRATEGIC_RECOMMENDATIONS_PROMPT.format(
                swot_data=swot_text,
                sentiment_summary=sentiment_text,
                menu_summary=menu_text,
                metrics_summary=metrics_text,
                market_context=market_text,
            )

            data = self._llm_json_call(prompt)
            if data:
                pillars = [StrategicPillar(**p) for p in data.get("pillars", [])]
                return StrategicRecommendations(
                    pillars=pillars,
                    immediate_actions=data.get("immediate_actions", []),
                    short_term_goals=data.get("short_term_goals", []),
                    long_term_vision=data.get("long_term_vision", ""),
                )
        except Exception as e:
            logger.error(f"[ANALYST] Strategic recommendations failed: {e}")
        return StrategicRecommendations()

    def _build_positioning(
        self, target_name: str, swot_analyses: list[SWOTAnalysis], competitors: list
    ) -> CompetitivePositioning:
        """Build competitive positioning from SWOT data (no LLM call)."""
        target_swot = next((s for s in swot_analyses if s.restaurant_name == target_name), None)

        positions = {}
        for s in swot_analyses:
            if s.restaurant_name != target_name:
                positions[s.restaurant_name] = s.overall_impression

        return CompetitivePositioning(
            target_position=target_swot.overall_impression if target_swot else "",
            competitor_positions=positions,
            differentiation_factors=target_swot.strengths[:5] if target_swot else [],
            vulnerability_factors=target_swot.weaknesses[:5] if target_swot else [],
        )

    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------

    def _format_reviews(self, reviews_data: dict | list) -> str:
        """Format review data into readable text."""
        if isinstance(reviews_data, dict):
            review_list = reviews_data.get("reviews", [])
        elif isinstance(reviews_data, list):
            review_list = reviews_data
        else:
            return "No reviews available."

        texts = []
        for r in review_list[:15]:
            if isinstance(r, dict):
                rating = r.get("rating", "N/A")
                text = r.get("text", "")
                author = r.get("author", "Anonymous")
                texts.append(f"[{rating}/5 - {author}]: {text[:300]}")
            else:
                texts.append(str(r)[:300])

        return "\n".join(texts) if texts else "No reviews available."

    def _llm_json_call(self, prompt: str) -> dict | list | None:
        """Make an LLM call and extract JSON from the response."""
        try:
            messages = [
                SystemMessage(content=ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            return self._extract_json(response.content)
        except Exception as e:
            logger.error(f"[ANALYST] LLM call failed: {e}")
            return None

    def _extract_json(self, text: str) -> dict | list | None:
        """Robust JSON extractor."""
        try:
            return json.loads(text)
        except Exception:
            pass

        try:
            match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip())
        except Exception:
            pass

        # Try finding JSON object or array
        try:
            # Find the first { or [
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = text.find(start_char)
                if start != -1:
                    end = text.rfind(end_char)
                    if end > start:
                        return json.loads(text[start:end + 1])
        except Exception:
            pass

        return None
