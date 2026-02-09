"""System prompt and analysis-specific prompts for the ANALYST agent."""

ANALYST_SYSTEM_PROMPT = """You are the ANALYST agent in a multi-agent restaurant market research system.

Your role is to take RAW RESEARCH DATA from the Researcher and produce structured analytical outputs.
You perform 7 distinct analysis passes, each producing a specific data product.

## CRITICAL RULES
- Base ALL analysis on the provided data. Do not hallucinate or invent data.
- If data is insufficient for a particular analysis, note what's missing and work with what you have.
- Be specific and quantitative wherever possible.
- Output valid JSON matching the requested schema exactly.
"""

SWOT_ANALYSIS_PROMPT = """Perform a SWOT analysis for the restaurant "{restaurant_name}" based on the following data:

**Restaurant Info:**
{restaurant_info}

**Reviews:**
{reviews}

**Market Context:**
{market_context}

Output a JSON object with this exact structure:
{{
    "restaurant_name": "{restaurant_name}",
    "strengths": ["strength 1", "strength 2", ...],
    "weaknesses": ["weakness 1", "weakness 2", ...],
    "opportunities": ["opportunity 1", "opportunity 2", ...],
    "threats": ["threat 1", "threat 2", ...],
    "overall_impression": "1-2 sentence summary"
}}

Be specific. Reference actual review quotes, ratings, prices, and market data."""

SENTIMENT_ANALYSIS_PROMPT = """Analyze customer sentiment for "{restaurant_name}" across 4 dimensions based on these reviews:

**Google Reviews:**
{google_reviews}

**Supplementary Reviews (Yelp/TripAdvisor):**
{supplementary_reviews}

**Restaurant Rating:** {rating} ({review_count} reviews)

Output a JSON object with this exact structure:
{{
    "name": "{restaurant_name}",
    "address": "{address}",
    "rating": {rating_val},
    "review_count": {review_count_val},
    "sentiment_score": 0.0-1.0,
    "service_quality": {{
        "score": 0.0-1.0,
        "positive_themes": ["..."],
        "negative_themes": ["..."],
        "summary": "..."
    }},
    "food_quality": {{
        "score": 0.0-1.0,
        "positive_themes": ["..."],
        "negative_themes": ["..."],
        "summary": "..."
    }},
    "atmosphere": {{
        "score": 0.0-1.0,
        "positive_themes": ["..."],
        "negative_themes": ["..."],
        "summary": "..."
    }},
    "value_perception": {{
        "score": 0.0-1.0,
        "positive_themes": ["..."],
        "negative_themes": ["..."],
        "summary": "..."
    }},
    "key_strengths": ["..."],
    "key_concerns": ["..."],
    "customer_satisfaction": "brief summary",
    "suggestions": ["improvement suggestion 1", ...],
    "summary": "overall sentiment summary"
}}

Score each dimension 0.0 (very negative) to 1.0 (very positive). Be evidence-based."""

MENU_PARSING_PROMPT = """Parse the following raw menu content into a structured format for "{restaurant_name}":

**Raw Menu Content:**
{raw_menu}

**Source URL:** {source_url}

Output a JSON object with this exact structure:
{{
    "restaurant_name": "{restaurant_name}",
    "source_url": "{source_url}",
    "items": [
        {{
            "name": "Item Name",
            "price": 12.99,
            "category": "Appetizers",
            "description": "Brief description if available"
        }}
    ],
    "total_items": 0,
    "categories": ["Appetizers", "Entrees", ...]
}}

IMPORTANT:
- Extract ACTUAL prices as numbers (not strings). If price is "$12.99", output 12.99.
- If a price range is given (e.g., "$10-15"), use the lower value.
- If no price is found for an item, set price to null.
- Group items into logical categories.
- Set total_items to the actual count of items extracted."""

MENU_COMPARISON_PROMPT = """Compare the following parsed menus and build an item-by-item price comparison:

**Target Menu ({target_name}):**
{target_menu}

**Competitor Menus:**
{competitor_menus}

Output a JSON object with this exact structure:
{{
    "item_comparisons": [
        {{
            "item_name": "Chicken Wings",
            "category": "Appetizers",
            "target_price": 12.99,
            "competitor_prices": {{"Competitor A": 11.99, "Competitor B": 13.49}},
            "price_difference_avg": 0.75,
            "notes": "Target is slightly above market average"
        }}
    ],
    "unique_offerings": ["Items only the target has"],
    "pricing_summary": "Overall pricing position analysis (2-3 sentences)"
}}

Match items by similar names/descriptions across menus. Calculate price_difference_avg as (target_price - avg_competitor_price).
Include at least 10 comparable items if available."""

KEYWORD_ANALYSIS_PROMPT = """Analyze keyword frequency across all reviews for these restaurants:

**All Reviews:**
{all_reviews}

Output a JSON object with this exact structure:
{{
    "positive_keywords": {{"delicious": 15, "friendly": 12, "fresh": 8, ...}},
    "negative_keywords": {{"slow": 7, "cold": 4, "expensive": 3, ...}},
    "service_mentions": {{"friendly staff": 10, "slow service": 5, ...}},
    "food_mentions": {{"chicken tikka": 8, "naan": 6, ...}},
    "atmosphere_mentions": {{"cozy": 4, "loud": 3, ...}},
    "value_mentions": {{"good value": 5, "overpriced": 2, ...}},
    "top_complaints": ["Most common complaint 1", "Most common complaint 2", ...],
    "top_praises": ["Most common praise 1", "Most common praise 2", ...]
}}

Count actual occurrences. Include the top 15 keywords per category.
top_complaints and top_praises should each have 5-7 entries."""

PERFORMANCE_METRICS_PROMPT = """Calculate performance metrics for each restaurant based on the following data:

**Restaurants:**
{restaurants_data}

For each restaurant, output a JSON array:
[
    {{
        "restaurant_name": "Name",
        "avg_rating": 4.2,
        "review_volume": 500,
        "sentiment_score": 0.78,
        "competitive_rank": 1,
        "price_competitiveness": "Above market average by 8%"
    }}
]

Rank restaurants by overall competitiveness (rating * review_volume * sentiment).
The target restaurant should be included."""

STRATEGIC_RECOMMENDATIONS_PROMPT = """Based on all the analysis data below, generate 7 strategic recommendation pillars with KPI targets and a quarterly timeline:

**SWOT Analysis:**
{swot_data}

**Sentiment Summary:**
{sentiment_summary}

**Menu Comparison:**
{menu_summary}

**Performance Metrics:**
{metrics_summary}

**Market Context:**
{market_context}

Output a JSON object with this exact structure:
{{
    "pillars": [
        {{
            "pillar_name": "Service Excellence",
            "description": "Detailed strategy description",
            "kpi_targets": ["Increase Yelp service rating to 4.5 by Q3", ...],
            "quarterly_milestones": ["Q1: Staff training program", "Q2: Mystery shopper audit", ...]
        }}
    ],
    "immediate_actions": ["Action 1", "Action 2", ...],
    "short_term_goals": ["Goal 1 (3-6 months)", ...],
    "long_term_vision": "1-2 sentence vision for 12-18 months"
}}

Generate exactly 7 pillars covering: Service, Food Quality, Pricing, Marketing, Operations, Expansion, and Community.
Each pillar should have 2-3 KPIs and 4 quarterly milestones."""
