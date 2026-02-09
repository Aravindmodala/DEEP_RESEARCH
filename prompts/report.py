"""Section-specific prompts for the REPORT agent."""

REPORT_BASE_INSTRUCTIONS = """You are generating ONE SECTION of a comprehensive restaurant market analysis report.
Write in professional, bank-grade tone. Be evidence-based and specific.
Use inline citations in the format [Source N] where N matches the source registry provided.
Do NOT include markdown section headers - those are added automatically.
Write clear, well-structured prose with bullet points where appropriate."""

REPORT_SECTION_PROMPTS = {
    "executive_summary": """Write an Executive Summary for {target} in {location}.

DATA:
- Business Background: {business_background}
- Competitors: {competitors_summary}
- Overall Sentiment: {sentiment_summary}
- Market Position: {market_position}

REQUIREMENTS:
- Business overview (founding, owners, history) in 2-3 sentences
- Market position vs competitors in 2-3 sentences
- Top 3 key findings as bullet points
- Keep to ~300 words

{citation_instructions}""",

    "history_founding": """Write the History and Founding section for {target}.

DATA:
- Business Intelligence: {business_intel}
- Background: {business_background}

REQUIREMENTS:
- Founding year and founders/owners
- Origin story and concept evolution
- Key milestones
- Corporate structure if available
- If data is limited, note what was found and what wasn't

{citation_instructions}""",

    "dining_concept_menu": """Write the Dining Concept and Menu section for {target}.

DATA:
- Target Menu: {target_menu}
- Cuisine Type: {cuisine_type}
- Unique Offerings: {unique_offerings}

REQUIREMENTS:
- Describe the dining concept and cuisine focus
- Highlight signature items and price ranges
- Note menu breadth (number of items, categories)
- Describe any specialty dietary offerings
- If menu data is limited, note what was found

{citation_instructions}""",

    "allergy_dietary": """Write the Allergy-Friendly / Dietary Commitment section for {target}.

DATA:
- Menu Data: {target_menu}
- Business Intel: {business_intel}

REQUIREMENTS:
- Describe any allergen-friendly options (gluten-free, nut-free, etc.)
- Note vegetarian/vegan options
- Describe any dietary labeling practices
- If no specific allergy info found, note that and infer from menu if possible

{citation_instructions}""",

    "expansion_locations": """Write the Expansion and Locations section for {target}.

DATA:
- Business Background: {business_background}
- Business Intel: {business_intel}

REQUIREMENTS:
- Current number of locations
- Expansion timeline and strategy
- Recent openings or planned locations
- Geographic footprint
- Sister brands if any

{citation_instructions}""",

    "leadership_ownership": """Write the Leadership and Ownership section for {target}.

DATA:
- Business Background: {business_background}
- Business Intel: {business_intel}

REQUIREMENTS:
- Key leadership team members
- Ownership structure (franchise vs corporate, family-owned, etc.)
- Management style or philosophy if available
- Any notable executives or board members

{citation_instructions}""",

    "community_engagement": """Write the Community Engagement section for {target}.

DATA:
- Business Intel: {business_intel}
- Business Background: {business_background}
- Reviews: {review_summary}

REQUIREMENTS:
- Community partnerships and local involvement
- Charity events, sponsorships
- Local reputation and community standing
- Social media presence indicators

{citation_instructions}""",

    "awards_recognition": """Write the Awards and Recognition section for {target}.

DATA:
- Business Intel: {business_intel}
- Business Background: {business_background}

REQUIREMENTS:
- Industry awards and accolades
- Media mentions and press coverage
- Notable recognitions (Best of, Top 10, etc.)
- If no awards data found, note that explicitly

{citation_instructions}""",

    "comparison_analysis": """Write the Comparison Analysis section with SWOT for each restaurant.

DATA:
- SWOT Analyses: {swot_data}
- Competitors: {competitors_data}
- Competitive Positioning: {positioning_data}

REQUIREMENTS:
- Present SWOT (Strengths, Weaknesses, Opportunities, Threats) for the target restaurant
- Present SWOT for each analyzed competitor
- Use tables or structured bullet points for clarity
- Include an overall competitive positioning summary
- Note differentiation factors and vulnerabilities

{citation_instructions}""",

    "sentiment_analysis": """Write the Sentiment Analysis section with per-restaurant dimensional analysis.

DATA:
- Sentiment Data: {sentiment_data}
- Keyword Analysis: {keyword_data}

REQUIREMENTS:
- For each restaurant analyzed, present:
  - Overall sentiment score
  - Service quality score and themes
  - Food quality score and themes
  - Atmosphere score and themes
  - Value perception score and themes
- Include a comparative table if 3+ restaurants analyzed
- Highlight key complaints and top praises
- Reference specific review quotes or themes

{citation_instructions}""",

    "menu_comparison": """Write the Menu Comparison section with item-by-item price analysis.

DATA:
- Menu Comparison: {menu_comparison_data}
- Parsed Menus: {parsed_menus_summary}

REQUIREMENTS:
- Create an item-by-item price comparison table (markdown format)
- Include columns: Item | Category | Target Price | Competitor Prices | Difference
- Note pricing position (above/below/at market average)
- Highlight unique offerings only the target has
- Identify menu gaps (items competitors have that target doesn't)
- Include at least 10 items in the comparison table

{citation_instructions}""",

    "operational_analysis": """Write the Operational Analysis section covering keywords, metrics, and operational insights.

DATA:
- Keyword Analysis: {keyword_data}
- Performance Metrics: {metrics_data}
- Market Data: {market_data}

REQUIREMENTS:
- Present keyword frequency analysis (top positive and negative keywords)
- Show performance metrics comparison (rating, review volume, competitive rank)
- Identify operational strengths and weaknesses from review patterns
- Include foot traffic and demand indicators
- Create a competitive ranking table

{citation_instructions}""",

    "performance_strategy": """Write the Performance Strategy section with 7 strategic pillars, KPIs, and quarterly timeline.

DATA:
- Strategic Recommendations: {strategy_data}
- Performance Metrics: {metrics_data}
- SWOT Summary: {swot_summary}

REQUIREMENTS:
- Present 7 strategic recommendation pillars
- For each pillar include:
  - Description of the strategy
  - 2-3 KPI targets with specific numbers
  - Quarterly milestones (Q1 through Q4)
- Include immediate actions (next 30 days)
- Include short-term goals (3-6 months)
- Include long-term vision (12-18 months)
- Format as a structured, actionable roadmap

{citation_instructions}""",
}
