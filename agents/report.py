"""REPORT Agent - Generates client-ready market research reports."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import AgentConfig
from prompts import REPORT_SYSTEM_PROMPT
from utils import create_llm


class ReportAgent:
    """
    NODE 4 — REPORT AGENT
    
    Generates a client-ready Market Research Report using LLM.
    Returns markdown report text.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config, temperature=0.3)

    def generate(
        self,
        plan: dict[str, Any],
        research: dict[str, Any],
        critique: dict[str, Any],
    ) -> str:
        """
        Generate a comprehensive market research report.
        
        Args:
            plan: Research plan dict from Planner
            research: Research findings dict from Researcher
            critique: Quality assessment dict from Critic
            
        Returns:
            Markdown report text
        """
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        
        logger.info(f"[REPORT] Generating report for {target} in {location}")

        context = self._prepare_context(plan, research, critique)

        messages = [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        # LLM generates markdown report
        response = self.llm.invoke(messages)
        report_text = response.content if hasattr(response, 'content') else str(response)

        logger.info("[REPORT] Report generation complete")
        return report_text

    def _prepare_context(
        self,
        plan: dict[str, Any],
        research: dict[str, Any],
        critique: dict[str, Any],
    ) -> str:
        """Prepare comprehensive context for report generation."""
        target = plan.get("target_restaurant", "Unknown")
        location = plan.get("location", "Unknown")
        cuisine_type = plan.get("cuisine_type", "unknown")
        intent = plan.get("intent", "")
        
        competitors = research.get("competitors", [])
        pricing = research.get("pricing_analysis", {})
        sentiment = research.get("sentiment_analysis", {})
        market_signals = research.get("market_signals", {})
        menu = research.get("menu_comparison", {})
        raw_sources = research.get("raw_sources", [])
        research_notes = research.get("research_notes", [])
        
        decision = critique.get("decision", "UNKNOWN")
        quality_score = critique.get("overall_quality_score", 0)
        strengths = critique.get("strengths", [])
        banking_assessment = critique.get("banking_suitability_assessment", "")

        target_info = research.get("target", {}) or {}
        business_bg = target_info.get("business_background", {})
        
        # Format competitors summary
        competitor_summary = [
            {
                "name": c.get("name"),
                "distance_miles": c.get("distance_miles"),
                "rating": c.get("rating"),
                "review_count": c.get("review_count"),
                "price_level": c.get("price_level"),
                "cuisine_type": c.get("cuisine_type"),
                "cuisine_subtype": c.get("cuisine_subtype"),
            }
            for c in competitors[:10]
        ]

        # Build menu comparison table
        menu_comparison_table = self._build_menu_comparison_table(menu)
        
        return f"""## RESEARCH DATA FOR REPORT GENERATION

### Target Restaurant
- **Name:** {target}
- **Location:** {location}
- **Cuisine Type:** {cuisine_type}
- **Research Intent:** {intent}

### Business Background
```json
{json.dumps(business_bg, indent=2, default=str)}
```

### Selected Competitors
```json
{json.dumps(competitor_summary, indent=2, default=str)}
```

### Pricing Analysis
```json
{json.dumps(pricing, indent=2, default=str)}
```

### Menu Intelligence & Item-by-Item Comparison

{menu_comparison_table}

### Customer Sentiment
```json
{json.dumps(sentiment, indent=2, default=str)}
```

### Market Signals
```json
{json.dumps(market_signals, indent=2, default=str)}
```

### Quality Assessment (from Critic Agent)
- **Decision:** {decision}
- **Quality Score:** {quality_score:.2f}/1.00
- **Strengths Identified:** {strengths}
- **Banking Suitability:** {banking_assessment}

### Research Notes
{research_notes}

### Sources Used
{len(raw_sources)} sources referenced

---

Generate a comprehensive, professional Market Research Report based on ALL the data above.
Include the ITEM-BY-ITEM PRICE COMPARISON TABLE in the report.
Synthesize insights, don't just repeat data. Think like a commercial banker evaluating this restaurant."""

    def _build_menu_comparison_table(self, menu: dict) -> str:
        """Build a markdown table for menu item comparison."""
        if not menu:
            return "No menu data available."
        
        # Extract item comparisons
        item_comparisons = menu.get("item_comparisons", [])
        pricing_summary = menu.get("pricing_summary", "")
        unique_offerings = menu.get("unique_offerings", [])
        target_menu = menu.get("target_menu", {})
        competitor_menus = menu.get("competitor_menus", [])
        
        output = []
        
        # Pricing summary
        if pricing_summary:
            output.append(f"**Pricing Summary:** {pricing_summary}\n")
        
        # Menu stats
        if target_menu:
            target_items = target_menu.get("total_items", 0) if isinstance(target_menu, dict) else 0
            output.append(f"**Target Menu Items:** {target_items}")
        
        if competitor_menus:
            comp_stats = ", ".join([
                f"{m.get('restaurant_name', 'Unknown')}: {m.get('total_items', 0)} items" 
                for m in competitor_menus if isinstance(m, dict)
            ])
            output.append(f"**Competitor Menu Items:** {comp_stats}\n")
        
        # Item comparison table
        if item_comparisons:
            output.append("\n**Item-by-Item Price Comparison (Common Items):**\n")
            output.append("| Item | Category | Target Price | Competitor Prices | Difference |")
            output.append("|------|----------|--------------|-------------------|------------|")
            
            for comp in item_comparisons[:20]:  # Top 20
                if isinstance(comp, dict):
                    item_name = comp.get("item_name", "")[:30]
                    category = comp.get("category", "")[:15]
                    target_price = comp.get("target_price")
                    target_str = f"${target_price:.2f}" if target_price else "N/A"
                    
                    comp_prices = comp.get("competitor_prices", {})
                    comp_str = ", ".join([f"{k[:10]}: ${v:.2f}" for k, v in comp_prices.items() if v]) if comp_prices else "N/A"
                    
                    diff = comp.get("price_difference_avg")
                    diff_str = f"+${diff:.2f}" if diff and diff > 0 else (f"${diff:.2f}" if diff else "N/A")
                    
                    output.append(f"| {item_name} | {category} | {target_str} | {comp_str} | {diff_str} |")
        else:
            output.append("\nNo common items found for direct comparison.")
        
        # Unique offerings
        if unique_offerings:
            output.append(f"\n**Unique Items (Target Only):** {', '.join(unique_offerings[:10])}")
        
        return "\n".join(output)

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        planner_output = state.get("planner_output")
        researcher_output = state.get("researcher_output")
        critic_output = state.get("critic_output")

        if not all([planner_output, researcher_output, critic_output]):
            raise ValueError("Missing required outputs from previous agents")

        # All inputs are dicts - no conversion needed

        try:
            report_text = self.generate(planner_output, researcher_output, critic_output)
            return {
                **state,
                "report_output": report_text,  # Store as markdown string
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
