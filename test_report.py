# test_report.py - Run this to test the Report node in isolation

from agents.report import ReportAgent
from config import get_config

# Mock data that simulates what Planner, Researcher, and Critic would produce
mock_plan = {
    "target_restaurant": "Test Restaurant",
    "location": "New York, NY",
    "cuisine_type": "Italian",
    "intent": "Evaluate for lending"
}

mock_research = {
    "target_restaurant": {"name": "Test Restaurant", "rating": 4.5, "review_count": 200},
    "competitors": [
        {"name": "Competitor 1", "rating": 4.2, "review_count": 150, "distance_miles": 0.5, "price_level": "$$"},
        {"name": "Competitor 2", "rating": 4.0, "review_count": 100, "distance_miles": 0.8, "price_level": "$$$"},
    ],
    "pricing_analysis": {"avg_price": 25, "market_position": "mid-range"},
    "sentiment_analysis": {"overall": "positive", "strengths": ["good food", "nice ambiance"]},
    "market_signals": {"saturation": "moderate"},
    "menu_comparison": {"items_analyzed": 10},
    "raw_sources": [],
    "research_notes": ["Completed competitor analysis"]
}

mock_critique = {
    "decision": "ACCEPT",
    "overall_quality_score": 0.85,
    "strengths": ["Good competitor data", "Solid pricing analysis"],
    "banking_suitability_assessment": "Suitable for lending consideration"
}

# Test the Report agent
config = get_config()
agent = ReportAgent(config)

print("Testing Report Agent...")
try:
    report = agent.generate(mock_plan, mock_research, mock_critique)
    print("\n[SUCCESS] Report generated:")
    print("-" * 60)
    print(report[:2000] + "..." if len(report) > 2000 else report)
except Exception as e:
    print(f"\n[ERROR]: {e}")
    import traceback
    traceback.print_exc()

