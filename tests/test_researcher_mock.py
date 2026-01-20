
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.researcher import ResearcherAgent
from config import AgentConfig
from models.schemas import ResearcherOutput
from langchain_core.messages import AIMessage, ToolMessage

class TestResearcherAgent(unittest.TestCase):
    
    def setUp(self):
        self.mock_config = AgentConfig()
        # Mock keys to pass init validation if any
        self.mock_config.google_maps_api_key = "test_key"
        self.mock_config.tavily_api_key = "test_key"
        self.mock_config.openai_api_key = "test_key"

    @patch("agents.researcher.create_llm")
    @patch("agents.researcher.create_google_maps_tools")
    @patch("agents.researcher.create_tavily_tools")
    @patch("agents.researcher.create_tavily_extract_tools")
    @patch("agents.researcher.create_scraper_tools")
    def test_research_flow(self, mock_scraper, mock_extract, mock_tavily, mock_gmaps, mock_create_llm):
        # 1. Setup Data
        plan = {
            "target_restaurant": "Test Target",
            "location": "Test City",
            "cuisine_type": "Test Cuisine",
            "search_queries": ["Query 1", "Query 2"]
        }

        # 2. Mock LLM
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        
        # Mock Tools
        mock_tool = MagicMock()
        mock_tool.name = "find_restaurant"
        mock_tool.invoke.return_value = {"name": "Test Target", "place_id": "123", "address": "123 Main St"}
        
        mock_gmaps.return_value = [mock_tool]
        mock_tavily.return_value = []
        mock_extract.return_value = []
        mock_scraper.return_value = []

        # 3. Setup LLM Responses for the ReAct Loop
        # 1. Tool Call: find_restaurant
        tool_call_1 = AIMessage(
            content="Finding target...", 
            tool_calls=[{"name": "find_restaurant", "args": {"name": "Test Target"}, "id": "call_1"}]
        )
        
        # 2. Tool Call: web_search (for business background)
        tool_call_2 = AIMessage(
            content="Searching history...", 
            tool_calls=[{"name": "web_search", "args": {"query": "Test Target history"}, "id": "call_2"}]
        )

        # 3. Final "No tool calls"
        final_msg = AIMessage(content="Research complete.")
        
        # Sentiment Analysis Response
        sentiment_msg = AIMessage(content="""
        {
            "restaurants": []
        }
        """)

        # Menu Comparison Response
        menu_msg = AIMessage(content='{"item_comparisons": [], "pricing_summary": "Test Summary"}')
        
        # Business Background Response
        bg_msg = AIMessage(content='{"founding_year": "2020", "history_summary": "Test History"}')

        mock_llm.bind_tools.return_value = mock_llm 
        mock_llm.invoke.side_effect = [
            tool_call_1,    # ReAct Iteration 1
            tool_call_2,    # ReAct Iteration 2
            final_msg,      # ReAct Iteration 3 (Stop)
            sentiment_msg,  # Sentiment
            bg_msg,         # Background (Called first in _build_output)
            menu_msg        # Menu (Called second in _build_output)
        ]

        # Setup Mock Tool Returns
        # We need to handle different tool calls dynamically or mock the tool_map logic better
        # Since currently tool_map is manually injected in test, let's fix that.
        
        mock_find = MagicMock()
        mock_find.name = "find_restaurant"
        mock_find.invoke.return_value = {"name": "Test Target", "place_id": "123", "address": "123 Main St"}
        
        mock_search = MagicMock()
        mock_search.name = "web_search"
        # Return content with trigger words "founded", "history"
        mock_search.invoke.return_value = {"content": "Test Target was founded in 2020 by John Doe. Great history."}
        
        agent = ResearcherAgent(self.mock_config)
        agent.tool_map = {
            "find_restaurant": mock_find,
            "web_search": mock_search
        }
        
        output = agent.research(plan)

        # 5. Assertions
        print(f"Output Type: {type(output)}")
        print(f"Target Name: {output.target.name}")
        
        self.assertIsInstance(output, ResearcherOutput)
        self.assertEqual(output.target.name, "Test Target")
        self.assertEqual(output.target.place_id, "123")
        self.assertEqual(output.target.business_background.founding_year, "2020")

if __name__ == "__main__":
    unittest.main()
