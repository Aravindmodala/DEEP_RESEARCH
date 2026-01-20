"""PLANNER Agent - Converts user queries into research plans."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from tavily import TavilyClient

from config import AgentConfig
from models.schemas import PlannerOutput
from prompts import PLANNER_SYSTEM_PROMPT
from utils import create_llm


console = Console()


def create_planner_tavily_tool(api_key: str | None = None):
    """Create a Tavily search tool for the planner agent."""
    tavily_key = api_key or os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY not found")
    
    client = TavilyClient(api_key=tavily_key)
    
    @tool
    def web_search(query: str) -> str:
        """
        Search the web to find information about a restaurant's cuisine type.
        Use this to identify the exact cuisine type of a restaurant.
        
        Args:
            query: Search query - include restaurant name, location, and 'cuisine type'
        
        Returns:
            Search results with relevant information about the restaurant
        """
        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_domains=["yelp.com", "tripadvisor.com", "google.com", "zomato.com", "grubhub.com", "doordash.com"]
            )
            
            results = response.get("results", [])
            if not results:
                return "No search results found."
            
            output = []
            for r in results[:5]:
                output.append(f"Title: {r.get('title', 'Unknown')}\nContent: {r.get('content', '')[:500]}\n")
            
            return "\n---\n".join(output)
        except Exception as e:
            logger.error(f"[PLANNER] Tavily search error: {e}")
            return f"Search failed: {str(e)}"
    
    return web_search


class PlannerAgent:
    """
    NODE 1 — PLANNER AGENT
    
    Converts user's natural-language request into a research plan.
    Uses Tavily search tool to identify cuisine type.
    Returns a validated PlannerOutput Pydantic model.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config)
        
        # Initialize Tavily client and tool for cuisine detection
        tavily_key = config.tavily_api_key or os.getenv("TAVILY_API_KEY")
        if tavily_key:
            self.tavily = TavilyClient(api_key=tavily_key)
            self.tavily_tool = create_planner_tavily_tool(tavily_key)
            self.tools = [self.tavily_tool]
            # Bind tools to LLM for tool calling
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.tavily = None
            self.tavily_tool = None
            self.tools = []
            self.llm_with_tools = self.llm
            logger.warning("[PLANNER] Tavily API key not found - cuisine detection may be limited")

    def plan(self, user_query: str) -> PlannerOutput:
        """
        Convert a user query into a research plan.
        Uses Tavily search tool to identify cuisine type.
        
        Returns validated PlannerOutput Pydantic model.
        """
        logger.info(f"[PLANNER] Processing query: {user_query[:100]}...")

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {user_query}"),
        ]

        # Tool calling loop - allow LLM to use web_search for cuisine detection
        max_iterations = 3
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Use LLM with tools bound
            if self.tools:
                response = self.llm_with_tools.invoke(messages)
            else:
                response = self.llm.invoke(messages)
            
            # Check if response has tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"[PLANNER] LLM requested {len(response.tool_calls)} tool call(s)")
                
                # Add AI message with tool calls to history
                messages.append(response)
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")
                    
                    logger.info(f"[PLANNER] Executing tool: {tool_name} with args: {tool_args}")
                    
                    if tool_name == "web_search" and self.tavily_tool:
                        try:
                            result = self.tavily_tool.invoke(tool_args)
                            logger.info(f"[PLANNER] Tool result received ({len(str(result))} chars)")
                        except Exception as e:
                            result = f"Tool execution failed: {str(e)}"
                            logger.error(f"[PLANNER] Tool error: {e}")
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                
                # Continue loop to get final response
                continue
            
            # No tool calls - extract JSON from response
            break
        
        # Extract JSON from final response
        response_content = response.content if hasattr(response, 'content') else str(response)
        plan_dict = self._extract_json(response_content)

        if not plan_dict:
            logger.error("[PLANNER] Failed to extract JSON from response")
            # Return minimal plan rather than failing
            plan_dict = {
                "target_restaurant": "Unknown",
                "location": "Unknown",
                "cuisine_type": "unknown",
                "intent": user_query,
                "search_queries": [user_query],
            }

        # Validate and return as Pydantic model
        output = PlannerOutput(**plan_dict)
        logger.info(f"[PLANNER] Plan created for {output.target_restaurant} (cuisine: {output.cuisine_type})")
        return output

    def plan_with_confirmation(self, user_query: str) -> PlannerOutput:
        """Generate a research plan with human confirmation."""
        console.print("\n[bold cyan]Analyzing your request...[/bold cyan]\n")
        
        # Generate initial plan with cuisine detection via tool calling
        console.print("[dim]Detecting cuisine type via web search...[/dim]")
        plan_output = self.plan(user_query)
        
        # Convert to dict for interactive editing
        plan = plan_output.model_dump()
        
        # If cuisine type is still unknown, try fallback detection
        if plan.get("cuisine_type", "unknown").lower() == "unknown":
            console.print("[dim]Attempting fallback cuisine detection...[/dim]")
            cuisine = self._detect_cuisine_from_web(
                plan.get("target_restaurant", ""),
                plan.get("location", "")
            )
            if cuisine and cuisine.lower() != "unknown":
                plan["cuisine_type"] = cuisine
                # Update first search query with cuisine type
                queries = plan.get("search_queries", [])
                if queries:
                    queries[0] = f"{cuisine} restaurants competitors near {plan.get('location', '')}"
                    plan["search_queries"] = queries
        
        # Confirmation loop
        while True:
            self._display_plan_for_confirmation(plan)
            
            choice = Prompt.ask(
                "\n[bold cyan]Your choice[/bold cyan]",
                choices=["y", "e", "q"],
                default="y"
            ).lower()
            
            if choice == "y":
                console.print("\n[bold green]✓ Plan confirmed! Starting research...[/bold green]\n")
                return PlannerOutput(**plan)
            elif choice == "e":
                plan = self._interactive_edit_plan(plan)
            elif choice == "q":
                console.print("\n[yellow]Research cancelled by user.[/yellow]")
                raise KeyboardInterrupt("User cancelled research")
    
    def _detect_cuisine_from_web(self, restaurant: str, location: str) -> str:
        """Detect cuisine type using Tavily web search + LLM extraction."""
        if not self.tavily or not restaurant:
            return "unknown"
        
        try:
            search_query = f"{restaurant} {location} restaurant cuisine type food menu"
            logger.info(f"[PLANNER] Searching for cuisine: {search_query}")
            
            response = self.tavily.search(
                query=search_query,
                search_depth="basic",
                max_results=5,
                include_domains=["yelp.com", "tripadvisor.com", "google.com", "zomato.com", "grubhub.com"]
            )
            
            results = response.get("results", [])
            if not results:
                return "unknown"
            
            search_context = "\n\n".join([
                f"Source: {r.get('title', 'Unknown')}\nContent: {r.get('content', '')}"
                for r in results[:5]
            ])
            
            extraction_prompt = f"""Based on the following web search results, identify the EXACT CUISINE SUBTYPE of "{restaurant}" in {location}.

SEARCH RESULTS:
{search_context}

Respond with ONLY the specific cuisine type (e.g., "South Indian" instead of just "Indian", "Tuscan" instead of "Italian").
If specific subtype is not clear, use the primary cuisine.
Do not include any explanation."""

            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            cuisine = response.content.strip().strip('"').strip("'").split("\n")[0].strip()
            
            if len(cuisine) > 50:
                cuisine = "unknown"
            
            logger.info(f"[PLANNER] Detected cuisine from web: {cuisine}")
            return cuisine
            
        except Exception as e:
            logger.error(f"[PLANNER] Error detecting cuisine: {e}")
            return "unknown"
    
    def _display_plan_for_confirmation(self, plan: dict[str, Any]) -> None:
        """Display the research plan for user confirmation."""
        queries = plan.get("search_queries", [])
        queries_text = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(queries)])
        
        content = f"""[cyan]Restaurant:[/cyan]  {plan.get('target_restaurant', 'Unknown')}
[cyan]Location:[/cyan]    {plan.get('location', 'Unknown')}
[cyan]Cuisine:[/cyan]     {plan.get('cuisine_type', 'unknown')}
[cyan]Goal:[/cyan]        {plan.get('intent', 'Unknown')}

[bold]Research Queries:[/bold]
{queries_text}

[dim]─────────────────────────────────────────────────────────[/dim]

[bold]Options:[/bold]
  [green][Y][/green] Proceed with research
  [yellow][E][/yellow] Edit plan
  [red][Q][/red] Quit"""
        
        console.print("\n")
        console.print(Panel.fit(content, title="[bold white]RESEARCH PLAN[/bold white]", border_style="cyan", padding=(1, 2)))
    
    def _interactive_edit_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Allow user to interactively edit plan fields."""
        console.print("\n[bold yellow]What would you like to edit?[/bold yellow]")
        console.print("  [1] Restaurant name")
        console.print("  [2] Location")
        console.print("  [3] Cuisine type")
        console.print("  [4] Research goal")
        console.print("  [5] Search queries")
        console.print("  [B] Back (no changes)")
        
        choice = Prompt.ask("\n[cyan]Choice[/cyan]", choices=["1", "2", "3", "4", "5", "b"], default="b").lower()
        
        if choice == "b":
            return plan
        
        if choice == "1":
            plan["target_restaurant"] = Prompt.ask(
                f"[cyan]Restaurant name[/cyan]",
                default=plan.get("target_restaurant", "")
            )
        elif choice == "2":
            plan["location"] = Prompt.ask(
                f"[cyan]Location[/cyan]",
                default=plan.get("location", "")
            )
        elif choice == "3":
            new_cuisine = Prompt.ask(
                f"[cyan]Cuisine type[/cyan]",
                default=plan.get("cuisine_type", "unknown")
            )
            plan["cuisine_type"] = new_cuisine
            # Update first query
            if new_cuisine.lower() != "unknown":
                queries = plan.get("search_queries", [])
                if queries:
                    queries[0] = f"{new_cuisine} restaurants competitors near {plan.get('location', '')}"
                    plan["search_queries"] = queries
        elif choice == "4":
            plan["intent"] = Prompt.ask(
                f"[cyan]Research goal[/cyan]",
                default=plan.get("intent", "")
            )
        elif choice == "5":
            queries = plan.get("search_queries", [])
            new_queries = []
            console.print("\n[dim]Enter new queries (press Enter to keep current):[/dim]")
            for i, query in enumerate(queries):
                new_val = Prompt.ask(f"  [cyan]Query {i+1}[/cyan]", default=query)
                new_queries.append(new_val)
            plan["search_queries"] = new_queries
        
        return plan

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from LLM response text."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try markdown JSON block
        json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end])
            except json.JSONDecodeError:
                pass

        return None

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible call interface."""
        user_query = state.get("user_query", "")
        if not user_query:
            raise ValueError("No user_query found in state")

        try:
            output = self.plan(user_query)
            return {
                **state,
                "planner_output": output.model_dump(),  # Convert Pydantic to dict for state
                "current_node": "researcher",
            }
        except Exception as e:
            logger.error(f"[PLANNER] Error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Planner error: {str(e)}"],
                "current_node": "error",
            }
