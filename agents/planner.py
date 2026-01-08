"""PLANNER Agent - Converts user queries into structured research plans with human confirmation."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from tavily import TavilyClient

from config import AgentConfig
from models.schemas import PlannerOutput
from prompts import PLANNER_SYSTEM_PROMPT
from utils import create_llm


console = Console()


class PlannerAgent:
    """
    NODE 1 — PLANNER AGENT
    
    Responsibility: Convert user's natural-language request into a clear research plan.
    
    Features:
    - Extracts restaurant name, location, and intent from user query
    - Detects cuisine type via Tavily web search + LLM extraction
    - Presents plan to user for confirmation before proceeding
    - Allows user to edit any field before starting research
    
    Outputs:
    - target_restaurant: Name of the restaurant to analyze
    - location: City, State
    - cuisine_type: Detected from web search (e.g., Indian, Italian)
    - intent: Inferred user intent (lending risk, expansion, etc.)
    - search_queries: Exactly 4 research queries
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = create_llm(config)
        
        # Initialize Tavily client for cuisine detection
        tavily_key = config.tavily_api_key or os.getenv("TAVILY_API_KEY")
        if tavily_key:
            self.tavily = TavilyClient(api_key=tavily_key)
        else:
            self.tavily = None
            logger.warning("[PLANNER] Tavily API key not found - cuisine detection may be limited")

    def plan(self, user_query: str) -> PlannerOutput:
        """
        Convert a user query into a structured research plan.
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            PlannerOutput with structured research plan
        """
        logger.info(f"[PLANNER] Processing query: {user_query[:100]}...")

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {user_query}"),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content

        # Extract JSON from response
        plan_json = self._extract_json(response_text)

        if not plan_json:
            logger.error("[PLANNER] Failed to extract valid JSON from response")
            raise ValueError("Planner failed to generate valid JSON output")

        # Validate and create PlannerOutput
        try:
            output = PlannerOutput(**plan_json)
            logger.info(
                f"[PLANNER] Plan created for {output.target_restaurant} in {output.location}"
            )
            logger.info(f"[PLANNER] Intent: {output.intent}")
            logger.info(f"[PLANNER] Generated {len(output.search_queries)} queries")
            return output
        except ValidationError as e:
            logger.error(f"[PLANNER] Validation error: {e}")
            raise ValueError(f"Planner output validation failed: {e}")

    def plan_with_confirmation(self, user_query: str) -> PlannerOutput:
        """
        Generate a research plan with human confirmation.
        
        This method:
        1. Generates initial plan from user query
        2. Detects cuisine type via Tavily web search
        3. Presents plan to user for confirmation
        4. Allows editing until user approves
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            Confirmed PlannerOutput ready for research
        """
        console.print("\n[bold cyan]Analyzing your request...[/bold cyan]\n")
        
        # Step 1: Generate initial plan
        plan = self.plan(user_query)
        
        # Step 2: Detect cuisine type via web search
        console.print("[dim]Detecting cuisine type from web sources...[/dim]")
        cuisine = self._detect_cuisine_from_web(plan.target_restaurant, plan.location)
        plan.cuisine_type = cuisine
        
        # Step 3: Update first search query with cuisine type
        if cuisine and cuisine.lower() != "unknown":
            plan.search_queries[0] = f"{cuisine} restaurants competitors near {plan.location}"
        
        # Step 4: Confirmation loop
        while True:
            self._display_plan_for_confirmation(plan)
            
            choice = Prompt.ask(
                "\n[bold cyan]Your choice[/bold cyan]",
                choices=["y", "e", "q"],
                default="y"
            ).lower()
            
            if choice == "y":
                console.print("\n[bold green]✓ Plan confirmed! Starting research...[/bold green]\n")
                return plan
            elif choice == "e":
                plan = self._interactive_edit_plan(plan)
            elif choice == "q":
                console.print("\n[yellow]Research cancelled by user.[/yellow]")
                raise KeyboardInterrupt("User cancelled research")
    
    def _detect_cuisine_from_web(self, restaurant: str, location: str) -> str:
        """
        Detect cuisine type using Tavily web search + LLM extraction.
        
        Always searches the web first, then uses LLM to extract cuisine
        from the search results (not from LLM's world knowledge).
        
        Args:
            restaurant: Restaurant name
            location: City, State
            
        Returns:
            Detected cuisine type or "unknown"
        """
        if not self.tavily:
            logger.warning("[PLANNER] Tavily not available for cuisine detection")
            return "unknown"
        
        try:
            # Search for restaurant info
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
                logger.warning("[PLANNER] No search results for cuisine detection")
                return "unknown"
            
            # Combine search results
            search_context = "\n\n".join([
                f"Source: {r.get('title', 'Unknown')}\nContent: {r.get('content', '')}"
                for r in results[:5]
            ])
            
            # Ask LLM to extract cuisine FROM the search results
            extraction_prompt = f"""Based on the following web search results, identify the cuisine type of "{restaurant}" in {location}.

SEARCH RESULTS:
{search_context}

RULES:
- Extract the cuisine type ONLY from the search results above
- Do NOT use your own knowledge about the restaurant
- If results mention specific regional cuisine (e.g., "South Indian", "Tex-Mex", "Sichuan"), include that detail
- If cuisine is unclear from results, respond with "unknown"
- Common cuisine types: Indian, Italian, Mexican, Chinese, Japanese, Thai, American, Mediterranean, French, Korean, Vietnamese, Greek, etc.

Respond with ONLY the cuisine type (e.g., "Indian", "South Indian", "Italian", "Mexican", "American", "unknown").
Do not include any explanation or other text."""

            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            cuisine = response.content.strip().strip('"').strip("'")
            
            # Clean up the response
            cuisine = cuisine.split("\n")[0].strip()  # Take only first line
            if len(cuisine) > 50:  # If response is too long, it's probably an explanation
                cuisine = "unknown"
            
            logger.info(f"[PLANNER] Detected cuisine from web: {cuisine}")
            return cuisine
            
        except Exception as e:
            logger.error(f"[PLANNER] Error detecting cuisine: {e}")
            return "unknown"
    
    def _display_plan_for_confirmation(self, plan: PlannerOutput) -> None:
        """
        Display the research plan in a formatted Rich panel for user confirmation.
        
        Args:
            plan: The PlannerOutput to display
        """
        # Create a table for plan details
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Field", style="cyan", width=15)
        table.add_column("Value", style="white")
        
        table.add_row("Restaurant", plan.target_restaurant)
        table.add_row("Location", plan.location)
        table.add_row("Cuisine", plan.cuisine_type or "unknown")
        table.add_row("Goal", plan.intent)
        
        # Create queries section
        queries_text = "\n".join([
            f"  {i+1}. {q}" for i, q in enumerate(plan.search_queries)
        ])
        
        # Build the full content
        content = f"""
{table.__rich_console__(console, console.options).__next__()}

[bold]Research Queries:[/bold]
{queries_text}

[dim]─────────────────────────────────────────────────────────[/dim]

[bold]Options:[/bold]
  [green][Y][/green] Proceed with research
  [yellow][E][/yellow] Edit plan
  [red][Q][/red] Quit
"""
        
        # Display in a panel
        panel = Panel(
            content,
            title="[bold white]RESEARCH PLAN[/bold white]",
            border_style="cyan",
            padding=(1, 2),
        )
        
        # Actually render the table properly
        console.print("\n")
        console.print(Panel.fit(
            self._build_plan_display(plan),
            title="[bold white]RESEARCH PLAN[/bold white]",
            border_style="cyan",
            padding=(1, 2),
        ))
    
    def _build_plan_display(self, plan: PlannerOutput) -> str:
        """Build the plan display string."""
        lines = [
            f"[cyan]Restaurant:[/cyan]  {plan.target_restaurant}",
            f"[cyan]Location:[/cyan]    {plan.location}",
            f"[cyan]Cuisine:[/cyan]     {plan.cuisine_type or 'unknown'}",
            f"[cyan]Goal:[/cyan]        {plan.intent}",
            "",
            "[bold]Research Queries:[/bold]",
        ]
        
        for i, query in enumerate(plan.search_queries, 1):
            lines.append(f"  {i}. {query}")
        
        lines.extend([
            "",
            "[dim]─────────────────────────────────────────────────────────[/dim]",
            "",
            "[bold]Options:[/bold]",
            "  [green][Y][/green] Proceed with research",
            "  [yellow][E][/yellow] Edit plan",
            "  [red][Q][/red] Quit",
        ])
        
        return "\n".join(lines)
    
    def _interactive_edit_plan(self, plan: PlannerOutput) -> PlannerOutput:
        """
        Allow user to interactively edit plan fields.
        
        Args:
            plan: Current PlannerOutput
            
        Returns:
            Updated PlannerOutput
        """
        console.print("\n[bold yellow]What would you like to edit?[/bold yellow]")
        console.print("  [1] Restaurant name")
        console.print("  [2] Location")
        console.print("  [3] Cuisine type")
        console.print("  [4] Research goal")
        console.print("  [5] Search queries")
        console.print("  [A] All fields")
        console.print("  [B] Back (no changes)")
        
        choice = Prompt.ask(
            "\n[cyan]Choice[/cyan]",
            choices=["1", "2", "3", "4", "5", "a", "b"],
            default="b"
        ).lower()
        
        if choice == "b":
            return plan
        
        # Create a mutable copy
        plan_dict = plan.model_dump()
        
        if choice in ["1", "a"]:
            new_val = Prompt.ask(
                f"[cyan]Restaurant name[/cyan] (current: {plan.target_restaurant})",
                default=plan.target_restaurant
            )
            plan_dict["target_restaurant"] = new_val
        
        if choice in ["2", "a"]:
            new_val = Prompt.ask(
                f"[cyan]Location[/cyan] (current: {plan.location})",
                default=plan.location
            )
            plan_dict["location"] = new_val
        
        if choice in ["3", "a"]:
            new_val = Prompt.ask(
                f"[cyan]Cuisine type[/cyan] (current: {plan.cuisine_type})",
                default=plan.cuisine_type or "unknown"
            )
            plan_dict["cuisine_type"] = new_val
            # Update first query with new cuisine
            if new_val.lower() != "unknown":
                plan_dict["search_queries"][0] = f"{new_val} restaurants competitors near {plan_dict['location']}"
        
        if choice in ["4", "a"]:
            new_val = Prompt.ask(
                f"[cyan]Research goal[/cyan] (current: {plan.intent})",
                default=plan.intent
            )
            plan_dict["intent"] = new_val
        
        if choice in ["5", "a"]:
            console.print("\n[dim]Enter new queries (press Enter to keep current):[/dim]")
            new_queries = []
            for i, query in enumerate(plan.search_queries):
                new_val = Prompt.ask(
                    f"  [cyan]Query {i+1}[/cyan]",
                    default=query
                )
                new_queries.append(new_val)
            plan_dict["search_queries"] = new_queries
        
        # Create new PlannerOutput
        return PlannerOutput(**plan_dict)

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from LLM response text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown
        import re

        json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end])
            except json.JSONDecodeError:
                pass

        return None

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        LangGraph-compatible call interface.
        
        Args:
            state: Current agent state with user_query
            
        Returns:
            Updated state with planner_output
        """
        user_query = state.get("user_query", "")
        if not user_query:
            raise ValueError("No user_query found in state")

        try:
            output = self.plan(user_query)
            return {
                **state,
                "planner_output": output,
                "current_node": "researcher",
            }
        except Exception as e:
            logger.error(f"[PLANNER] Error: {e}")
            return {
                **state,
                "error_log": state.get("error_log", []) + [f"Planner error: {str(e)}"],
                "current_node": "error",
            }
