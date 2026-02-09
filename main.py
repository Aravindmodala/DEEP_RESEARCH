#!/usr/bin/env python3
"""
Deep Research Agent - Restaurant Market Analysis
================================================

A multi-agent system for comprehensive restaurant market analysis
designed for Commercial Banking decisions.

Usage:
    python main.py "Analyze Chipotle in Austin, TX for expansion lending"
    python main.py --interactive

Author: AI Engineer
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import get_config
from orchestrator import DeepResearchOrchestrator, format_report_as_markdown

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()


def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> None:
    """Configure logging for the application."""
    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # Add file handler if enabled
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
        )


def validate_environment() -> bool:
    """Validate required environment variables are set."""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API (or set ANTHROPIC_API_KEY)",
        "GOOGLE_MAPS_API_KEY": "Google Maps/Places API",
        "TAVILY_API_KEY": "Tavily Search API",
    }

    missing = []
    for var, desc in required_vars.items():
        if not os.getenv(var):
            # Check alternative
            if var == "OPENAI_API_KEY" and os.getenv("ANTHROPIC_API_KEY"):
                continue
            missing.append(f"  - {var}: {desc}")

    if missing:
        console.print(
            Panel(
                "[bold red]Missing Required Environment Variables[/bold red]\n\n"
                + "\n".join(missing)
                + "\n\n"
                "Please set these in your .env file or environment.",
                title="Configuration Error",
                border_style="red",
            )
        )
        return False
    return True


def run_research(query: str, output_file: str | None = None, skip_confirmation: bool = False) -> dict:
    """
    Run the deep research pipeline.

    Args:
        query: Natural language research query
        output_file: Optional path to save the report
        skip_confirmation: If True, skip interactive plan confirmation

    Returns:
        Final state with all agent outputs
    """
    console.print(
        Panel(
            f"[bold cyan]{query}[/bold cyan]",
            title="Research Query",
            border_style="cyan",
        )
    )

    # Initialize orchestrator
    config = get_config()
    orchestrator = DeepResearchOrchestrator(config, skip_confirmation=skip_confirmation)

    # Run with progress indication
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running deep research...", total=None)

        try:
            result = orchestrator.run(query)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            raise

        progress.update(task, description="[green]Research complete!")

    # Check for errors
    if result.get("error_log"):
        console.print(
            Panel(
                "\n".join(result["error_log"]),
                title="[bold red]Errors Encountered[/bold red]",
                border_style="red",
            )
        )

    # Display report if available
    report_output = result.get("report_output")
    if report_output:
        md_report = format_report_as_markdown(report_output)

        console.print("\n")
        console.print(Markdown(md_report))

        # Save report if requested
        if output_file:
            output_path = Path(output_file)
            
            # Save markdown
            output_path.with_suffix(".md").write_text(md_report, encoding="utf-8")
            console.print(f"\n[green]Report saved to:[/green] {output_path.with_suffix('.md')}")

            # Save JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"[green]Full data saved to:[/green] {json_path}")

    return result


def interactive_mode() -> None:
    """Run in interactive mode with multiple queries."""
    console.print(
        Panel(
            "[bold cyan]Deep Research Agent[/bold cyan]\n"
            "[dim]Restaurant Market Analysis for Commercial Banking[/dim]\n\n"
            "Enter your research queries below.\n"
            "Type [bold]'quit'[/bold] or [bold]'exit'[/bold] to end the session.\n"
            "Type [bold]'save <filename>'[/bold] after a query to save results.",
            title="Interactive Mode",
            border_style="cyan",
        )
    )

    last_result = None

    while True:
        try:
            query = console.input("\n[bold cyan]Query>[/bold cyan] ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if query.lower().startswith("save "):
                if last_result and last_result.get("report_output"):
                    filename = query[5:].strip()
                    md_report = format_report_as_markdown(last_result["report_output"])
                    Path(filename).with_suffix(".md").write_text(md_report, encoding="utf-8")
                    console.print(f"[green]Saved to {filename}.md[/green]")
                else:
                    console.print("[yellow]No report to save. Run a query first.[/yellow]")
                continue

            if query.lower() == "help":
                console.print(
                    Panel(
                        "**Example Queries:**\n\n"
                        "- Analyze Chipotle in Austin, TX for lending risk\n"
                        "- Research expansion viability for Olive Garden in Seattle, WA\n"
                        "- Competitive analysis of local Thai restaurants in Denver, CO\n"
                        "- Market research for Five Guys in Miami, FL\n\n"
                        "**Commands:**\n"
                        "- `save <filename>` - Save last report\n"
                        "- `quit` or `exit` - End session",
                        title="Help",
                    )
                )
                continue

            last_result = run_research(query)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            logger.exception("Error in interactive mode")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - Restaurant Market Analysis for Commercial Banking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Analyze Chipotle in Austin, TX for lending risk"
  python main.py "Research expansion viability for Olive Garden in Seattle, WA" -o report
  python main.py --interactive
        """,
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Research query (e.g., 'Analyze Chipotle in Austin, TX')",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (without extension)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip interactive plan confirmation (auto-accept)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, not args.no_log_file)

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Run appropriate mode
    if args.interactive:
        interactive_mode()
    elif args.query:
        run_research(args.query, args.output, skip_confirmation=args.no_confirm)
    else:
        # No query provided - prompt for input
        console.print(
            Panel(
                "[bold cyan]Deep Research Agent[/bold cyan]\n"
                "[dim]Restaurant Market Analysis for Commercial Banking[/dim]",
                border_style="cyan",
            )
        )
        console.print("\n[bold]Example queries:[/bold]")
        console.print("  • Analyze Chipotle in Austin, TX for lending risk")
        console.print("  • Research expansion viability for Olive Garden in Seattle, WA")
        console.print("  • Competitive analysis of McDonald's in Miami, FL")
        console.print("")
        
        try:
            query = console.input("[bold cyan]Enter your research query:[/bold cyan] ").strip()
            
            if query:
                run_research(query, args.output)
            else:
                console.print("[yellow]No query entered. Exiting.[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")


if __name__ == "__main__":
    main()

