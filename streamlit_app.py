"""
Streamlit Chat Interface for Deep Research Restaurant Market Analysis Agent.

Simple ChatGPT-like interface where users enter queries and receive banking-grade reports.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import streamlit as st

from orchestrator import DeepResearchOrchestrator, format_report_as_markdown


def init_session_state() -> None:
    """Initialize session state for chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "run_in_progress" not in st.session_state:
        st.session_state.run_in_progress = False


def main():
    st.set_page_config(
        page_title="Deep Research v2 - Restaurant Market Analysis",
        page_icon="🏦",
        layout="centered",
    )

    init_session_state()

    # Header
    st.title("🏦 Deep Research Agent v2")
    st.caption("Commercial banking-grade restaurant market analysis with SWOT, sentiment, and strategic recommendations")

    # Chat history display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input (chat interface)
    if prompt := st.chat_input("Enter your research query (e.g., 'Analyze 110 Grill in Canton, CT')"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Show assistant thinking
        with st.chat_message("assistant"):
            with st.status("🔍 Analyzing your request...", expanded=False) as status:
                st.write("🧠 Planning research strategy...")

                try:
                    orchestrator = DeepResearchOrchestrator()
                    st.session_state.run_in_progress = True

                    # Run the orchestrator and collect final state
                    final_state = None
                    for event in orchestrator.run_streaming(prompt):
                        # Update status as we progress
                        for node_name, node_state in event.items():
                            if node_name == "planner" and node_state.get("planner_output"):
                                st.write("✅ Research plan generated")
                            elif node_name == "researcher" and node_state.get("researcher_output"):
                                st.write("✅ Phase-gated research complete (6 phases)")
                            elif node_name == "analyst" and node_state.get("analyst_output"):
                                st.write("✅ Multi-pass analysis complete (SWOT, sentiment, menus, keywords, strategy)")
                            elif node_name == "critic" and node_state.get("critic_output"):
                                decision = node_state.get("critic_output", {}).get("decision", "?")
                                score = node_state.get("critic_output", {}).get("overall_quality_score", 0)
                                st.write(f"✅ Quality review: {decision} (score: {score:.2f})")
                            elif node_name == "report" and node_state.get("report_output"):
                                st.write("📄 13-section report generated")
                                final_state = node_state

                    status.update(label="✅ Analysis complete!", state="complete")

                    # Get the report
                    if final_state and final_state.get("report_output"):
                        report = final_state["report_output"]
                        report_md = format_report_as_markdown(report)

                        # Add assistant response to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": report_md
                        })

                        # Display the report
                        st.markdown(report_md)
                    else:
                        error_msg = "Research completed but no report was generated."
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {error_msg}"
                        })

                except Exception as e:
                    error_msg = f"Research failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {error_msg}"
                    })
                finally:
                    st.session_state.run_in_progress = False

    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Deep Research Agent v2** produces comprehensive reports with:
        - 6-phase research pipeline
        - SWOT analysis per restaurant
        - Multi-dimensional sentiment analysis
        - Item-by-item menu comparison
        - Review keyword frequency analysis
        - 7-pillar strategic recommendations
        - Inline source citations

        **Example queries:**
        - "Analyze 110 Grill in Canton, CT"
        - "Research expansion viability for Olive Garden in Seattle, WA"
        - "Competitive analysis of McDonald's in Miami, FL"
        """)

        if st.session_state.messages:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
