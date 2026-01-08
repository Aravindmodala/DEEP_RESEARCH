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
        page_title="Deep Research – Restaurant Market Analysis",
        page_icon="🏦",
        layout="centered",
    )

    init_session_state()

    # Header
    st.title("🏦 Deep Research Agent")
    st.caption("Commercial banking-grade restaurant market analysis")

    # Chat history display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input (chat interface)
    if prompt := st.chat_input("Enter your research query (e.g., 'Analyze Chipotle in Austin, TX')"):
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
                                st.write("🔍 Researching competitors, menus, and market signals...")
                            elif node_name == "critic" and node_state.get("critic_output"):
                                st.write("✅ Validating research quality...")
                            elif node_name == "report" and node_state.get("report_output"):
                                st.write("📄 Generating final report...")
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
                            "content": f"❌ Error: {error_msg}"
                        })
                        
                except Exception as e:
                    error_msg = f"Research failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Error: {error_msg}"
                    })
                finally:
                    st.session_state.run_in_progress = False

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        Enter a natural language query to analyze a restaurant's market position.
        
        **Example queries:**
        - "Analyze Chipotle in Austin, TX for lending risk"
        - "Research expansion viability for Olive Garden in Seattle, WA"
        - "Competitive analysis of McDonald's in Miami, FL"
        """)
        
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
