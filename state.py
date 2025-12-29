from typing import List, Dict, Annotated, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    plan: str
    search_queries: List[str]
    research_findings: Dict[str, str]
    critique: str
    final_report: str
    iteration: int

