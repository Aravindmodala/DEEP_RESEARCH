import os

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.planner import plan_research
from agents.researcher import research_sections
from agents.critic import critique_and_decide
from agents.reporter import generate_report
from state import AgentState
from logger import logger

workflow = StateGraph(AgentState)

workflow.add_node("planner", plan_research)
workflow.add_node("researcher", research_sections)
workflow.add_node("critic", critique_and_decide)
workflow.add_node("reporter", generate_report)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "critic")

# Conditional: if critique says "incomplete" → back to researcher (bounded)
def _route_after_critic(x: dict) -> str:
    critique = (x.get("critique") or "")
    iteration = int(x.get("iteration") or 0)
    max_iters = int(os.getenv("MAX_ITERATIONS", "2"))

    if iteration >= max_iters:
        logger.warning("ROUTER max iterations reached ({}) forcing done", max_iters)
        return "done"

    route = "continue" if "incomplete" in critique.lower() else "done"
    logger.info("ROUTER critic -> {} | critique_snippet={}", route, critique[:120])
    return route

workflow.add_conditional_edges(
    "critic",
    _route_after_critic,
    {"continue": "researcher", "done": "reporter"},
)
workflow.add_edge("reporter", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)