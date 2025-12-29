from graph import app
from langchain_core.messages import HumanMessage
import uuid

thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": thread_id}}

query = input("Enter your research query: ")

for chunk in app.stream(
    {"messages": [HumanMessage(content=query)], "query": query, "iteration": 0},
    config,
    stream_mode="values"
):
    if "final_report" in chunk and chunk["final_report"]:
        print("\n\n=== FINAL REPORT ===\n")
        print(chunk["final_report"])
    elif chunk.get("messages"):
        chunk["messages"][-1].pretty_print()