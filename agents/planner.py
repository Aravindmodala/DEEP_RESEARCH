import json
from langchain_core.prompts import ChatPromptTemplate
from tools.openai import get_openai_llm
from logger import logger, preview

with open("prompts/planner_prompt.txt", "r") as f:
    PLANNER_PROMPT = f.read()

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_PROMPT),
    ("human", "{query}")
])

llm = get_openai_llm(temperature=0.1)

def plan_research(state: dict) -> dict:
    logger.info("NODE planner | query={}", state.get("query"))
    try:
        msgs = planner_prompt.format_messages(query=state["query"])
        logger.debug("planner prompt messages={}", preview(msgs, 3000))
    except Exception as e:
        logger.warning("planner prompt format failed: {}", e)

    chain = planner_prompt | llm
    response = chain.invoke({"query": state["query"]})
    
    content = response.content
    logger.debug("planner raw output={}", preview(content, 3000))
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
        
    try:
        search_queries = json.loads(content)["search_queries"]
    except Exception as e:
        logger.error("planner JSON parse failed: {} | content={}", e, preview(content, 2000))
        # fallback: basic 4 queries derived from the user query
        q = state.get("query", "")
        search_queries = [
            q,
            f"{q} official website",
            f"{q} PDF",
            f"{q} menu",
        ][:4]

    # normalize to exactly 4
    search_queries = [str(s).strip() for s in search_queries if str(s).strip()]
    if len(search_queries) > 4:
        search_queries = search_queries[:4]
    while len(search_queries) < 4:
        search_queries.append(search_queries[-1] if search_queries else state.get("query", ""))

    logger.info("planner search_queries={}", search_queries)
    return {"search_queries": search_queries, "plan": "\n".join(search_queries)}
