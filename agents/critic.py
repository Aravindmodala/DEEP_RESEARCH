from tools.openai import get_openai_llm
from langchain_core.prompts import ChatPromptTemplate
from logger import logger, preview

with open("prompts/critic_prompt.txt", "r") as f:
    CRITIC_PROMPT = f.read()

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", CRITIC_PROMPT),
    ("human", "Evaluate these findings.")
])

critic_llm = get_openai_llm(temperature=0.0)
critic_chain = critic_prompt | critic_llm

def critique_and_decide(state: dict) -> dict:
    findings = state["messages"][-1].content if state["messages"] else "No findings yet."
    logger.info("NODE critic | iteration={}", state.get("iteration", 0))
    logger.debug("critic findings chars={} preview={}", len(findings), preview(findings, 2000))
    
    response = critic_chain.invoke({
        "original_query": state["query"],
        "findings": findings
    })
    
    critique = response.content.strip()
    logger.debug("critic raw output={}", preview(critique, 2000))
    
    updates = {
        "critique": critique,
        "iteration": state.get("iteration", 0) + 1
    }

    # NOTE: We intentionally do not mutate search_queries here.
    # The researcher already handles official-first extraction + fallback within each query.
    return updates

