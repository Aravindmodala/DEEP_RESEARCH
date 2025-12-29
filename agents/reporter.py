from tools.openai import get_openai_llm
from langchain_core.prompts import ChatPromptTemplate
from logger import logger, preview

with open("prompts/report_prompt.txt", "r") as f:
    REPORTER_PROMPT = f.read()

report_prompt = ChatPromptTemplate.from_messages([
    ("system", REPORTER_PROMPT),
    ("human", "Generate the final report.")
])

reporter_llm = get_openai_llm(temperature=0.3, max_tokens=4096)
reporter_chain = report_prompt | reporter_llm

def generate_report(state: dict) -> dict:
    findings = "\n\n".join([
        f"### {section}\n{summary}" for section, summary in state["research_findings"].items()
    ])
    logger.info("NODE reporter | findings_chars={}", len(findings))
    
    report = reporter_chain.invoke({
        "original_query": state["query"],
        "findings": findings
    })
    
    return {"final_report": report.content}
