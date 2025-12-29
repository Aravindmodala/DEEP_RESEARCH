from langchain_openai import ChatOpenAI

from config.settings import config
from logger import logger


def get_openai_llm(temperature: float = 0.0, max_tokens: int = 4096) -> ChatOpenAI:
    logger.debug(
        "OpenAI LLM | model={} | temperature={} | max_tokens={}",
        config.openai_model,
        temperature,
        max_tokens,
    )
    return ChatOpenAI(
        model=config.openai_model,
        api_key=config.openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
