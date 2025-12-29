from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import config
import os

def get_gemini_llm(temperature=0.0, max_output_tokens=8192):
    return ChatGoogleGenerativeAI(
        model=config.gemini_model,
        google_api_key=config.google_api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
