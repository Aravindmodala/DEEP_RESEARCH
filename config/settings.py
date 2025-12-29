from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()

class Config(BaseModel):
    tavily_api_key: str = os.getenv("TAVILY_API_KEY")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.2")

config = Config()