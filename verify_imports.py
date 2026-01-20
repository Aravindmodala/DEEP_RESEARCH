from dotenv import load_dotenv
import os
import sys

# Load env if present
load_dotenv()

try:
    print("Importing config...")
    from config import AgentConfig
    
    print("Importing schemas...")
    from models.schemas import (
        PlannerOutput, ResearcherOutput, ReportOutput, 
        SentimentAnalysis, BusinessBackground
    )
    
    print("Importing agents...")
    from agents.planner import PlannerAgent
    from agents.researcher import ResearcherAgent
    from agents.critic import CriticAgent
    from agents.report import ReportAgent
    from orchestrator import DeepResearchOrchestrator
    
    print("All imports successful!")
    
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
