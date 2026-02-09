"""System prompts for all agent nodes."""

from .planner import PLANNER_SYSTEM_PROMPT
from .researcher import RESEARCHER_SYSTEM_PROMPT
from .analyst import ANALYST_SYSTEM_PROMPT
from .critic import CRITIC_SYSTEM_PROMPT
from .report import REPORT_SECTION_PROMPTS

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "RESEARCHER_SYSTEM_PROMPT",
    "ANALYST_SYSTEM_PROMPT",
    "CRITIC_SYSTEM_PROMPT",
    "REPORT_SECTION_PROMPTS",
]
