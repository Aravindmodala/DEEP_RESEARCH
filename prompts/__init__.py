"""System prompts for all agent nodes."""

from .planner import PLANNER_SYSTEM_PROMPT
from .researcher import RESEARCHER_SYSTEM_PROMPT
from .critic import CRITIC_SYSTEM_PROMPT
from .report import REPORT_SYSTEM_PROMPT

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "RESEARCHER_SYSTEM_PROMPT",
    "CRITIC_SYSTEM_PROMPT",
    "REPORT_SYSTEM_PROMPT",
]


