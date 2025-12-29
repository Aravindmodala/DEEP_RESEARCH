import os
import sys
from typing import Any

from loguru import logger


def preview(value: Any, limit: int = 2000) -> str:
    """Safe string preview for logs (prevents huge payloads)."""
    try:
        s = value if isinstance(value, str) else repr(value)
    except Exception:
        s = "<unreprable>"
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n\n... [truncated, {len(s)} chars total]"


# Remove default logger
logger.remove()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "agent.log")

# Add a formatted console logger
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
)

# Also log to a file (always keep DEBUG for deep inspection)
logger.add(LOG_FILE, rotation="10 MB", level="DEBUG")

