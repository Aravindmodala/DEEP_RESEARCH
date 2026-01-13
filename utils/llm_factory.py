"""
LLM Factory - Single source of truth for LLM provider initialization.

This module provides a centralized way to create LLM instances,
making it easy to switch between providers (OpenAI, Anthropic, Gemini/Vertex, etc.)
without modifying agent code.

Supported Providers:
- openai: OpenAI GPT models (gpt-4o, gpt-4-turbo, etc.)
- anthropic: Anthropic Claude models (claude-3-opus, claude-3-sonnet, etc.)
- vertex: Google Vertex AI / Gemini models (gemini-1.5-pro, gemini-1.5-flash, etc.)
- ollama: Local Ollama models (llama3, mistral, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.language_models import BaseChatModel
from loguru import logger

if TYPE_CHECKING:
    from config import AgentConfig

# Supported provider types
LLMProvider = Literal["openai", "anthropic", "vertex", "gemini", "ollama"]

SUPPORTED_PROVIDERS: dict[str, dict] = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "requires": ["OPENAI_API_KEY"],
    },
    "anthropic": {
        "name": "Anthropic",
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "requires": ["ANTHROPIC_API_KEY"],
    },
    "vertex": {
        "name": "Google Vertex AI",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        "requires": ["GCP_PROJECT_ID"],
    },
    "gemini": {
        "name": "Google Gemini (via Vertex)",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        "requires": ["GCP_PROJECT_ID"],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "models": ["llama3", "llama3:70b", "mistral", "mixtral", "phi3"],
        "requires": [],
    },
}


def get_supported_providers() -> dict[str, dict]:
    """Get information about supported LLM providers."""
    return SUPPORTED_PROVIDERS


def create_llm(
    config: "AgentConfig",
    temperature: float | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Factory function to create LLM instances.
    
    This is the SINGLE SOURCE OF TRUTH for LLM initialization.
    All agents should use this function instead of initializing LLMs directly.
    
    Args:
        config: Agent configuration with provider settings
        temperature: Override temperature (uses config default if None)
        streaming: Enable streaming responses
        
    Returns:
        Configured LLM instance (BaseChatModel)
        
    Raises:
        ValueError: If provider is not supported or required config is missing
        ImportError: If provider package is not installed
        
    Example:
        >>> from config import get_config
        >>> from utils import create_llm
        >>> 
        >>> config = get_config()
        >>> llm = create_llm(config)
        >>> llm = create_llm(config, temperature=0.0)  # Override temp
    """
    provider = config.llm_provider.lower()
    temp = temperature if temperature is not None else config.llm_temperature
    model = config.llm_model

    logger.debug(f"Creating LLM: provider={provider}, model={model}, temp={temp}")

    match provider:
        case "openai":
            return _create_openai(config, model, temp, streaming)
        
        case "anthropic":
            return _create_anthropic(config, model, temp, streaming)
        
        case "vertex" | "gemini":
            return _create_vertex(config, model, temp, streaming)
        
        case "ollama":
            return _create_ollama(config, model, temp, streaming)
        
        case _:
            supported = ", ".join(SUPPORTED_PROVIDERS.keys())
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported providers: {supported}"
            )


def _create_openai(
    config: "AgentConfig",
    model: str,
    temperature: float,
    streaming: bool,
) -> BaseChatModel:
    """Create OpenAI ChatGPT instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is required for OpenAI provider. "
            "Install with: pip install langchain-openai"
        ) from e

    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=config.openai_api_key,
        streaming=streaming,
    )


def _create_anthropic(
    config: "AgentConfig",
    model: str,
    temperature: float,
    streaming: bool,
) -> BaseChatModel:
    """Create Anthropic Claude instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError(
            "langchain-anthropic is required for Anthropic provider. "
            "Install with: pip install langchain-anthropic"
        ) from e

    if not config.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=config.anthropic_api_key,
        streaming=streaming,
    )


def _create_vertex(
    config: "AgentConfig",
    model: str,
    temperature: float,
    streaming: bool,
) -> BaseChatModel:
    """Create Google Vertex AI / Gemini instance."""
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError as e:
        raise ImportError(
            "langchain-google-vertexai is required for Vertex/Gemini provider. "
            "Install with: pip install langchain-google-vertexai"
        ) from e

    if not config.gcp_project_id:
        raise ValueError(
            "GCP_PROJECT_ID is required for Vertex/Gemini provider. "
            "Also ensure you have authenticated with: gcloud auth application-default login"
        )

    return ChatVertexAI(
        model_name=model,
        temperature=temperature,
        project=config.gcp_project_id,
        location=config.gcp_location,
        streaming=streaming,
    )


def _create_ollama(
    config: "AgentConfig",
    model: str,
    temperature: float,
    streaming: bool,
) -> BaseChatModel:
    """Create Ollama (local) instance."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "langchain-ollama is required for Ollama provider. "
            "Install with: pip install langchain-ollama"
        ) from e

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=config.ollama_base_url,
    )


