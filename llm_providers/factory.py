"""Create a LangChain chat model for the configured provider."""

from langchain_core.language_models import BaseChatModel

from config import settings


def get_llm(provider: str | None = None, model: str | None = None, **kwargs) -> BaseChatModel:
    provider = provider or settings.llm_provider
    params = {
        "model": model or settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        **kwargs,
    }

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**params)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**params)

    raise ValueError(f"Unsupported LLM provider: {provider!r}")
