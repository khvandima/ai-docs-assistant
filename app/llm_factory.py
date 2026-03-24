from langchain_core.language_models import BaseChatModel
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


def get_llm() -> BaseChatModel:
    provider = settings.PROVIDER
    model = settings.MODEL

    logger.info(f"Инициализация LLM: провайдер={provider}, модель={model}")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_url,
            model=model,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=model,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            model=model,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=model,
        )

    else:
        raise ValueError(
            f"Неизвестный провайдер: '{provider}'. "
            f"Доступные: ollama, openai, anthropic, groq"
        )