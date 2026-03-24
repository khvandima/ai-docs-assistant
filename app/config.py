from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LangSmith
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = ""

    # LLM Provider
    PROVIDER: str
    MODEL: str

    # Embeddings provider: "ollama" | "fastembed"
    EMBED_PROVIDER: str = "ollama"

    # Ollama (используется если EMBED_PROVIDER=ollama или PROVIDER=ollama)
    OLLAMA_HOST: str = ""
    OLLAMA_PORT: int = 11434
    OLLAMA_EMBED_MODEL: str = ""

    # FastEmbed (используется если EMBED_PROVIDER=fastembed, для деплоя без Ollama)
    # Рекомендуется: "intfloat/multilingual-e5-large" — совместима с multilingual-e5-large из Ollama
    FASTEMBED_MODEL: str = "intfloat/multilingual-e5-large"

    # API ключи — опциональные
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Qdrant
    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str

    # Embeddings
    VECTOR_SIZE: int

    # Ingestion
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # Retrieval
    SCORE_THRESHOLD: float

    # Reranking
    RERANK_MODEL: str
    RERANK_TOP_K: int

    @model_validator(mode="after")
    def validate_config(self):
        # LLM провайдер
        if self.PROVIDER == "ollama" and not self.OLLAMA_HOST:
            raise ValueError("OLLAMA_HOST обязателен для PROVIDER=ollama")
        if self.PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY обязателен для PROVIDER=openai")
        if self.PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY обязателен для PROVIDER=anthropic")
        if self.PROVIDER == "groq" and not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY обязателен для PROVIDER=groq")

        # Embeddings провайдер
        if self.EMBED_PROVIDER == "ollama" and not self.OLLAMA_HOST:
            raise ValueError("OLLAMA_HOST обязателен для EMBED_PROVIDER=ollama")
        if self.EMBED_PROVIDER not in ("ollama", "fastembed"):
            raise ValueError(
                f"Неизвестный EMBED_PROVIDER: '{self.EMBED_PROVIDER}'. "
                f"Доступные: ollama, fastembed"
            )

        return self

    @property
    def ollama_url(self) -> str:
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"


settings = Settings()