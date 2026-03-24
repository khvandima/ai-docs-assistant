from functools import lru_cache
from typing import List
from langchain_core.embeddings import Embeddings
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


class FastEmbedWrapper(Embeddings):
    """
    Тонкая обёртка над fastembed.TextEmbedding.
    Нужна потому что FastEmbedEmbeddings из langchain-community
    не совместима с fastembed>=0.4 — self._model остаётся None.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        # query_embed даёт чуть лучше качество для поискового запроса
        results = list(self._model.query_embed([text]))
        return results[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    EMBED_PROVIDER=ollama    → OllamaEmbeddings (локально)
    EMBED_PROVIDER=fastembed → FastEmbedWrapper (для деплоя без Ollama)
    """
    provider = settings.EMBED_PROVIDER

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        logger.info(f"Embeddings: Ollama ({settings.OLLAMA_EMBED_MODEL})")
        return OllamaEmbeddings(
            base_url=settings.ollama_url,
            model=settings.OLLAMA_EMBED_MODEL,
        )

    elif provider == "fastembed":
        logger.info(f"Embeddings: FastEmbed ({settings.FASTEMBED_MODEL})")
        return FastEmbedWrapper(model_name=settings.FASTEMBED_MODEL)

    else:
        raise ValueError(
            f"Неизвестный EMBED_PROVIDER: '{provider}'. "
            f"Доступные: ollama, fastembed"
        )


# Обратная совместимость
embeddings = get_embeddings()