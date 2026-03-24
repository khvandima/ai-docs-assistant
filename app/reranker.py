from sentence_transformers import CrossEncoder
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

reranker = CrossEncoder(
    model_name=settings.RERANK_MODEL,
    max_length=512,
    device="cpu",
)


def rerank(question: str, chunks: list[str]) -> list[str]:
    """
    Переранжирует чанки по релевантности к вопросу.
    Возвращает top_k наиболее релевантных чанков.
    """
    if not chunks:
        return []

    pairs = [(question, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info(
        f"Reranking: {len(chunks)} чанков → топ {settings.RERANK_TOP_K}, "
        f"лучший score: {ranked[0][1]:.3f}"
    )

    return [chunk for chunk, _ in ranked[:settings.RERANK_TOP_K]]
