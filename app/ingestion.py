import uuid
from pathlib import Path

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.embeddings import get_embeddings
from app.sparse import sparse_model  # единственный экземпляр
from app.retry import qdrant_retry
from app.logger import get_logger

logger = get_logger(__name__)

client = QdrantClient(url=settings.qdrant_url)

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)


def extract_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def ensure_collection() -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        logger.info(f"Коллекция создана: {settings.QDRANT_COLLECTION_NAME}")


def split_markdown(text: str) -> list[str]:
    header_chunks = header_splitter.split_text(text)
    result = []
    for chunk in header_chunks:
        if len(chunk.page_content) > settings.CHUNK_SIZE:
            sub_chunks = char_splitter.split_text(chunk.page_content)
            result.extend(sub_chunks)
        else:
            result.append(chunk.page_content)
    return result


@qdrant_retry
def ingest_file(file_path: str | Path) -> int:
    path = Path(file_path)
    logger.info(f"Индексация: {path.name}")

    try:
        ensure_collection()

        text = path.read_text(encoding="utf-8")
        texts = split_markdown(text)

        if not texts:
            logger.warning(f"Файл пустой или не удалось разбить: {path.name}")
            return 0

        embeddings = get_embeddings()
        dense_vectors = embeddings.embed_documents(texts)
        sparse_vectors = list(sparse_model.embed(texts))

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vector,
                    "sparse": SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist(),
                    ),
                },
                payload={
                    "text": chunk_text,
                    "source": path.name,
                    "chunk_index": i,
                    "heading": extract_heading(chunk_text),
                },
            )
            for i, (chunk_text, dense_vector, sv) in enumerate(
                zip(texts, dense_vectors, sparse_vectors)
            )
        ]

        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
        )

        logger.info(f"Проиндексирован: {path.name}, чанков: {len(points)}")
        return len(points)

    except Exception as e:
        logger.error(f"Ошибка индексации {path.name}: {e}")
        raise


@qdrant_retry
def delete_file(filename: str) -> None:
    logger.info(f"Удаление чанков: {filename}")
    try:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename),
                    )
                ]
            ),
        )
        logger.info(f"Удалено: {filename}")
    except Exception as e:
        logger.error(f"Ошибка удаления {filename}: {e}")
        raise


def reindex_file(file_path: str | Path) -> int:
    path = Path(file_path)
    logger.info(f"Переиндексация: {path.name}")
    delete_file(path.name)
    return ingest_file(path)