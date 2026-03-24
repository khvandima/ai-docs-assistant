from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.agent import agent
from app.config import settings
from app.ingestion import ingest_file, delete_file, reindex_file, ensure_collection
from app.logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")

    if settings.LANGCHAIN_TRACING_V2 == "true" and settings.LANGCHAIN_API_KEY:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        logger.info(f"LangSmith tracing enabled, project: {settings.LANGCHAIN_PROJECT}")

    ensure_collection()
    await auto_index_uploads()
    yield
    logger.info("Stopping application")


async def auto_index_uploads():
    """Индексирует .md файлы из uploads/ которых ещё нет в Qdrant."""
    files = list(UPLOAD_DIR.glob("*.md"))
    if not files:
        logger.info("Папка uploads/ пуста — индексация не нужна")
        return

    client = QdrantClient(url=settings.qdrant_url)

    for file_path in files:
        results = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchValue(value=file_path.name),
                )]
            ),
            limit=1,
        )

        if results[0]:
            logger.info(f"Уже проиндексирован: {file_path.name}")
        else:
            logger.info(f"Индексируем: {file_path.name}")
            ingest_file(file_path)


app = FastAPI(title="Course RAG Agent", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# --- Модели ---

class QuestionRequest(BaseModel):
    question: str
    thread_id: str = "default"


# --- Роуты ---

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/ask")
async def ask(body: QuestionRequest):
    logger.info(f"Вопрос: {body.question}")
    try:
        config = {"configurable": {"thread_id": body.thread_id}}
        result = await agent.ainvoke({"question": body.question}, config)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }
    except Exception as e:
        # logger.error(f"Ошибка агента: {e}")
        logger.error(f"Ошибка агента: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Ошибка агента")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Только .md файлы")

    logger.info(f"Загрузка: {file.filename}")
    try:
        file_path = UPLOAD_DIR / file.filename
        file_path.write_bytes(await file.read())
        count = ingest_file(file_path)
        return {"filename": file.filename, "chunks": count}
    except Exception as e:
        logger.error(f"Ошибка загрузки {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка индексации")


@app.delete("/files/{filename}")
async def remove_file(filename: str):
    logger.info(f"Удаление: {filename}")
    try:
        delete_file(filename)
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        return {"deleted": filename}
    except Exception as e:
        logger.error(f"Ошибка удаления {filename}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка удаления")


@app.post("/files/{filename}/reindex")
async def reindex(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    logger.info(f"Переиндексация: {filename}")
    try:
        count = reindex_file(file_path)
        return {"filename": filename, "chunks": count}
    except Exception as e:
        logger.error(f"Ошибка переиндексации {filename}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка переиндексации")


@app.get("/files")
async def list_files():
    files = [f.name for f in UPLOAD_DIR.glob("*.md")]
    return {"files": files}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/detailed")
async def health_detailed():
    result = {
        "status": "ok",
        "components": {
            "qdrant": {"status": "unknown"},
            "collection": {"status": "unknown"},
        },
    }

    # Qdrant
    try:
        client = QdrantClient(url=settings.qdrant_url)
        collections = client.get_collections()
        result["components"]["qdrant"] = {"status": "ok"}

        names = [c.name for c in collections.collections]
        if settings.QDRANT_COLLECTION_NAME in names:
            info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
            result["components"]["collection"] = {
                "status": "ok",
                "vectors_count": info.vectors_count,
            }
        else:
            result["components"]["collection"] = {"status": "empty"}

    except Exception as e:
        result["status"] = "degraded"
        result["components"]["qdrant"] = {"status": "error", "detail": str(e)}
        result["components"]["collection"] = {"status": "unknown"}

    # Ollama — проверяем только если он используется
    if settings.EMBED_PROVIDER == "ollama" or settings.PROVIDER == "ollama":
        result["components"]["ollama"] = {"status": "unknown"}
        try:
            import httpx
            response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                ollama_models = [m["name"] for m in response.json().get("models", [])]
                result["components"]["ollama"] = {
                    "status": "ok",
                    "models": ollama_models,
                }
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            result["status"] = "degraded"
            result["components"]["ollama"] = {"status": "error", "detail": str(e)}

    return result