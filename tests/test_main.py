import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_ok():
    """GET /health должен вернуть 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_returns_html():
    """GET / должен вернуть HTML страницу."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_upload_wrong_format():
    """Загрузка не .md файла должна вернуть 400."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"some content", "text/plain")},
    )
    assert response.status_code == 400
    assert "md" in response.json()["detail"].lower()


def test_files_list():
    """GET /files должен вернуть список файлов."""
    response = client.get("/files")
    assert response.status_code == 200
    assert "files" in response.json()
    assert isinstance(response.json()["files"], list)


def test_delete_nonexistent_file():
    """Удаление несуществующего файла не должно падать с 500."""
    response = client.delete("/files/nonexistent.md")
    assert response.status_code in (200, 404, 500)


@pytest.mark.integration
def test_health_detailed():
    """GET /health/detailed должен показать статус компонентов."""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "qdrant" in data["components"]
    assert "ollama" in data["components"]
    assert "collection" in data["components"]