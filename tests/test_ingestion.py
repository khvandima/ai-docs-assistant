import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.ingestion import split_markdown, extract_heading


# --- Unit tests (no external dependencies) ---

def test_extract_heading_with_h2():
    text = "## Что такое RAG\nRAG это архитектурный паттерн..."
    assert extract_heading(text) == "Что такое RAG"


def test_extract_heading_with_h1():
    text = "# Введение\nЭто введение в курс."
    assert extract_heading(text) == "Введение"


def test_extract_heading_no_heading():
    text = "Просто текст без заголовка"
    assert extract_heading(text) == ""


def test_split_markdown_basic():
    text = "## Раздел 1\nТекст первого раздела.\n\n## Раздел 2\nТекст второго раздела."
    chunks = split_markdown(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_split_markdown_empty():
    chunks = split_markdown("")
    assert chunks == []


def test_split_markdown_large_chunk(tmp_path):
    """Большой чанк должен быть разбит на подчанки."""
    long_text = "## Большой раздел\n" + ("Очень длинный текст. " * 200)
    chunks = split_markdown(long_text)
    assert len(chunks) > 1


# --- Integration tests (require running Qdrant) ---

@pytest.mark.integration
def test_ingest_file(tmp_path):
    """Тест индексации реального файла."""
    md_file = tmp_path / "test.md"
    md_file.write_text(
        "## Тестовый раздел\nЭто тестовый контент для проверки индексации.",
        encoding="utf-8",
    )

    from app.ingestion import ingest_file, delete_file
    count = ingest_file(md_file)
    assert count > 0

    # Cleanup
    delete_file("test.md")


@pytest.mark.integration
def test_delete_file(tmp_path):
    """Тест удаления файла из базы."""
    md_file = tmp_path / "delete_test.md"
    md_file.write_text("## Удаляемый раздел\nКонтент для удаления.", encoding="utf-8")

    from app.ingestion import ingest_file, delete_file
    ingest_file(md_file)
    delete_file("delete_test.md")  # не должно падать с ошибкой