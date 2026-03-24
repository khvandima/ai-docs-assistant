import pytest
from unittest.mock import patch, MagicMock


def test_agent_state_structure():
    """State должен содержать все нужные поля."""
    from app.agent import AgentState
    from typing import get_type_hints
    hints = get_type_hints(AgentState)
    assert "question" in hints
    assert "context" in hints
    assert "sources" in hints
    assert "answer" in hints
    assert "route" in hints
    assert "messages" in hints


def test_route_question_rag():
    """Роутер должен вернуть retrieve для rag."""
    from app.agent import route_question, AgentState
    state = AgentState(
        question="Что такое RAG?",
        context=[],
        sources=[],
        answer="",
        route="rag",
        relevance="",
        retry_count=0,
        messages=[],
    )
    assert route_question(state) == "retrieve"


def test_route_question_chat():
    """Роутер должен вернуть direct_answer для chat."""
    from app.agent import route_question, AgentState
    state = AgentState(
        question="Привет",
        context=[],
        sources=[],
        answer="",
        route="chat",
        relevance="",
        retry_count=0,
        messages=[],
    )
    assert route_question(state) == "direct_answer"


def test_route_after_grade_relevant():
    """После grade relevant должен идти generate."""
    from app.agent import route_after_grade, AgentState
    state = AgentState(
        question="Вопрос",
        context=["контекст"],
        sources=["file.md"],
        answer="",
        route="rag",
        relevance="relevant",
        retry_count=1,
        messages=[],
    )
    assert route_after_grade(state) == "generate"


def test_route_after_grade_irrelevant():
    """После grade irrelevant должен идти no_answer."""
    from app.agent import route_after_grade, AgentState
    state = AgentState(
        question="Вопрос",
        context=[],
        sources=[],
        answer="",
        route="rag",
        relevance="irrelevant",
        retry_count=1,
        messages=[],
    )
    assert route_after_grade(state) == "no_answer"


def test_graph_compiled():
    """Граф должен успешно компилироваться."""
    from app.agent import agent
    assert agent is not None


@pytest.mark.integration
def test_agent_chat_route():
    """Приветствие должно идти в direct_answer без обращения к базе."""
    from app.agent import agent
    config = {"configurable": {"thread_id": "test-chat"}}
    result = agent.invoke({"question": "Привет"}, config=config)
    assert "answer" in result
    assert len(result["answer"]) > 0