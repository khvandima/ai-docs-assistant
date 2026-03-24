from typing import TypedDict, Literal, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import QdrantClient, models

from app.config import settings
from app.embeddings import get_embeddings
from app.llm_factory import get_llm
from app.reranker import rerank
from app.sparse import sparse_model
from app.retry import llm_retry, qdrant_retry
from app.logger import get_logger

logger = get_logger(__name__)

client = QdrantClient(url=settings.qdrant_url)
llm = get_llm()

# Сколько последних сообщений оставлять без суммаризации
RECENT_MESSAGES_LIMIT = 6

# --- Промпты ---
ROUTER_SYSTEM = (
    "Ты определяешь нужно ли искать ответ в базе знаний курса. "
    "Ответь 'rag' если сообщение содержит любой вопрос о курсе, его содержании, темах, материалах, или любой технический вопрос. "
    "Ответь 'chat' ТОЛЬКО если это чистое приветствие ('привет', 'здравствуй', 'добрый день') или благодарность ('спасибо', 'благодарю') без вопроса. "
    "Если есть хоть малейшее сомнение — отвечай 'rag'. "
    "Отвечай строго одним словом: chat или rag."
)

SUMMARIZE_SYSTEM = (
    "Ты создаёшь краткое резюме диалога между пользователем и ассистентом.\n\n"
    "Правила:\n"
    "- Сохрани ключевые темы и факты которые обсуждались.\n"
    "- Не пересказывай каждый вопрос и ответ — выдели суть.\n"
    "- Максимум 5-7 предложений.\n"
    "- Пиши на том же языке что и диалог.\n"
    "- Если уже есть предыдущее резюме — объедини его с новыми сообщениями в одно резюме."
)

GENERATE_SYSTEM = (
    "Ты помощник по материалам курса по AI агентам.\n\n"

    "Думай пошагово, но пиши только финальный ответ — без промежуточных рассуждений.\n"
    "Если ответа в контексте нет — честно скажи об этом.\n\n"
    
    "Правила ответа:\n"
    "- Сначала дай прямой короткий ответ на вопрос.\n"
    "- Затем кратко объясни опираясь только на контекст.\n"
    "- Не пересказывай контекст целиком.\n"
    "- Отвечай на том же языке что и вопрос.\n\n"
    "- Отвечай строго на основе того что написано в контексте — не интерпретируй и не преуменьшай.\n"

    "Примеры:\n"
    "Вопрос: Что такое LangGraph?\n"
    "Ответ: LangGraph — библиотека для построения агентов в виде графа состояний. "
    "В отличие от простых цепочек, граф позволяет делать циклы и условные переходы между узлами.\n\n"

    "Вопрос: Есть ли в курсе материалы по математике?\n"
    "Ответ: Нет, математика в курсе не рассматривается. "
    "Курс сфокусирован на инженерии LLM — как строить агентов, работать с векторными базами и деплоить модели.\n\n"

    "Вопрос: Что такое RAG?\n"
    "Ответ: RAG (Retrieval-Augmented Generation) — паттерн где модель отвечает на основе найденных документов, "
    "а не только своих весов. Это снижает галлюцинации и позволяет работать с актуальными данными."
)

CHAT_SYSTEM = (
    "Ты дружелюбный помощник по курсу AI агентов. "
    "Отвечай естественно на приветствия и общие вопросы. "
    "Можешь кратко напомнить что ты помощник по материалам курса. "
    "Отвечай на том же языке что и вопрос."
)

NO_CONTEXT_MSG = (
    "В базе знаний пока нет материалов. "
    "Загрузи .md файлы курса через вкладку Files."
)

IRRELEVANT_MSG = (
    "Не нашёл релевантной информации по этому вопросу в материалах курса. "
    "Попробуй переформулировать вопрос."
)


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str          # резюме старых сообщений, "" если истории мало
    question: str
    context: list[str]
    sources: list[str]
    answer: str
    route: str


# --- Helpers ---
def build_history(state: AgentState) -> list[BaseMessage]:
    """
    Собирает историю для передачи в LLM:
    - Если есть summary — добавляет его как SystemMessage перед свежими сообщениями.
    - Берёт последние RECENT_MESSAGES_LIMIT сообщений.
    """
    recent = state["messages"][-RECENT_MESSAGES_LIMIT:] if state["messages"] else []

    if state.get("summary"):
        summary_msg = SystemMessage(
            content=f"Краткое резюме предыдущего разговора:\n{state['summary']}"
        )
        return [summary_msg] + recent

    return recent


# --- Узлы ---
@llm_retry
async def router(state: AgentState) -> dict:
    logger.info(f"Routing: {state['question']}")

    response = await llm.ainvoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=state["question"]),
    ])

    route = response.content.strip().lower()
    if route not in ("chat", "rag"):
        route = "rag"

    logger.info(f"Route: {route}")
    return {"route": route}


@qdrant_retry
async def retrieve(state: AgentState) -> dict:
    logger.info(f"Hybrid search: {state['question']}")

    embeddings = get_embeddings()
    dense_vector = embeddings.embed_query(state["question"])
    sparse_results = list(sparse_model.embed([state["question"]]))
    sparse_vector = sparse_results[0]

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=8,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
                using="sparse",
                limit=8,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=6,
    ).points

    context = [r.payload["text"] for r in results]
    sources = list({r.payload["source"] for r in results})

    logger.info(f"Found chunks: {len(context)}, sources: {sources}")
    return {"context": context, "sources": sources}


def rerank_chunks(state: AgentState) -> dict:
    logger.info("Reranking chunks")
    if not state["context"]:
        return {"context": []}
    reranked = rerank(state["question"], state["context"])
    return {"context": reranked}


@llm_retry
async def generate(state: AgentState) -> dict:
    if not state.get("context"):
        logger.info("Empty context — returning no_context message")
        answer = NO_CONTEXT_MSG
        return {
            "answer": answer,
            "sources": [],
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
        }

    logger.info("Generating answer")
    context_str = "\n\n".join(state["context"])
    history = build_history(state)

    messages = [SystemMessage(content=GENERATE_SYSTEM)]
    messages.extend(history)
    messages.append(HumanMessage(
        content=f"Контекст из курса:\n{context_str}\n\nВопрос: {state['question']}"
    ))

    response = await llm.ainvoke(messages)
    answer = response.content

    logger.info("Answer generated")
    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ],
    }


@llm_retry
async def direct_answer(state: AgentState) -> dict:
    logger.info("Direct answer without retrieval")

    history = build_history(state)

    messages = [SystemMessage(content=CHAT_SYSTEM)]
    messages.extend(history)
    messages.append(HumanMessage(content=state["question"]))

    response = await llm.ainvoke(messages)
    answer = response.content

    return {
        "answer": answer,
        "sources": [],
        "context": [],
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ],
    }


def no_context(state: AgentState) -> dict:
    answer = IRRELEVANT_MSG
    return {
        "answer": answer,
        "sources": [],
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ],
    }


@llm_retry
async def summarize(state: AgentState) -> dict:
    """
    Суммаризирует историю когда messages > RECENT_MESSAGES_LIMIT.
    Оставляет последние RECENT_MESSAGES_LIMIT сообщений,
    остальные сжимает в summary и удаляет из messages.
    """
    messages = state["messages"]
    old_messages = messages[:-RECENT_MESSAGES_LIMIT]
    recent_messages = messages[-RECENT_MESSAGES_LIMIT:]

    # Форматируем старые сообщения в текст для LLM
    dialogue_lines = []
    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            dialogue_lines.append(f"Пользователь: {msg.content}")
        elif isinstance(msg, AIMessage):
            dialogue_lines.append(f"Ассистент: {msg.content}")
    dialogue_text = "\n".join(dialogue_lines)

    # Если уже есть summary — просим объединить с новыми сообщениями
    user_content = (
        f"Предыдущее резюме:\n{state['summary']}\n\nНовые сообщения:\n{dialogue_text}"
        if state.get("summary")
        else f"Диалог:\n{dialogue_text}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=SUMMARIZE_SYSTEM),
        HumanMessage(content=user_content),
    ])

    new_summary = response.content.strip()
    logger.info(f"Summary updated, collapsed {len(old_messages)} messages")

    # Заменяем messages только свежими — старые теперь в summary
    return {
        "summary": new_summary,
        "messages": recent_messages,
    }


# --- Условные рёбра ---
def route_question(state: AgentState) -> Literal["retrieve", "direct_answer"]:
    return "retrieve" if state["route"] == "rag" else "direct_answer"


def route_after_rerank(state: AgentState) -> Literal["generate", "no_context"]:
    return "generate" if state.get("context") else "no_context"


def should_summarize(state: AgentState) -> Literal["summarize", "__end__"]:
    """Суммаризируем если накопилось больше RECENT_MESSAGES_LIMIT сообщений."""
    if len(state.get("messages", [])) > RECENT_MESSAGES_LIMIT:
        return "summarize"
    return "__end__"


# --- Граф ---
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank_chunks)
    graph.add_node("generate", generate)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("no_context", no_context)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_question,
        {"retrieve": "retrieve", "direct_answer": "direct_answer"},
    )

    graph.add_edge("retrieve", "rerank")

    graph.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {"generate": "generate", "no_context": "no_context"},
    )

    # После generate и direct_answer — проверяем нужна ли суммаризация
    graph.add_conditional_edges(
        "generate",
        should_summarize,
        {"summarize": "summarize", "__end__": END},
    )
    graph.add_conditional_edges(
        "direct_answer",
        should_summarize,
        {"summarize": "summarize", "__end__": END},
    )

    graph.add_edge("no_context", END)
    graph.add_edge("summarize", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


agent = build_graph()