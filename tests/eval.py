# """
# RAG pipeline evaluation using Ragas.
# Run: python -m tests.eval
# """
# import asyncio
# from datasets import Dataset
# from ragas import evaluate
# from ragas.llms import llm_factory
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
# from ragas.run_config import RunConfig

# from app.agent import agent
# from app.config import settings
# from app.embeddings import embeddings
# from app.logger import get_logger

# logger = get_logger(__name__)

# # Ragas через Groq OpenAI-совместимый endpoint
# from openai import OpenAI as OpenAIClient

# groq_client = OpenAIClient(
#     api_key=settings.GROQ_API_KEY,
#     base_url="https://api.groq.com/openai/v1",
# )

# ragas_llm = llm_factory(
#     model="llama-3.1-8b-instant",
#     client=groq_client,
# )

# from ragas.embeddings import LangchainEmbeddingsWrapper
# from app.embeddings import embeddings

# ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# # Метрики инициализируем без аргументов — llm передаём через evaluate()
# faithfulness = Faithfulness()
# answer_relevancy = AnswerRelevancy()
# context_precision = ContextPrecision()
# context_recall = ContextRecall()

# # --- Test dataset based on real course materials ---
# TEST_DATASET = [
#     {
#         "question": "Чему посвящён этот курс?",
#         "ground_truth": (
#             "Курс посвящён инженерии LLM — написанию продуктивного безопасного кода на Python. "
#             "Не написанию loss-функций, а разработке реальных AI-систем."
#         ),
#     },
#     {
#         "question": "Что такое embedding?",
#         "ground_truth": (
#             "Embedding — это числовое представление текста в виде вектора, "
#             "где семантически близкие фразы находятся близко в векторном пространстве."
#         ),
#     },
#     {
#         "question": "Чем Qdrant отличается от Chroma?",
#         "ground_truth": (
#             "Chroma — встраиваемая БД для прототипов, не требует сервера. "
#             "Qdrant — высокопроизводительная БД на Rust для продакшена, "
#             "поддерживает фильтрацию по метаданным и масштабирование."
#         ),
#     },
#     {
#         "question": "Что такое reranking и зачем он нужен?",
#         "ground_truth": (
#             "Reranking — повторная оценка релевантности найденных чанков. "
#             "Векторный поиск сортирует по эмбеддинговому сходству, "
#             "но reranker оценивает реальную смысловую релевантность. "
#             "FlashRank улучшает точность RAG на 15-30%."
#         ),
#     },
#     {
#         "question": "Какой оптимальный размер чанка для технической документации?",
#         "ground_truth": (
#             "300-500 токенов — золотая середина для технической документации. "
#             "Слишком длинный чанк перегружает LLM, слишком короткий теряет контекст."
#         ),
#     },
#     {
#         "question": "Что такое RAG?",
#         "ground_truth": (
#             "RAG (Retrieval-Augmented Generation) — архитектура где LLM отвечает "
#             "на основе найденных документов из векторной базы данных, "
#             "а не только своих весов. Это основа работы с приватными данными без дообучения."
#         ),
#     },
#     {
#         "question": "Зачем добавлять метаданные к чанкам?",
#         "ground_truth": (
#             "Метаданные нужны для фильтрации (искать только по определённому источнику), "
#             "отладки (понять откуда пришёл фрагмент) и безопасности "
#             "(исключить конфиденциальные данные из ответа)."
#         ),
#     },
#     {
#         "question": "Есть ли в курсе материалы по DataScience?",
#         "ground_truth": (
#             "Нет. Курс про инженерию LLM, а не DataScience. "
#             "Не нужно писать loss-функции — нужно писать продуктивный код на Python."
#         ),
#     },
# ]


# async def run_agent(question: str) -> tuple[str, list[str]]:
#     """Run agent and return answer and contexts."""
#     config = {"configurable": {"thread_id": f"eval-{question[:20]}"}}
#     result = agent.invoke({"question": question}, config=config)
#     return result["answer"], result.get("context", [])


# async def build_eval_dataset() -> Dataset:
#     """Run test questions through agent and collect dataset."""
#     questions = []
#     answers = []
#     contexts = []
#     ground_truths = []

#     for item in TEST_DATASET:
#         logger.info(f"Evaluating: {item['question']}")
#         answer, context = await run_agent(item["question"])

#         questions.append(item["question"])
#         answers.append(answer)
#         contexts.append(context if context else [""])
#         ground_truths.append(item["ground_truth"])

#     return Dataset.from_dict({
#         "question": questions,
#         "answer": answers,
#         "contexts": contexts,
#         "ground_truth": ground_truths,
#     })


# async def main():
#     logger.info("Starting RAG evaluation")

#     dataset = await build_eval_dataset()

#     logger.info("Running Ragas metrics")

#     results = evaluate(
#         dataset=dataset,
#         metrics=[
#             faithfulness,
#             answer_relevancy,
#             context_precision,
#             context_recall,
#         ],
#         llm=ragas_llm,
#         embeddings=ragas_embeddings,
#         run_config=RunConfig(max_workers=1, timeout=60),
#     )

#     print("\n" + "=" * 50)
#     print("RAG EVALUATION RESULTS")
#     print("=" * 50)
#     print(f"Faithfulness:      {results['faithfulness']:.3f}")
#     print(f"Answer relevancy:  {results['answer_relevancy']:.3f}")
#     print(f"Context precision: {results['context_precision']:.3f}")
#     print(f"Context recall:    {results['context_recall']:.3f}")
#     print("=" * 50)

#     results_df = results.to_pandas()
#     results_df.to_csv("eval_results.csv", index=False)
#     logger.info("Results saved to eval_results.csv")

#     return results


# if __name__ == "__main__":
#     asyncio.run(main())



# """
# RAG pipeline evaluation using Ragas.
# Run: python -m tests.eval
# """
# import asyncio
# from datasets import Dataset
# from ragas import evaluate
# from ragas.llms import llm_factory
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from ragas.metrics import Faithfulness, AnswerRelevancy
# from ragas.run_config import RunConfig
# from openai import OpenAI as OpenAIClient

# from app.agent import agent
# from app.config import settings
# from app.embeddings import embeddings
# from app.logger import get_logger

# logger = get_logger(__name__)

# groq_client = OpenAIClient(
#     api_key=settings.GROQ_API_KEY,
#     base_url="https://api.groq.com/openai/v1",
# )

# ragas_llm = llm_factory(model="llama-3.1-8b-instant", client=groq_client)
# ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# faithfulness = Faithfulness()
# answer_relevancy = AnswerRelevancy()

# # 6 вопросов — покрывают разные сценарии включая граничный случай
# TEST_DATASET = [
#     {
#         "question": "Чему посвящён этот курс?",
#         "ground_truth": (
#             "Курс посвящён инженерии LLM — написанию продуктивного безопасного кода на Python. "
#             "Не написанию loss-функций, а разработке реальных AI-систем."
#         ),
#     },
#     {
#         "question": "Что такое RAG?",
#         "ground_truth": (
#             "RAG (Retrieval-Augmented Generation) — архитектура где LLM отвечает "
#             "на основе найденных документов из векторной базы данных. "
#             "Это основа работы с приватными данными без дообучения."
#         ),
#     },
#     {
#         "question": "Чем Qdrant отличается от Chroma?",
#         "ground_truth": (
#             "Chroma — встраиваемая БД для прототипов, не требует сервера. "
#             "Qdrant — высокопроизводительная БД на Rust для продакшена, "
#             "поддерживает фильтрацию по метаданным и масштабирование."
#         ),
#     },
#     {
#         "question": "Что такое embedding?",
#         "ground_truth": (
#             "Embedding — это числовое представление текста в виде вектора, "
#             "где семантически близкие фразы находятся близко в векторном пространстве."
#         ),
#     },
#     {
#         "question": "Что такое reranking и зачем он нужен?",
#         "ground_truth": (
#             "Reranking — повторная оценка релевантности найденных чанков. "
#             "Векторный поиск сортирует по эмбеддинговому сходству, "
#             "но reranker оценивает реальную смысловую релевантность. "
#             "FlashRank улучшает точность RAG на 15-30%."
#         ),
#     },
#     {
#         "question": "Есть ли в курсе материалы по DataScience?",
#         "ground_truth": (
#             "Нет. Курс про инженерию LLM, а не DataScience. "
#             "Не нужно писать loss-функции — нужно писать продуктивный код на Python."
#         ),
#     },
# ]


# async def run_agent(question: str) -> tuple[str, list[str]]:
#     config = {"configurable": {"thread_id": f"eval-{question[:20]}"}}
#     result = agent.invoke({"question": question}, config=config)
#     return result["answer"], result.get("context", [])


# async def build_eval_dataset() -> Dataset:
#     questions, answers, contexts, ground_truths = [], [], [], []

#     for item in TEST_DATASET:
#         logger.info(f"Evaluating: {item['question']}")
#         answer, context = await run_agent(item["question"])
#         questions.append(item["question"])
#         answers.append(answer)
#         contexts.append(context if context else [""])
#         ground_truths.append(item["ground_truth"])

#     return Dataset.from_dict({
#         "question": questions,
#         "answer": answers,
#         "contexts": contexts,
#         "ground_truth": ground_truths,
#     })


# async def main():
#     logger.info("Starting RAG evaluation")
#     dataset = await build_eval_dataset()

#     logger.info("Running Ragas metrics")
#     results = evaluate(
#         dataset=dataset,
#         metrics=[faithfulness, answer_relevancy],
#         llm=ragas_llm,
#         embeddings=ragas_embeddings,
#         run_config=RunConfig(max_workers=1, timeout=60),
#     )

#     print("\n" + "=" * 50)
#     print("RAG EVALUATION RESULTS")
#     print("=" * 50)
#     df = results.to_pandas()
#     print(f"Faithfulness:      {df['faithfulness'].mean():.3f}")
#     print(f"Answer relevancy:  {df['answer_relevancy'].mean():.3f}")
#     print("=" * 50)

#     df.to_csv("eval_results.csv", index=False)
#     logger.info("Results saved to eval_results.csv")


# if __name__ == "__main__":
#     asyncio.run(main())




"""
RAG pipeline evaluation using Ragas.
Run: python -m tests.eval
"""
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.run_config import RunConfig
from openai import OpenAI as OpenAIClient

from app.agent import agent
from app.config import settings
from app.embeddings import embeddings
from app.logger import get_logger

logger = get_logger(__name__)

groq_client = OpenAIClient(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

ragas_llm = llm_factory(model="llama-3.1-8b-instant", client=groq_client, max_tokens=4096)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy()
context_recall = ContextRecall()

# 6 вопросов — покрывают разные сценарии включая граничный случай
TEST_DATASET = [
    {
        "question": "Чему посвящён этот курс?",
        "ground_truth": (
            "Курс посвящён инженерии LLM — написанию продуктивного безопасного кода на Python. "
            "Не написанию loss-функций, а разработке реальных AI-систем."
        ),
    },
    {
        "question": "Что такое RAG?",
        "ground_truth": (
            "RAG (Retrieval-Augmented Generation) — архитектура где LLM отвечает "
            "на основе найденных документов из векторной базы данных. "
            "Это основа работы с приватными данными без дообучения."
        ),
    },
    {
        "question": "Чем Qdrant отличается от Chroma?",
        "ground_truth": (
            "Chroma — встраиваемая БД для прототипов, не требует сервера. "
            "Qdrant — высокопроизводительная БД на Rust для продакшена, "
            "поддерживает фильтрацию по метаданным и масштабирование."
        ),
    },
    {
        "question": "Что такое embedding?",
        "ground_truth": (
            "Embedding — это числовое представление текста в виде вектора, "
            "где семантически близкие фразы находятся близко в векторном пространстве."
        ),
    },
    {
        "question": "Что такое reranking и зачем он нужен?",
        "ground_truth": (
            "Reranking — повторная оценка релевантности найденных чанков. "
            "Векторный поиск сортирует по эмбеддинговому сходству, "
            "но reranker оценивает реальную смысловую релевантность. "
            "FlashRank улучшает точность RAG на 15-30%."
        ),
    },
    {
        "question": "Зачем добавлять метаданные к чанкам?",
        "ground_truth": (
            "Метаданные нужны для фильтрации (искать только по определённому источнику), "
            "отладки (понять откуда пришёл фрагмент) и безопасности "
            "(исключить конфиденциальные данные из ответа). "
            "Без метаданных RAG слепой."
        ),
    },
]


async def run_agent(question: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": f"eval-{question[:20]}"}}
    result = agent.invoke({"question": question}, config=config)
    return result["answer"], result.get("context", [])


async def build_eval_dataset() -> Dataset:
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_DATASET:
        logger.info(f"Evaluating: {item['question']}")
        answer, context = await run_agent(item["question"])
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context if context else [""])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


async def main():
    logger.info("Starting RAG evaluation")
    dataset = await build_eval_dataset()

    logger.info("Running Ragas metrics")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=1, timeout=60),
    )

    print("\n" + "=" * 50)
    print("RAG EVALUATION RESULTS")
    print("=" * 50)
    df = results.to_pandas()
    print(f"Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"Answer relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"Context recall:    {df['context_recall'].mean():.3f}")
    print("=" * 50)

    df.to_csv("eval_results.csv", index=False)
    logger.info("Results saved to eval_results.csv")


if __name__ == "__main__":
    asyncio.run(main())