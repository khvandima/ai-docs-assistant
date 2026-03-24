from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import httpx
from app.logger import get_logger

logger = get_logger(__name__)


def log_retry(retry_state):
    logger.warning(
        f"Retry attempt {retry_state.attempt_number} "
        f"for {retry_state.fn.__name__} "
        f"after error: {retry_state.outcome.exception()}"
    )


# Сетевые ошибки, которые имеет смысл ретраить
_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)

# Для вызовов LLM
llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_NETWORK_ERRORS),
    before_sleep=log_retry,
    reraise=True,
)

# Для операций с Qdrant
qdrant_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_NETWORK_ERRORS),
    before_sleep=log_retry,
    reraise=True,
)