import os

from anthropic import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class Config:
    MODEL: str = "claude-sonnet-4-5"
    TEMP_CEO: float = 0.0
    TEMP_PM: float = 0.0
    TEMP_DESIGNER: float = 0.7
    TEMP_ENGINEER: float = 0.3
    TEMP_QA: float = 0.0
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))
    OUTPUT_DIR: str = "outputs"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # Claude Sonnet pricing per million tokens
    PRICE_INPUT_PER_M: float = 3.00
    PRICE_OUTPUT_PER_M: float = 15.00


cfg = Config()

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


def _log_retry(retry_state) -> None:
    from loguru import logger  # deferred — loguru may not be configured yet

    ex = retry_state.outcome.exception()
    logger.warning(
        f"API call failed (attempt {retry_state.attempt_number}/3): "
        f"{type(ex).__name__}: {ex} — retrying..."
    )


llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(exp_base=2, min=2, max=60),
    retry=retry_if_exception_type(_RETRYABLE),
    before_sleep=_log_retry,
)
