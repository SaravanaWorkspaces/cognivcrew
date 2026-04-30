import os

from anthropic import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class Config:
    VERSION: str = "1.0.0"
    MODEL: str = "claude-sonnet-4-5"
    TEMP_CEO: float = 0.0
    TEMP_PM: float = 0.0
    TEMP_ARCHITECT: float = 0.0
    TEMP_DESIGNER: float = 0.7
    TEMP_ENGINEER: float = 0.3
    TEMP_QA: float = 0.0
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))
    MAX_ARCHITECT_ITERATIONS: int = 3
    OUTPUT_DIR: str = "outputs"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # Claude Sonnet pricing per million tokens
    PRICE_INPUT_PER_M: float = 3.00
    PRICE_OUTPUT_PER_M: float = 15.00
    # Output file names (numbered for pipeline order)
    FILE_PROJECT_SUMMARY: str = "00_project_summary.md"
    FILE_PRODUCT_SPEC: str = "01_product_spec.md"
    FILE_ARCHITECT_BRIEF: str = "02_architect_brief.md"
    FILE_DESIGN_BRIEF: str = "03_design_brief.md"
    FILE_IMPLEMENTATION_PLAN: str = "04_implementation_plan.md"
    FILE_QA_REPORT: str = "05_qa_report.md"


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
