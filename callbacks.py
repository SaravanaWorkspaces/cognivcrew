import sys
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger

from config import cfg


class TokenUsageHandler(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # Try usage_metadata on AIMessage (langchain >= 0.2)
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg:
                    meta = getattr(msg, "usage_metadata", None)
                    if meta:
                        self.input_tokens += meta.get("input_tokens", 0)
                        self.output_tokens += meta.get("output_tokens", 0)
                        return
        # Fallback: llm_output dict from older langchain-anthropic versions
        if response.llm_output:
            usage = response.llm_output.get("usage", {})
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def estimated_cost(self) -> float:
        return (
            self.input_tokens * cfg.PRICE_INPUT_PER_M / 1_000_000
            + self.output_tokens * cfg.PRICE_OUTPUT_PER_M / 1_000_000
        )

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0


usage_handler = TokenUsageHandler()


def setup_logging(output_dir: str) -> None:
    log_level = cfg.LOG_LEVEL
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )
    log_path = Path(output_dir) / "cognivcrew.log"
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        encoding="utf-8",
    )
    logger.info(f"Logging initialised — log file: {log_path}")
