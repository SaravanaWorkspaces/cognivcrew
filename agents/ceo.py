from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from callbacks import usage_handler
from config import cfg, llm_retry
from graph.state import CognivCrewState

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "ceo_prompt.txt"
_llm = ChatAnthropic(model=cfg.MODEL, temperature=cfg.TEMP_CEO)


@llm_retry
def _call_llm(messages: list):
    return _llm.invoke(messages, config=RunnableConfig(callbacks=[usage_handler]))


def ceo_node(state: CognivCrewState) -> CognivCrewState:
    logger.info("CEO agent starting")
    system_prompt = _PROMPT_PATH.read_text()
    logger.debug(f"CEO system prompt ({len(system_prompt)} chars)")
    logger.debug(f"CEO user request: {state['user_request'][:200]}")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_request"]),
    ]

    response = _call_llm(messages)
    state["strategy"] = response.content

    logger.debug(f"CEO response ({len(response.content)} chars)")
    logger.info("CEO agent complete — strategy written to state")

    console.print(
        Panel(
            "[bold green]CEO strategy complete.[/bold green] Strategy written to state.",
            title="[bold cyan]CEO Agent[/bold cyan]",
            expand=False,
        )
    )

    return state
