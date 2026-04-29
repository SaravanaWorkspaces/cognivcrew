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

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pm_prompt.txt"
_llm = ChatAnthropic(model=cfg.MODEL, temperature=cfg.TEMP_PM)


@llm_retry
def _call_llm(messages: list):
    return _llm.invoke(messages, config=RunnableConfig(callbacks=[usage_handler]))


def pm_node(state: CognivCrewState) -> CognivCrewState:
    logger.info("PM agent starting")
    system_prompt = _PROMPT_PATH.read_text()
    logger.debug(f"PM system prompt ({len(system_prompt)} chars)")

    human_message = (
        f"## Original User Request\n\n{state['user_request']}\n\n"
        f"## CEO Strategy\n\n{state['strategy']}"
    )
    logger.debug(f"PM human message ({len(human_message)} chars)")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    response = _call_llm(messages)
    state["product_spec"] = response.content

    output_path = Path(state["output_dir"]) / cfg.FILE_PRODUCT_SPEC
    output_path.write_text(response.content)

    logger.info(f"PM agent complete — product spec written to {output_path}")

    console.print(
        Panel(
            f"[bold green]Product specification complete.[/bold green]\n"
            f"Saved to: [bold white]{output_path}[/bold white]",
            title="[bold cyan]PM Agent[/bold cyan]",
            expand=False,
        )
    )

    return state
