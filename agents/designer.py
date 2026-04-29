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

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "designer_prompt.txt"
_llm = ChatAnthropic(model=cfg.MODEL, temperature=cfg.TEMP_DESIGNER)


@llm_retry
def _call_llm(messages: list):
    return _llm.invoke(messages, config=RunnableConfig(callbacks=[usage_handler]))


def designer_node(state: CognivCrewState) -> CognivCrewState:
    logger.info("Designer agent starting")
    system_prompt = _PROMPT_PATH.read_text()
    logger.debug(f"Designer system prompt ({len(system_prompt)} chars)")

    human_message = (
        f"## Original User Request\n\n{state['user_request']}\n\n"
        f"## CEO Strategy\n\n{state['strategy']}\n\n"
        f"## Product Specification\n\n{state['product_spec']}"
    )
    logger.debug(f"Designer human message ({len(human_message)} chars)")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    response = _call_llm(messages)
    state["design_brief"] = response.content

    output_path = Path(state["output_dir"]) / cfg.FILE_DESIGN_BRIEF
    output_path.write_text(response.content)

    logger.info(f"Designer agent complete — design brief written to {output_path}")

    console.print(
        Panel(
            f"[bold green]Design brief complete.[/bold green]\n"
            f"Saved to: [bold white]{output_path}[/bold white]",
            title="[bold cyan]Designer Agent[/bold cyan]",
            expand=False,
        )
    )

    return state
