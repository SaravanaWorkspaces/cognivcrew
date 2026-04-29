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

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "architect_prompt.txt"
_llm = ChatAnthropic(model=cfg.MODEL, temperature=cfg.TEMP_ARCHITECT)


@llm_retry
def _call_llm(messages: list):
    return _llm.invoke(messages, config=RunnableConfig(callbacks=[usage_handler]))


def architect_node(state: CognivCrewState) -> CognivCrewState:
    architect_iteration = state.get("architect_iteration", 0)
    logger.info(f"Architect agent starting (architect_iteration {architect_iteration})")

    system_prompt = _PROMPT_PATH.read_text()
    logger.debug(f"Architect system prompt ({len(system_prompt)} chars)")

    human_feedback = state.get("human_feedback", "").strip()
    previous_brief = state.get("architect_brief", "").strip()

    if human_feedback and previous_brief:
        logger.info("Architect revising brief based on human feedback")
        human_message = (
            f"## Original User Request\n\n{state['user_request']}\n\n"
            f"## CEO Strategy\n\n{state['strategy']}\n\n"
            f"## Product Specification\n\n{state['product_spec']}\n\n"
            f"## Your Previous Architecture Brief\n\n{previous_brief}\n\n"
            f"## Human Feedback on Your Brief\n\n{human_feedback}\n\n"
            f"Before producing your revised Architecture Brief, acknowledge each feedback point "
            f"individually by number or bullet, stating whether you are accepting or rejecting it "
            f"and why. Then produce the complete revised Architecture Brief in full — do not "
            f"reference sections from your previous brief; rewrite every section."
        )
    else:
        human_message = (
            f"## Original User Request\n\n{state['user_request']}\n\n"
            f"## CEO Strategy\n\n{state['strategy']}\n\n"
            f"## Product Specification\n\n{state['product_spec']}"
        )

    logger.debug(f"Architect human message ({len(human_message)} chars)")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    response = _call_llm(messages)
    state["architect_brief"] = response.content
    state["architect_iteration"] = architect_iteration + 1

    output_path = Path(state["output_dir"]) / cfg.FILE_ARCHITECT_BRIEF
    output_path.write_text(response.content)

    new_iteration = state["architect_iteration"]
    iteration_label = f"iteration {new_iteration}" if new_iteration > 1 else "initial brief"
    logger.info(
        f"Architect agent complete ({iteration_label}) — brief written to {output_path}"
    )

    console.print(
        Panel(
            f"[bold green]Architecture brief complete.[/bold green] "
            f"Iteration: [bold white]{new_iteration}[/bold white]\n"
            f"Saved to: [bold white]{output_path}[/bold white]",
            title="[bold cyan]Architect Agent[/bold cyan]",
            expand=False,
        )
    )

    return state
