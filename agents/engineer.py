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

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "engineer_prompt.txt"
_llm = ChatAnthropic(model=cfg.MODEL, temperature=cfg.TEMP_ENGINEER)


@llm_retry
def _call_llm(messages: list):
    return _llm.invoke(messages, config=RunnableConfig(callbacks=[usage_handler]))


def engineer_node(state: CognivCrewState) -> CognivCrewState:
    iteration = state.get("iteration", 0)
    logger.info(f"Engineer agent starting (iteration {iteration})")

    system_prompt = _PROMPT_PATH.read_text()
    logger.debug(f"Engineer system prompt ({len(system_prompt)} chars)")

    qa_feedback = state.get("qa_feedback", "").strip()
    if qa_feedback:
        logger.warning(f"Engineer revising due to QA feedback:\n{qa_feedback}")
        feedback_block = f"## QA Feedback (Revision Required)\n\n{qa_feedback}\n\n"
    else:
        feedback_block = ""

    human_message = (
        f"## Original User Request\n\n{state['user_request']}\n\n"
        f"## CEO Strategy\n\n{state['strategy']}\n\n"
        f"## Product Specification\n\n{state['product_spec']}\n\n"
        f"## Design Brief\n\n{state['design_brief']}\n\n"
        f"{feedback_block}"
    ).rstrip()

    logger.debug(f"Engineer human message ({len(human_message)} chars)")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    response = _call_llm(messages)
    state["implementation_plan"] = response.content

    output_path = Path(state["output_dir"]) / "03_implementation_plan.md"
    output_path.write_text(response.content)

    iteration_label = f" (revision {iteration})" if iteration > 0 else ""
    logger.info(f"Engineer agent complete{iteration_label} — plan written to {output_path}")

    console.print(
        Panel(
            f"[bold green]Implementation plan complete{iteration_label}.[/bold green]\n"
            f"Saved to: [bold white]{output_path}[/bold white]",
            title="[bold cyan]Engineer Agent[/bold cyan]",
            expand=False,
        )
    )

    return state
