import re
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

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "qa_prompt.txt"
_llm = ChatAnthropic(model=cfg.MODEL, temperature=cfg.TEMP_QA)


@llm_retry
def _call_llm(messages: list):
    return _llm.invoke(messages, config=RunnableConfig(callbacks=[usage_handler]))


def _extract_fail_reasons(report: str) -> str:
    match = re.search(r"FAIL REASONS:\s*\n((?:\s*-[^\n]+\n?)+)", report)
    if not match:
        return ""
    return match.group(1).strip()


def qa_node(state: CognivCrewState) -> CognivCrewState:
    iteration = state.get("iteration", 0) + 1
    logger.info(f"QA agent starting (will be iteration {iteration})")

    system_prompt = _PROMPT_PATH.read_text()
    logger.debug(f"QA system prompt ({len(system_prompt)} chars)")

    human_message = (
        f"## Original User Request\n\n{state['user_request']}\n\n"
        f"## CEO Strategy\n\n{state['strategy']}\n\n"
        f"## Product Specification\n\n{state['product_spec']}\n\n"
        f"## Design Brief\n\n{state['design_brief']}\n\n"
        f"## Implementation Plan\n\n{state['implementation_plan']}"
    )
    logger.debug(f"QA human message ({len(human_message)} chars)")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    response = _call_llm(messages)
    report = response.content

    passed = "VERDICT: PASS" in report
    fail_reasons = "" if passed else _extract_fail_reasons(report)

    state["qa_verdict"] = report
    state["qa_feedback"] = fail_reasons
    state["iteration"] = iteration

    output_path = Path(state["output_dir"]) / cfg.FILE_QA_REPORT
    output_path.write_text(report)

    if passed:
        logger.info(f"QA agent complete — VERDICT: PASS (iteration {iteration})")
    else:
        logger.warning(
            f"QA agent complete — VERDICT: FAIL (iteration {iteration})\n{fail_reasons}"
        )

    verdict_display = (
        "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
    )
    console.print(
        Panel(
            f"Verdict: {verdict_display}   |   Iteration: [bold white]{iteration}[/bold white]\n"
            f"Saved to: [bold white]{output_path}[/bold white]",
            title="[bold cyan]QA Agent[/bold cyan]",
            expand=False,
        )
    )

    return state
