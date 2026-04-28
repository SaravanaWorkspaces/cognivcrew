from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel

from graph.state import CognivCrewState

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "ceo_prompt.txt"

_llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)


def ceo_node(state: CognivCrewState) -> CognivCrewState:
    system_prompt = _PROMPT_PATH.read_text()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_request"]),
    ]

    response = _llm.invoke(messages)
    state["strategy"] = response.content

    console.print(
        Panel(
            "[bold green]CEO strategy complete.[/bold green] Strategy written to state.",
            title="[bold cyan]CEO Agent[/bold cyan]",
            expand=False,
        )
    )

    return state
