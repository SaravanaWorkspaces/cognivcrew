from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel

from graph.state import CognivCrewState

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pm_prompt.txt"

_llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)


def pm_node(state: CognivCrewState) -> CognivCrewState:
    system_prompt = _PROMPT_PATH.read_text()

    human_message = (
        f"## Original User Request\n\n{state['user_request']}\n\n"
        f"## CEO Strategy\n\n{state['strategy']}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    response = _llm.invoke(messages)
    state["product_spec"] = response.content

    output_path = Path(state["output_dir"]) / "01_product_spec.md"
    output_path.write_text(response.content)

    console.print(
        Panel(
            f"[bold green]Product specification complete.[/bold green]\n"
            f"Saved to: [bold white]{output_path}[/bold white]",
            title="[bold cyan]PM Agent[/bold cyan]",
            expand=False,
        )
    )

    return state
