from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from graph.state import CognivCrewState, default_state
from graph.workflow import app

cli = typer.Typer()
console = Console()


@cli.command()
def run(project: str = typer.Argument(..., help="Software project description")):
    """Run the CognivCrew AI pipeline on a project description."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold yellow]{project}[/bold yellow]\n"
            f"Output folder: [bold white]{output_dir}[/bold white]",
            title="[bold cyan]CognivCrew — Starting Pipeline[/bold cyan]",
            expand=False,
        )
    )

    initial_state: CognivCrewState = {
        **default_state(),
        "user_request": project,
        "output_dir": str(output_dir),
    }

    result = app.invoke(initial_state)

    console.print(
        Panel(
            Markdown(result.get("strategy", "")),
            title="[bold green]CEO Strategy[/bold green]",
            expand=True,
        )
    )


if __name__ == "__main__":
    cli()
