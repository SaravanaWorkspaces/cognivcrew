from dotenv import load_dotenv

load_dotenv()

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from langchain_core.runnables import RunnableConfig
from loguru import logger
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from callbacks import setup_logging, usage_handler
from config import cfg
from graph.state import CognivCrewState, default_state
from graph.workflow import app

cli = typer.Typer(
    name="cognivcrew",
    help="AI-powered multi-agent software project pipeline.",
    add_completion=False,
)
console = Console()

_REQUIRED_ENV = ["ANTHROPIC_API_KEY"]
_PROMPT_FILES = [
    "ceo_prompt.txt",
    "pm_prompt.txt",
    "designer_prompt.txt",
    "engineer_prompt.txt",
    "qa_prompt.txt",
]
_PROMPTS_DIR = Path(__file__).parent / "prompts"

_AGENT_LABELS: dict[str, str] = {
    "ceo": "CEO — Strategy",
    "pm": "PM — Product Spec",
    "designer": "Designer — Design Brief",
    "engineer": "Engineer — Implementation Plan",
    "qa": "QA — Quality Review",
    "final": "Final — Assembly",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_startup(exit_on_failure: bool = True) -> list[str]:
    errors: list[str] = []

    for var in _REQUIRED_ENV:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: [bold]{var}[/bold]")

    for filename in _PROMPT_FILES:
        path = _PROMPTS_DIR / filename
        if not path.exists():
            errors.append(f"Prompt file not found: [bold]{path}[/bold]")

    outputs_dir = Path(cfg.OUTPUT_DIR)
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        probe = outputs_dir / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        errors.append(f"Output directory not writable: [bold]{outputs_dir}[/bold] ({exc})")

    if errors and exit_on_failure:
        lines = "\n".join(f"  • {e}" for e in errors)
        console.print(
            Panel(
                f"[bold red]Startup validation failed:[/bold red]\n\n{lines}",
                title="[bold red]CognivCrew — Configuration Error[/bold red]",
                expand=False,
            )
        )
        raise SystemExit(1)

    return errors


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@cli.command()
def run(
    project: str = typer.Argument(..., help="Software project description"),
    langsmith: bool = typer.Option(False, "--langsmith", help="Enable LangSmith tracing"),
):
    """Run the full CognivCrew AI pipeline on a project description."""
    _validate_startup()

    if langsmith and not os.getenv("LANGCHAIN_API_KEY"):
        console.print("[yellow]Warning: --langsmith set but LANGCHAIN_API_KEY is not set.[/yellow]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.OUTPUT_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir))
    usage_handler.reset()

    logger.info(f"Pipeline starting — project: {project[:120]}")
    logger.info(f"Output directory: {output_dir}")

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

    run_config = RunnableConfig(
        callbacks=[usage_handler],
        metadata={
            "run_timestamp": timestamp,
            "user_request_preview": project[:100],
            "project": "cognivcrew",
        },
        tags=["cognivcrew", f"run-{timestamp}"],
    )

    wall_start = time.perf_counter()
    final_state: CognivCrewState = {}
    node_timings: dict[str, float] = {}
    last_event_time = wall_start

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        pipeline_task = progress.add_task(
            "[cyan]Pipeline running...", total=len(_AGENT_LABELS)
        )

        for event in app.stream(initial_state, config=run_config):
            node_name = next(iter(event))
            if node_name.startswith("__"):
                continue

            now = time.perf_counter()
            elapsed = now - last_event_time
            last_event_time = now
            node_timings[node_name] = elapsed

            label = _AGENT_LABELS.get(node_name, node_name)
            progress.advance(pipeline_task)
            progress.print(f"  [green]✓[/green] {label} [dim]({elapsed:.1f}s)[/dim]")

            final_state = event[node_name]

    total_duration = time.perf_counter() - wall_start
    logger.info(
        f"Pipeline complete — duration {total_duration:.1f}s, "
        f"tokens in={usage_handler.input_tokens} out={usage_handler.output_tokens}, "
        f"cost=${usage_handler.estimated_cost():.4f}"
    )

    _print_run_summary(final_state, total_duration, node_timings)


@cli.command(name="list")
def list_runs():
    """List all past pipeline runs."""
    outputs_dir = Path(cfg.OUTPUT_DIR)
    if not outputs_dir.exists() or not any(outputs_dir.iterdir()):
        console.print(Panel("[yellow]No past runs found.[/yellow]", title="CognivCrew — Runs"))
        return

    runs = sorted(
        (d for d in outputs_dir.iterdir() if d.is_dir()),
        reverse=True,
    )

    table = Table(title="CognivCrew — Past Runs", show_lines=True)
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Project", style="white")
    table.add_column("Files", justify="right", style="green")

    for run_dir in runs:
        summary = run_dir / "00_project_summary.md"
        project_preview = "[dim]—[/dim]"
        if summary.exists():
            for line in summary.read_text().splitlines():
                if line.startswith("**User Request:**"):
                    raw = line.replace("**User Request:**", "").strip()
                    project_preview = raw[:72] + ("…" if len(raw) > 72 else "")
                    break
        file_count = sum(1 for f in run_dir.iterdir() if f.is_file())
        table.add_row(run_dir.name, project_preview, str(file_count))

    console.print(table)


@cli.command()
def show(run_id: str = typer.Argument(..., help="Run ID (timestamp folder name)")):
    """Display the summary for a specific past run."""
    run_dir = Path(cfg.OUTPUT_DIR) / run_id
    if not run_dir.exists():
        console.print(
            Panel(
                f"[red]Run not found:[/red] [bold]{run_dir}[/bold]",
                title="CognivCrew — Show",
            )
        )
        raise SystemExit(1)

    summary = run_dir / "00_project_summary.md"
    if summary.exists():
        console.print(Panel(Markdown(summary.read_text()), title=f"Run: {run_id}", expand=True))
    else:
        files = sorted(run_dir.iterdir())
        file_list = "\n".join(f"  • {f.name}" for f in files)
        console.print(
            Panel(
                f"[yellow]No summary file found.[/yellow]\n\nFiles:\n{file_list}",
                title=f"Run: {run_id}",
            )
        )


@cli.command()
def validate():
    """Check configuration, environment variables, and prompt files."""
    errors = _validate_startup(exit_on_failure=False)
    if errors:
        lines = "\n".join(f"  • {e}" for e in errors)
        console.print(
            Panel(
                f"[bold red]Validation failed:[/bold red]\n\n{lines}",
                title="[bold red]CognivCrew — Validate[/bold red]",
                expand=False,
            )
        )
        raise SystemExit(1)

    rows = [
        ("Model", cfg.MODEL),
        ("Max QA iterations", str(cfg.MAX_ITERATIONS)),
        ("Log level", cfg.LOG_LEVEL),
        ("Output directory", cfg.OUTPUT_DIR),
        ("Prompt files", f"{len(_PROMPT_FILES)} / {len(_PROMPT_FILES)} found"),
        ("ANTHROPIC_API_KEY", "set"),
        (
            "LangSmith tracing",
            "enabled" if os.getenv("LANGCHAIN_API_KEY") else "disabled (no LANGCHAIN_API_KEY)",
        ),
    ]
    table = Table(title="Configuration", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white")
    for key, val in rows:
        table.add_row(key, val)

    console.print(
        Panel(table, title="[bold green]CognivCrew — All Checks Passed[/bold green]", expand=False)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_run_summary(
    state: CognivCrewState,
    duration: float,
    node_timings: dict[str, float],
) -> None:
    cost = usage_handler.estimated_cost()
    iterations = state.get("iteration", 0)
    output_dir = state.get("output_dir", "—")

    timing_lines = "\n".join(
        f"  {_AGENT_LABELS.get(node, node):<40} {t:.1f}s"
        for node, t in node_timings.items()
    )

    summary = (
        f"[bold]Total duration:[/bold]   {duration:.1f}s\n"
        f"[bold]QA iterations:[/bold]    {iterations}\n"
        f"[bold]Input tokens:[/bold]     {usage_handler.input_tokens:,}\n"
        f"[bold]Output tokens:[/bold]    {usage_handler.output_tokens:,}\n"
        f"[bold]Total tokens:[/bold]     {usage_handler.total_tokens:,}\n"
        f"[bold]Estimated cost:[/bold]   ${cost:.4f} USD\n"
        f"[bold]Output path:[/bold]      {output_dir}\n"
        f"\n[bold]Per-agent timing:[/bold]\n{timing_lines}"
    )

    console.print(
        Panel(
            summary,
            title="[bold green]Run Summary[/bold green]",
            expand=False,
        )
    )


if __name__ == "__main__":
    cli()
