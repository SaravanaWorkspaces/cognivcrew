from dotenv import load_dotenv

load_dotenv()

import os
import time
from datetime import datetime
from pathlib import Path

import rich.box
import typer
from langchain_core.runnables import RunnableConfig
from loguru import logger
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
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

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PROMPT_FILES = [
    "ceo_prompt.txt",
    "pm_prompt.txt",
    "architect_prompt.txt",
    "designer_prompt.txt",
    "engineer_prompt.txt",
    "qa_prompt.txt",
]
_REQUIRED_PACKAGES = [
    "langchain_anthropic",
    "langgraph",
    "loguru",
    "tenacity",
    "rich",
    "typer",
    "langsmith",
]

_AGENT_EMOJIS: dict[str, str] = {
    "ceo":      "👔",
    "pm":       "📋",
    "architect":"🏗 ",
    "designer": "🎨",
    "engineer": "⚙️ ",
    "qa":       "🔍",
    "final":    "📦",
}
_AGENT_LABELS: dict[str, str] = {
    "ceo":      "CEO — Strategy",
    "pm":       "PM — Product Spec",
    "architect":"Architect — Architecture Brief",
    "designer": "Designer — Design Brief",
    "engineer": "Engineer — Implementation Plan",
    "qa":       "QA — Quality Review",
    "final":    "Final — Assembly",
}
_AGENT_DESCRIPTIONS: dict[str, str] = {
    "ceo":      "Sets vision, strategy, and product goals",
    "pm":       "Writes epics, stories, and acceptance criteria",
    "architect":"Designs system architecture (human review gate)",
    "designer": "Defines UX journeys and interaction patterns",
    "engineer": "Produces the full implementation plan",
    "qa":       "Reviews all deliverables and issues a pass/fail verdict",
    "final":    "Assembles the project summary and output index",
}

_OUTPUT_FILES_ORDERED = [
    cfg.FILE_PROJECT_SUMMARY,
    cfg.FILE_PRODUCT_SPEC,
    cfg.FILE_ARCHITECT_BRIEF,
    cfg.FILE_DESIGN_BRIEF,
    cfg.FILE_IMPLEMENTATION_PLAN,
    cfg.FILE_QA_REPORT,
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_items() -> list[tuple[str, bool, str]]:
    """Return list of (label, passed, fix_instruction) for every check."""
    items: list[tuple[str, bool, str]] = []

    # Env vars
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))
    items.append((
        "ANTHROPIC_API_KEY",
        api_key_set,
        "Set ANTHROPIC_API_KEY in your .env file — get a key at https://console.anthropic.com",
    ))
    langsmith_set = bool(os.getenv("LANGCHAIN_API_KEY"))
    items.append((
        "LANGCHAIN_API_KEY (optional — LangSmith tracing)",
        langsmith_set,
        "Set LANGCHAIN_API_KEY in .env to enable LangSmith tracing (optional)",
    ))

    # Prompt files
    for filename in _PROMPT_FILES:
        path = _PROMPTS_DIR / filename
        items.append((
            f"prompts/{filename}",
            path.exists(),
            f"Recreate {path} — see project docs for expected content",
        ))

    # outputs/ writable
    outputs_dir = Path(cfg.OUTPUT_DIR)
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        probe = outputs_dir / ".write_probe"
        probe.touch()
        probe.unlink()
        writable = True
    except OSError:
        writable = False
    items.append((
        f"{cfg.OUTPUT_DIR}/ directory writable",
        writable,
        f"Ensure the process has write permission to {outputs_dir.resolve()}",
    ))

    # Package imports
    for pkg in _REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            importable = True
        except ImportError:
            importable = False
        items.append((
            f"import {pkg}",
            importable,
            f"Run: uv sync   (package '{pkg}' is missing from the environment)",
        ))

    return items


def _validate_startup(exit_on_failure: bool = True) -> list[str]:
    """Return list of Rich-markup error strings; exits if exit_on_failure and any hard errors."""
    errors: list[str] = []
    for label, passed, fix in _check_items():
        if not passed and "optional" not in label.lower():
            errors.append(f"[bold]{label}[/bold] — {fix}")
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
# Live progress table
# ---------------------------------------------------------------------------

def _make_pipeline_table(
    agent_states: dict[str, dict],
    wall_start: float,
) -> Table:
    table = Table(
        title="[bold cyan]Pipeline Progress[/bold cyan]",
        box=rich.box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        expand=False,
        min_width=64,
    )
    table.add_column("Agent", min_width=36)
    table.add_column("Status", min_width=12)
    table.add_column("Time", justify="right", min_width=7)
    table.add_column("Tokens", justify="right", min_width=9)

    for node_name, label in _AGENT_LABELS.items():
        st = agent_states.get(node_name, {})
        status = st.get("status", "pending")
        emoji = _AGENT_EMOJIS.get(node_name, "• ")

        if status == "done":
            status_cell = "[green]✓ Done[/green]"
            time_cell   = f"{st['elapsed']:.1f}s"
            token_cell  = f"{st['tokens']:,}" if st.get("tokens") else "—"
        else:
            status_cell = "[dim]○ Pending[/dim]"
            time_cell   = "—"
            token_cell  = "—"

        table.add_row(f"{emoji} {label}", status_cell, time_cell, token_cell)

    elapsed_total = time.perf_counter() - wall_start
    table.caption = f"[dim]Total elapsed: {elapsed_total:.1f}s[/dim]"
    return table


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def _stream_phase(
    state_or_none,
    run_config: RunnableConfig,
    agent_states: dict[str, dict],
    node_timings: dict[str, float],
    last_time: list[float],
    last_tokens: list[int],
    live: Live,
    wall_start: float,
) -> CognivCrewState:
    final_state: CognivCrewState = {}
    for event in app.stream(state_or_none, config=run_config, stream_mode="updates"):
        node_name = next(iter(event))
        if node_name.startswith("__"):
            continue
        now = time.perf_counter()
        elapsed = now - last_time[0]
        last_time[0] = now

        current_tokens = usage_handler.total_tokens
        delta = current_tokens - last_tokens[0]
        last_tokens[0] = current_tokens

        node_timings[node_name] = elapsed
        agent_states[node_name] = {"status": "done", "elapsed": elapsed, "tokens": delta}
        live.update(_make_pipeline_table(agent_states, wall_start))
        final_state = event[node_name]
    return final_state


# ---------------------------------------------------------------------------
# Human Approval Gate
# ---------------------------------------------------------------------------

def handle_approval_gate(state: CognivCrewState) -> dict:
    architect_iteration = state.get("architect_iteration", 0)
    brief = state.get("architect_brief", "")

    console.print(
        Panel(
            Markdown(brief),
            title=f"[bold magenta]Architect Brief — Iteration {architect_iteration}[/bold magenta]",
            expand=True,
        )
    )
    console.print(Rule(style="dim"))
    console.print(
        "\n[bold yellow]Review the Architecture Brief above.[/bold yellow]\n"
        "  [bold green]A[/bold green] — Approve and continue the pipeline\n"
        "  [bold red]R[/bold red]   — Reject and stop (pipeline will not continue)\n"
        "  [bold cyan]M[/bold cyan] — Request modifications with written feedback\n"
    )

    while True:
        raw = console.input("[bold]Your decision (A/R/M):[/bold] ").strip().upper()
        if raw in ("A", "R", "M"):
            break
        console.print("[yellow]  Please enter A, R, or M.[/yellow]")

    if raw == "A":
        logger.info("Human approved architect brief")
        console.print("\n[bold green]✓ Approved — pipeline continuing.[/bold green]\n")
        return {"architect_approved": True, "human_feedback": ""}

    if raw == "R":
        logger.warning("Human rejected architect brief — pipeline stopped")
        console.print(
            Panel(
                "[bold red]Pipeline stopped.[/bold red]\n"
                "Re-run with a revised project description.",
                title="[bold red]Rejected[/bold red]",
                expand=False,
            )
        )
        raise SystemExit(0)

    # M — collect feedback
    console.print(
        "\n[bold cyan]Describe the required modifications.[/bold cyan] "
        "(Be specific — the architect will address each point.)\n"
    )
    while True:
        feedback = console.input("[bold]Feedback:[/bold] ").strip()
        if feedback:
            break
        console.print("[yellow]  Feedback cannot be empty.[/yellow]")

    logger.info(f"Human requested architect modifications: {feedback[:120]}")
    console.print(
        f"\n[bold cyan]✓ Feedback recorded.[/bold cyan] "
        f"Architect will revise (iteration {architect_iteration + 1}).\n"
    )
    return {"architect_approved": False, "human_feedback": feedback}


# ---------------------------------------------------------------------------
# CLI — run
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
    thread_id = f"run-{timestamp}"
    output_dir = Path(cfg.OUTPUT_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir))
    usage_handler.reset()

    logger.info(f"Pipeline starting — thread_id={thread_id}")
    logger.info(f"Project: {project[:200]}")
    logger.info(f"Output directory: {output_dir}")

    # Startup banner
    request_preview = project if len(project) <= 80 else project[:77] + "..."
    console.print(
        Panel(
            f"[bold yellow]{request_preview}[/bold yellow]\n\n"
            f"[dim]Output:[/dim]  [white]{output_dir}[/white]\n"
            f"[dim]Model:[/dim]   [white]{cfg.MODEL}[/white]\n"
            f"[dim]Thread:[/dim]  [white]{thread_id}[/white]",
            title="[bold cyan]CognivCrew v{} — Starting Pipeline[/bold cyan]".format(cfg.VERSION),
            expand=False,
            border_style="cyan",
        )
    )

    initial_state: CognivCrewState = {
        **default_state(),
        "user_request": project,
        "output_dir": str(output_dir),
    }
    run_config = RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=[usage_handler],
        metadata={
            "run_timestamp": timestamp,
            "user_request_preview": project[:100],
            "project": "cognivcrew",
        },
        tags=["cognivcrew", f"run-{timestamp}"],
    )

    wall_start   = time.perf_counter()
    node_timings: dict[str, float] = {}
    agent_states: dict[str, dict]  = {k: {} for k in _AGENT_LABELS}
    last_time    = [wall_start]
    last_tokens  = [0]
    final_state: CognivCrewState   = {}

    # Phase 1: stream until architect interrupt
    with Live(
        _make_pipeline_table(agent_states, wall_start),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        final_state = _stream_phase(
            initial_state, run_config, agent_states, node_timings,
            last_time, last_tokens, live, wall_start,
        )

    # Architect approval loop
    while True:
        graph_state = app.get_state(run_config)
        if not graph_state.next:
            break

        current_state       = dict(graph_state.values)
        architect_iteration = current_state.get("architect_iteration", 0)

        if architect_iteration >= cfg.MAX_ARCHITECT_ITERATIONS:
            logger.warning(
                f"Max architect iterations ({cfg.MAX_ARCHITECT_ITERATIONS}) reached — force-approving"
            )
            console.print(
                Panel(
                    f"[bold yellow]Maximum architect iterations ({cfg.MAX_ARCHITECT_ITERATIONS}) reached.[/bold yellow]\n"
                    f"Architecture brief auto-approved. See [white]{cfg.FILE_ARCHITECT_BRIEF}[/white].",
                    title="[bold yellow]Architect — Max Iterations[/bold yellow]",
                    expand=False,
                )
            )
            gate_updates: dict = {"architect_approved": True, "human_feedback": ""}
        else:
            gate_updates = handle_approval_gate(current_state)

        app.update_state(run_config, gate_updates, as_node="architect")
        logger.info(f"Gate applied — approved={gate_updates.get('architect_approved')}")

        # Resume stream in a new Live context
        with Live(
            _make_pipeline_table(agent_states, wall_start),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            final_state = _stream_phase(
                None, run_config, agent_states, node_timings,
                last_time, last_tokens, live, wall_start,
            )

    total_duration = time.perf_counter() - wall_start
    logger.info(
        f"Pipeline complete — {total_duration:.1f}s, "
        f"tokens in={usage_handler.input_tokens} out={usage_handler.output_tokens}, "
        f"cost=${usage_handler.estimated_cost():.4f}"
    )
    _print_run_summary(final_state, total_duration, node_timings)


# ---------------------------------------------------------------------------
# CLI — list
# ---------------------------------------------------------------------------

@cli.command(name="list")
def list_runs():
    """List all past pipeline runs."""
    outputs_dir = Path(cfg.OUTPUT_DIR)
    runs = sorted(
        (d for d in outputs_dir.iterdir() if d.is_dir()) if outputs_dir.exists() else [],
        reverse=True,
    )
    if not runs:
        console.print(Panel("[yellow]No past runs found.[/yellow]", title="CognivCrew — Runs"))
        return

    table = Table(
        title="[bold]CognivCrew — Past Runs[/bold]",
        box=rich.box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#",         justify="right", style="dim",  no_wrap=True, min_width=3)
    table.add_column("Timestamp", style="cyan",  no_wrap=True)
    table.add_column("Request",   style="white", min_width=40)
    table.add_column("QA Result", justify="center", min_width=10)

    for i, run_dir in enumerate(runs, 1):
        summary = run_dir / cfg.FILE_PROJECT_SUMMARY
        request_preview = "[dim]—[/dim]"
        qa_result       = "[dim]—[/dim]"

        if summary.exists():
            for line in summary.read_text().splitlines():
                if line.startswith("**User Request:**"):
                    raw = line.replace("**User Request:**", "").strip()
                    request_preview = raw[:68] + ("…" if len(raw) > 68 else "")
                elif line.startswith("**QA Verdict:**"):
                    verdict = line.replace("**QA Verdict:**", "").strip()
                    if verdict == "PASS":
                        qa_result = "[bold green]PASS[/bold green]"
                    elif verdict == "FAIL":
                        qa_result = "[bold red]FAIL[/bold red]"
                    else:
                        qa_result = f"[dim]{verdict}[/dim]"

        table.add_row(str(i), run_dir.name, request_preview, qa_result)

    console.print(table)


# ---------------------------------------------------------------------------
# CLI — show
# ---------------------------------------------------------------------------

@cli.command()
def show(run_id: str = typer.Argument(..., help="Run timestamp (e.g. 20240428_143022)")):
    """Print full contents of all output files for a past run."""
    run_dir = Path(cfg.OUTPUT_DIR) / run_id
    if not run_dir.exists():
        console.print(
            Panel(
                f"[red]Run not found:[/red] [bold]{run_dir}[/bold]",
                title="CognivCrew — Show",
            )
        )
        raise SystemExit(1)

    file_titles = {
        cfg.FILE_PROJECT_SUMMARY:     "Project Summary",
        cfg.FILE_PRODUCT_SPEC:        "Product Specification",
        cfg.FILE_ARCHITECT_BRIEF:     "Architecture Brief",
        cfg.FILE_DESIGN_BRIEF:        "Design Brief",
        cfg.FILE_IMPLEMENTATION_PLAN: "Implementation Plan",
        cfg.FILE_QA_REPORT:           "QA Report",
    }

    found_any = False
    for filename in _OUTPUT_FILES_ORDERED:
        path = run_dir / filename
        if not path.exists():
            continue
        found_any = True
        console.print(Rule(f"[bold cyan]{file_titles.get(filename, filename)}[/bold cyan]"))
        console.print(Markdown(path.read_text()))
        console.print()

    if not found_any:
        console.print(Panel(f"[yellow]No output files found in {run_dir}[/yellow]"))


# ---------------------------------------------------------------------------
# CLI — validate
# ---------------------------------------------------------------------------

@cli.command()
def validate():
    """Run a full configuration checklist and print pass/fail per item."""
    items = _check_items()

    table = Table(
        box=rich.box.SIMPLE,
        show_header=False,
        padding=(0, 1),
        expand=False,
        min_width=70,
    )
    table.add_column("Icon",  no_wrap=True, min_width=2)
    table.add_column("Label", style="white")
    table.add_column("Result")

    failures: list[tuple[str, str]] = []

    sections = [
        ("Environment Variables", ["ANTHROPIC_API_KEY", "LANGCHAIN_API_KEY (optional — LangSmith tracing)"]),
        ("Prompt Files",          [f"prompts/{f}" for f in _PROMPT_FILES]),
        ("System",                [f"{cfg.OUTPUT_DIR}/ directory writable"] + [f"import {p}" for p in _REQUIRED_PACKAGES]),
    ]

    item_map = {label: (passed, fix) for label, passed, fix in items}

    for section_title, labels in sections:
        table.add_row("", f"[bold dim]{section_title}[/bold dim]", "")
        for label in labels:
            passed, fix = item_map.get(label, (False, ""))
            optional = "optional" in label.lower()
            if passed:
                icon   = "[bold green]✓[/bold green]"
                result = "[green]ok[/green]"
            elif optional:
                icon   = "[dim]–[/dim]"
                result = "[dim]not set (optional)[/dim]"
            else:
                icon   = "[bold red]✗[/bold red]"
                result = "[red]FAIL[/red]"
                failures.append((label, fix))
            table.add_row(icon, label, result)

    overall_title = (
        "[bold green]CognivCrew — All Checks Passed[/bold green]"
        if not failures
        else "[bold red]CognivCrew — Checks Failed[/bold red]"
    )
    console.print(Panel(table, title=overall_title, expand=False))

    if failures:
        console.print("\n[bold red]Fix instructions:[/bold red]")
        for label, fix in failures:
            console.print(f"  [red]✗[/red] [bold]{label}[/bold]\n    → {fix}\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI — info
# ---------------------------------------------------------------------------

@cli.command()
def info():
    """Display project info, agent roster, and configuration."""
    # Header
    console.print(
        Panel(
            "[bold white]AI-powered multi-agent software project pipeline[/bold white]\n"
            "[dim]Turns a plain-English project description into a complete, QA-verified set\n"
            "of deliverables using a crew of specialised Claude agents and LangGraph.[/dim]",
            title=f"[bold cyan]CognivCrew[/bold cyan]  [dim]v{cfg.VERSION}[/dim]",
            expand=False,
            border_style="cyan",
        )
    )

    # Agent roster table
    roster = Table(
        title="[bold]Agent Roster[/bold]",
        box=rich.box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        expand=False,
        min_width=72,
    )
    roster.add_column("",     no_wrap=True, min_width=3)
    roster.add_column("Agent",       style="cyan bold", min_width=12)
    roster.add_column("Role",        style="white",     min_width=30)
    roster.add_column("Temp", justify="right", style="dim", min_width=6)

    _TEMPS = {
        "ceo":      cfg.TEMP_CEO,
        "pm":       cfg.TEMP_PM,
        "architect":cfg.TEMP_ARCHITECT,
        "designer": cfg.TEMP_DESIGNER,
        "engineer": cfg.TEMP_ENGINEER,
        "qa":       cfg.TEMP_QA,
        "final":    None,
    }

    for node, label in _AGENT_LABELS.items():
        name = label.split(" — ")[0]
        desc = _AGENT_DESCRIPTIONS[node]
        temp = _TEMPS[node]
        temp_cell = f"{temp:.1f}" if temp is not None else "[dim]n/a[/dim]"
        roster.add_row(_AGENT_EMOJIS[node], name, desc, temp_cell)

    # Config summary table
    config_tbl = Table(
        title="[bold]Configuration[/bold]",
        box=rich.box.ROUNDED,
        show_header=False,
        padding=(0, 1),
        expand=False,
        min_width=36,
    )
    config_tbl.add_column(style="dim")
    config_tbl.add_column(style="bold white")
    config_tbl.add_row("Model",           cfg.MODEL)
    config_tbl.add_row("QA loop cap",     str(cfg.MAX_ITERATIONS))
    config_tbl.add_row("Architect cap",   str(cfg.MAX_ARCHITECT_ITERATIONS))
    config_tbl.add_row("Output dir",      cfg.OUTPUT_DIR)
    config_tbl.add_row("Log level",       cfg.LOG_LEVEL)
    config_tbl.add_row("Input price",     f"${cfg.PRICE_INPUT_PER_M:.2f}/M tokens")
    config_tbl.add_row("Output price",    f"${cfg.PRICE_OUTPUT_PER_M:.2f}/M tokens")

    console.print(Columns([roster, config_tbl], equal=False, expand=False))


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _print_run_summary(
    state: CognivCrewState,
    duration: float,
    node_timings: dict[str, float],
) -> None:
    cost                 = usage_handler.estimated_cost()
    qa_iterations        = state.get("iteration", 0)
    architect_iterations = state.get("architect_iteration", 0)
    output_dir           = state.get("output_dir", "—")

    timing_lines = "\n".join(
        f"  {_AGENT_LABELS.get(node, node):<44} {t:.1f}s"
        for node, t in node_timings.items()
    )

    summary = (
        f"[bold]Total duration:[/bold]         {duration:.1f}s\n"
        f"[bold]Architect iterations:[/bold]   {architect_iterations}\n"
        f"[bold]QA iterations:[/bold]          {qa_iterations}\n"
        f"[bold]Input tokens:[/bold]           {usage_handler.input_tokens:,}\n"
        f"[bold]Output tokens:[/bold]          {usage_handler.output_tokens:,}\n"
        f"[bold]Total tokens:[/bold]           {usage_handler.total_tokens:,}\n"
        f"[bold]Estimated cost:[/bold]         ${cost:.4f} USD\n"
        f"[bold]Output path:[/bold]            {output_dir}\n"
        f"\n[bold]Per-agent timing:[/bold]\n{timing_lines}"
    )

    console.print(
        Panel(summary, title="[bold green]Run Summary[/bold green]", expand=False)
    )


if __name__ == "__main__":
    cli()
