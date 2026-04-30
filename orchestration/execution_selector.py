"""
Execution mode selector — resolves which executor to use for a pipeline run.

Modes
-----
api         Default. Uses the Anthropic API via ANTHROPIC_API_KEY.
            The existing LangGraph workflow handles execution.
pro_native  Uses Claude Code CLI with the user's Claude Pro plan.
            No API key required. Returns a ProNativeExecutor instance.
mock        Zero LLM calls. Pre-canned outputs for every pipeline stage.
            Engineer stage always produces a hello_world.py application.

Test layer
----------
Set TEST_MODE=true in your .env to force mock mode regardless of EXECUTION_MODE.
Useful for CI, smoke tests, and pipeline validation without any API costs.
"""
import os
from enum import Enum
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel

from config.execution import EXECUTION_MODE as _ENV_MODE, TEST_MODE

console = Console()


class ExecutionMode(str, Enum):
    API = "api"
    PRO_NATIVE = "pro_native"
    MOCK = "mock"


def get_execution_mode() -> ExecutionMode:
    """
    Resolve the active execution mode from the environment.

    TEST_MODE=true overrides EXECUTION_MODE and always returns MOCK so that
    CI and smoke-test runs never touch the LLM APIs.
    Falls back to API on unknown EXECUTION_MODE values.
    """
    if TEST_MODE:
        logger.info("TEST_MODE=true — forcing execution mode to MOCK")
        return ExecutionMode.MOCK

    raw = _ENV_MODE.lower().strip()
    try:
        return ExecutionMode(raw)
    except ValueError:
        logger.warning(f"Unknown EXECUTION_MODE='{raw}' — falling back to 'api'")
        return ExecutionMode.API


def select_executor(mode: Optional[ExecutionMode] = None):
    """
    Return the appropriate executor for *mode*.

    - ``ExecutionMode.API``        → ``None``  (caller uses LangGraph workflow)
    - ``ExecutionMode.PRO_NATIVE`` → ``ProNativeExecutor`` after auth check

    Raises ``SystemExit(1)`` when Pro Native auth fails so the CLI surfaces
    a clear error rather than an obscure traceback.
    """
    if mode is None:
        mode = get_execution_mode()

    if mode == ExecutionMode.API:
        logger.info("Execution mode: API — using ANTHROPIC_API_KEY")
        return None

    if mode == ExecutionMode.PRO_NATIVE:
        # Import here so that API-only installs don't require claude on PATH.
        from orchestration.pro_native_executor import ProNativeExecutor

        ok, message = ProNativeExecutor.check_auth()
        if not ok:
            console.print(Panel(
                f"[bold red]Pro Native authentication failed[/bold red]\n\n"
                f"{message}\n\n"
                f"[yellow]To authenticate:[/yellow]  [bold]cognivcrew auth-pro[/bold]\n"
                f"[dim]Or switch back to API mode:[/dim]  "
                f"[bold]EXECUTION_MODE=api[/bold] with [bold]ANTHROPIC_API_KEY[/bold] set",
                title="[bold red]Authentication Error — Pro Native Mode[/bold red]",
                expand=False,
            ))
            raise SystemExit(1)

        logger.info(f"Execution mode: PRO_NATIVE — {message}")
        console.print(Panel(
            f"[bold green]{message}[/bold green]\n"
            f"[dim]Cost: $0 (included in Claude Pro plan)[/dim]\n"
            f"[dim]Token counts are estimated (Pro plan does not expose exact counts)[/dim]",
            title="[bold cyan]Pro Native Mode — Active[/bold cyan]",
            expand=False,
        ))
        return ProNativeExecutor()

    if mode == ExecutionMode.MOCK:
        from mock.mock_executor import MockExecutor
        test_layer_note = (
            "\n[bold red]TEST_MODE=true — all LLM calls suppressed.[/bold red]"
            if TEST_MODE else ""
        )
        logger.info("Execution mode: MOCK — no LLM calls, zero cost")
        console.print(Panel(
            "[bold yellow]Mock mode active — no LLM calls will be made.[/bold yellow]\n"
            "[dim]Cost: $0.00  |  All outputs are pre-canned content.[/dim]\n"
            "[dim]Engineer stage produces a hello_world.py application.[/dim]"
            + test_layer_note,
            title="[bold yellow]Mock Mode — Active[/bold yellow]",
            expand=False,
        ))
        return MockExecutor()

    raise ValueError(f"Unhandled execution mode: {mode!r}")
