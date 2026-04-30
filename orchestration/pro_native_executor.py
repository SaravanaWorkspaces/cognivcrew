"""
Pro Native Executor — runs the CognivCrew pipeline using the Claude Code CLI
so that both planning and code-generation phases consume Claude Pro plan tokens.
No Anthropic API key is required.

Phase 1 — Planning : CEO → PM → Architect (with approval gate) → Designer
Phase 2 — Code Gen : Engineer → QA (with QA retry loop)
"""
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel

from config import cfg
from config.execution import PRO_NATIVE_CONFIG
from graph.state import CognivCrewState

console = Console()

_CLAUDE_CLI = "claude"
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ─────────────────────────────────────────────────────────────────────────────
# Token estimation (Pro plan does not expose exact counts via the CLI)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenEstimate:
    """Tracks character counts and converts them to approximate token counts."""
    input_chars: int = 0
    output_chars: int = 0
    _chars_per_token: int = field(default_factory=lambda: PRO_NATIVE_CONFIG["chars_per_token"])

    @property
    def estimated_input_tokens(self) -> int:
        return self.input_chars // self._chars_per_token

    @property
    def estimated_output_tokens(self) -> int:
        return self.output_chars // self._chars_per_token

    @property
    def estimated_total_tokens(self) -> int:
        return self.estimated_input_tokens + self.estimated_output_tokens

    @property
    def cost_display(self) -> str:
        return PRO_NATIVE_CONFIG["cost_display"]


# ─────────────────────────────────────────────────────────────────────────────
# ProNativeExecutor
# ─────────────────────────────────────────────────────────────────────────────

class ProNativeExecutor:
    """
    Executes the full CognivCrew agent pipeline via Claude Code CLI.

    Both planning and code-generation phases use the same Claude Pro plan
    allocation — no mixing of Pro and API tokens occurs.

    Usage::

        executor = ProNativeExecutor()
        state = executor.run_planning_phase(state, approval_callback=handle_approval_gate)
        state = executor.run_codegen_phase(state)
        print(executor.cost_report())
    """

    def __init__(self) -> None:
        self.token_estimate = TokenEstimate()
        self._agent_timings: dict[str, float] = {}
        self._agent_tokens: dict[str, int] = {}

    # ── Auth / availability ───────────────────────────────────────────────

    @staticmethod
    def is_cli_available() -> bool:
        """Return True if the Claude Code CLI binary is on PATH."""
        return shutil.which(_CLAUDE_CLI) is not None

    @staticmethod
    def check_auth() -> tuple[bool, str]:
        """
        Verify that Claude Code CLI is installed and authenticated.

        Returns:
            (is_ok, human_readable_message)
        """
        if not ProNativeExecutor.is_cli_available():
            return (
                False,
                "Claude Code CLI not found. Install it with:\n"
                "  npm install -g @anthropic-ai/claude-code",
            )

        # Quick version check — ensures the binary is executable.
        try:
            result = subprocess.run(
                [_CLAUDE_CLI, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return False, f"Claude Code CLI error: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return False, "Claude Code CLI timed out during version check."
        except OSError as exc:
            return False, f"Could not launch Claude Code CLI: {exc}"

        # Auth probe: a trivial prompt that requires a valid session.
        try:
            result = subprocess.run(
                [_CLAUDE_CLI, "--print", "Reply with the single word OK"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return True, "Claude Code CLI authenticated — Pro plan tokens active"

            stderr = result.stderr.strip().lower()
            if any(kw in stderr for kw in ("auth", "login", "not logged", "sign in", "unauthorized")):
                return (
                    False,
                    "Claude Code is not authenticated.\n"
                    "Run:  cognivcrew auth-pro   (or 'claude' in your terminal)",
                )
            return False, f"Claude Code returned an error:\n{result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return False, "Authentication probe timed out (30 s)."
        except OSError as exc:
            return False, f"Could not run authentication probe: {exc}"

    # ── Core LLM call ─────────────────────────────────────────────────────

    def _call_claude(self, prompt: str, system_prompt: str = "") -> str:
        """
        Call the Claude Code CLI in non-interactive print mode.

        The system prompt is prepended to the human prompt and piped via stdin
        to avoid command-line argument length limits on long prompts.
        """
        full_prompt = f"{system_prompt}\n\n---\n\n{prompt}" if system_prompt else prompt
        self.token_estimate.input_chars += len(full_prompt)

        timeout = PRO_NATIVE_CONFIG["call_timeout_seconds"]
        try:
            # Pipe prompt via stdin to avoid OS argument-length limits on
            # large prompts (Architect/Engineer/QA receive many KB of context).
            result = subprocess.run(
                [_CLAUDE_CLI, "--print", "-"],
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Claude Code CLI timed out after {timeout}s. "
                "Try increasing 'call_timeout_seconds' in config/execution.py."
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to launch Claude Code CLI: {exc}")

        if result.returncode != 0:
            raise RuntimeError(
                f"Claude Code CLI exited with code {result.returncode}:\n"
                f"{result.stderr.strip()}"
            )

        response = result.stdout.strip()
        self.token_estimate.output_chars += len(response)
        return response

    def _load_prompt(self, agent_name: str) -> str:
        path = _PROMPTS_DIR / f"{agent_name}_prompt.txt"
        return path.read_text()

    def _run_agent(self, agent_name: str, human_message: str) -> str:
        """Invoke one agent via Claude Code CLI; record timing and token estimate."""
        t0 = time.perf_counter()
        system_prompt = self._load_prompt(agent_name)

        tokens_before = self.token_estimate.estimated_total_tokens
        response = self._call_claude(human_message, system_prompt)
        tokens_delta = self.token_estimate.estimated_total_tokens - tokens_before

        elapsed = time.perf_counter() - t0
        self._agent_timings[agent_name] = elapsed
        self._agent_tokens[agent_name] = tokens_delta

        logger.info(
            f"{agent_name.upper()} agent complete — "
            f"{elapsed:.1f}s, ~{tokens_delta:,} est. tokens"
        )
        return response

    # ── Phase 1 — Planning ────────────────────────────────────────────────

    def run_planning_phase(
        self,
        state: CognivCrewState,
        approval_callback: Optional[Callable[[CognivCrewState], dict]] = None,
        max_architect_iterations: int = 3,
    ) -> CognivCrewState:
        """
        Run the planning phase using Pro plan tokens.

        Agents: CEO → PM → Architect (approval loop) → Designer

        Args:
            state: Pipeline state (mutated in place and returned).
            approval_callback: Called after each Architect run; must return a
                dict with keys ``architect_approved`` (bool) and
                ``human_feedback`` (str).  When omitted the brief is
                auto-approved.
            max_architect_iterations: Force-approve after this many attempts.
        """
        logger.info("Pro Native — Phase 1 (Planning) starting")
        console.print(Panel(
            "[bold cyan]Phase 1 — Planning[/bold cyan]\n"
            "[dim]CEO → PM → Architect → Designer  |  Pro plan tokens[/dim]",
            title="[bold]Pro Native Executor[/bold]",
            expand=False,
            border_style="cyan",
        ))

        output_dir = Path(state["output_dir"])

        # ── CEO ──────────────────────────────────────────────────────────
        state["strategy"] = self._run_agent("ceo", state["user_request"])
        console.print(Panel(
            "[bold green]CEO strategy complete.[/bold green]",
            title="[bold cyan]CEO Agent[/bold cyan]",
            expand=False,
        ))

        # ── PM ───────────────────────────────────────────────────────────
        pm_input = (
            f"## User Request\n\n{state['user_request']}\n\n"
            f"## CEO Strategy\n\n{state['strategy']}"
        )
        state["product_spec"] = self._run_agent("pm", pm_input)
        (output_dir / cfg.FILE_PRODUCT_SPEC).write_text(state["product_spec"])
        console.print(Panel(
            "[bold green]Product specification complete.[/bold green]\n"
            f"Saved to: [white]{output_dir / cfg.FILE_PRODUCT_SPEC}[/white]",
            title="[bold cyan]PM Agent[/bold cyan]",
            expand=False,
        ))

        # ── Architect (with approval loop) ───────────────────────────────
        state.setdefault("architect_iteration", 0)
        state.setdefault("human_feedback", "")

        for arch_iter in range(max_architect_iterations):
            state["architect_iteration"] = arch_iter + 1

            feedback_block = ""
            if state.get("human_feedback"):
                feedback_block = (
                    f"## Human Feedback (Revision {arch_iter})\n\n"
                    f"{state['human_feedback']}\n\n"
                )

            architect_input = (
                f"## User Request\n\n{state['user_request']}\n\n"
                f"## CEO Strategy\n\n{state['strategy']}\n\n"
                f"## Product Specification\n\n{state['product_spec']}\n\n"
                f"{feedback_block}"
            ).rstrip()

            state["architect_brief"] = self._run_agent("architect", architect_input)
            (output_dir / cfg.FILE_ARCHITECT_BRIEF).write_text(state["architect_brief"])
            console.print(Panel(
                f"[bold green]Architecture brief complete (iteration {arch_iter + 1}).[/bold green]\n"
                f"Saved to: [white]{output_dir / cfg.FILE_ARCHITECT_BRIEF}[/white]",
                title="[bold cyan]Architect Agent[/bold cyan]",
                expand=False,
            ))

            # Auto-approve at max iterations
            if arch_iter >= max_architect_iterations - 1:
                logger.warning(
                    f"Max architect iterations ({max_architect_iterations}) reached — force-approving"
                )
                console.print(Panel(
                    f"[bold yellow]Maximum architect iterations ({max_architect_iterations}) reached.[/bold yellow]\n"
                    "Architecture brief auto-approved.",
                    title="[bold yellow]Architect — Max Iterations[/bold yellow]",
                    expand=False,
                ))
                state["architect_approved"] = True
                state["human_feedback"] = ""
                break

            if approval_callback is not None:
                gate = approval_callback(state)
                state["architect_approved"] = gate.get("architect_approved", True)
                state["human_feedback"] = gate.get("human_feedback", "")
            else:
                state["architect_approved"] = True
                state["human_feedback"] = ""

            if state["architect_approved"]:
                break

        # ── Designer ─────────────────────────────────────────────────────
        designer_input = (
            f"## User Request\n\n{state['user_request']}\n\n"
            f"## Product Specification\n\n{state['product_spec']}\n\n"
            f"## Architecture Brief\n\n{state['architect_brief']}"
        )
        state["design_brief"] = self._run_agent("designer", designer_input)
        (output_dir / cfg.FILE_DESIGN_BRIEF).write_text(state["design_brief"])
        console.print(Panel(
            "[bold green]Design brief complete.[/bold green]\n"
            f"Saved to: [white]{output_dir / cfg.FILE_DESIGN_BRIEF}[/white]",
            title="[bold cyan]Designer Agent[/bold cyan]",
            expand=False,
        ))

        logger.info("Pro Native — Phase 1 (Planning) complete")
        return state

    # ── Phase 2 — Code Generation ─────────────────────────────────────────

    def run_codegen_phase(
        self,
        state: CognivCrewState,
        max_qa_iterations: int = 3,
    ) -> CognivCrewState:
        """
        Run the code-generation phase using Pro plan tokens.

        Agents: Engineer → QA (with retry loop on FAIL verdict)

        Args:
            state: Pipeline state from ``run_planning_phase`` (mutated in place).
            max_qa_iterations: Maximum Engineer/QA revision cycles.
        """
        logger.info("Pro Native — Phase 2 (Code Gen) starting")
        console.print(Panel(
            "[bold cyan]Phase 2 — Code Generation[/bold cyan]\n"
            "[dim]Engineer → QA  |  Pro plan tokens[/dim]",
            title="[bold]Pro Native Executor[/bold]",
            expand=False,
            border_style="cyan",
        ))

        output_dir = Path(state["output_dir"])
        state.setdefault("iteration", 0)
        state.setdefault("qa_feedback", "")

        for qa_iter in range(max_qa_iterations):
            state["iteration"] = qa_iter

            # ── Engineer ─────────────────────────────────────────────────
            feedback_block = ""
            if state.get("qa_feedback", "").strip():
                feedback_block = (
                    f"## QA Feedback (Revision Required)\n\n"
                    f"{state['qa_feedback']}\n\n"
                )

            engineer_input = (
                f"## Original User Request\n\n{state['user_request']}\n\n"
                f"## CEO Strategy\n\n{state['strategy']}\n\n"
                f"## Product Specification\n\n{state['product_spec']}\n\n"
                f"## Design Brief\n\n{state['design_brief']}\n\n"
                f"{feedback_block}"
            ).rstrip()

            state["implementation_plan"] = self._run_agent("engineer", engineer_input)
            (output_dir / cfg.FILE_IMPLEMENTATION_PLAN).write_text(state["implementation_plan"])

            iteration_label = f" (revision {qa_iter})" if qa_iter > 0 else ""
            console.print(Panel(
                f"[bold green]Implementation plan complete{iteration_label}.[/bold green]\n"
                f"Saved to: [white]{output_dir / cfg.FILE_IMPLEMENTATION_PLAN}[/white]",
                title="[bold cyan]Engineer Agent[/bold cyan]",
                expand=False,
            ))

            # ── QA ───────────────────────────────────────────────────────
            qa_input = (
                f"## User Request\n\n{state['user_request']}\n\n"
                f"## Product Specification\n\n{state['product_spec']}\n\n"
                f"## Architecture Brief\n\n{state['architect_brief']}\n\n"
                f"## Design Brief\n\n{state['design_brief']}\n\n"
                f"## Implementation Plan\n\n{state['implementation_plan']}"
            )

            state["qa_verdict"] = self._run_agent("qa", qa_input)
            (output_dir / cfg.FILE_QA_REPORT).write_text(state["qa_verdict"])

            console.print(Panel(
                f"[bold green]QA review complete{iteration_label}.[/bold green]\n"
                f"Saved to: [white]{output_dir / cfg.FILE_QA_REPORT}[/white]",
                title="[bold cyan]QA Agent[/bold cyan]",
                expand=False,
            ))

            # ── Verdict routing ──────────────────────────────────────────
            if "VERDICT: FAIL" in state["qa_verdict"] and qa_iter < max_qa_iterations - 1:
                fail_lines: list[str] = []
                in_fail_section = False
                for line in state["qa_verdict"].splitlines():
                    if "FAIL REASON" in line.upper():
                        in_fail_section = True
                    if in_fail_section:
                        fail_lines.append(line)
                state["qa_feedback"] = "\n".join(fail_lines)
                logger.warning(
                    f"QA FAIL (iteration {qa_iter}) — engineer will revise"
                )
            else:
                state["qa_feedback"] = ""
                break

        logger.info("Pro Native — Phase 2 (Code Gen) complete")
        return state

    # ── Reporting ─────────────────────────────────────────────────────────

    def cost_report(self) -> str:
        """Human-readable cost and token-estimate summary."""
        est = self.token_estimate
        return (
            f"Estimated tokens : ~{est.estimated_total_tokens:,} "
            f"(input ~{est.estimated_input_tokens:,}, "
            f"output ~{est.estimated_output_tokens:,})\n"
            f"Cost             : {est.cost_display}"
        )

    @property
    def agent_timings(self) -> dict[str, float]:
        return dict(self._agent_timings)

    @property
    def agent_tokens(self) -> dict[str, int]:
        return dict(self._agent_tokens)
