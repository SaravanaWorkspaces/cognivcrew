import atexit
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from agents.architect import architect_node
from agents.ceo import ceo_node
from agents.designer import designer_node
from agents.engineer import engineer_node
from agents.pm import pm_node
from agents.qa import qa_node
from config import cfg
from graph.state import CognivCrewState

console = Console()

_DELIVERABLES = [
    (cfg.FILE_PRODUCT_SPEC,          "Product specification — epics, stories, and acceptance criteria"),
    (cfg.FILE_ARCHITECT_BRIEF,       "Architecture brief — ADRs, tech stack, component diagram"),
    (cfg.FILE_DESIGN_BRIEF,          "Design brief — user journeys, IA, and interaction patterns"),
    (cfg.FILE_IMPLEMENTATION_PLAN,   "Implementation plan — architecture, data models, and API design"),
    (cfg.FILE_QA_REPORT,             "QA report — traceability matrix, issue log, and final verdict"),
]

# Ensure outputs dir exists before opening the checkpoint DB.
# SqliteSaver.from_conn_string returns a context manager; enter it manually
# and register cleanup so the connection closes cleanly on process exit.
Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)
_CHECKPOINT_DB = Path(cfg.OUTPUT_DIR) / "cognivcrew_checkpoints.db"
_checkpointer_cm = SqliteSaver.from_conn_string(str(_CHECKPOINT_DB))
checkpointer = _checkpointer_cm.__enter__()
atexit.register(_checkpointer_cm.__exit__, None, None, None)


# ---------------------------------------------------------------------------
# Non-LLM assembly node
# ---------------------------------------------------------------------------

def final_node(state: CognivCrewState) -> CognivCrewState:
    logger.info("Final assembly node starting")
    output_dir = Path(state["output_dir"])

    qa_verdict_text = state.get("qa_verdict", "")
    if "VERDICT: PASS" in qa_verdict_text:
        qa_result = "PASS"
    elif "VERDICT: FAIL" in qa_verdict_text:
        qa_result = "FAIL"
    else:
        qa_result = "unknown"

    summary_lines = [
        f"# CognivCrew Project Summary\n",
        f"**User Request:** {state['user_request']}\n",
        f"**Output Directory:** {output_dir}\n",
        f"**Architect Iterations:** {state.get('architect_iteration', 0)}\n",
        f"**QA Iterations:** {state.get('iteration', 0)}\n",
        f"**QA Verdict:** {qa_result}\n",
        "\n## Deliverables\n",
    ]
    for filename, description in _DELIVERABLES:
        file_path = output_dir / filename
        status = "✓" if file_path.exists() else "✗"
        summary_lines.append(f"- {status} `{filename}` — {description}\n")

    summary_path = output_dir / cfg.FILE_PROJECT_SUMMARY
    summary_path.write_text("".join(summary_lines))

    tree = Tree(f"[bold white]{output_dir}[/bold white]")
    for f in sorted(output_dir.iterdir()):
        tree.add(f"[green]{f.name}[/green]")

    console.print(
        Panel(
            tree,
            title="[bold green]Pipeline Complete — Output Directory[/bold green]",
            expand=False,
        )
    )
    console.print(
        f"\n[bold green]All deliverables saved to:[/bold green] [bold white]{output_dir}[/bold white]\n"
    )

    logger.info(f"Final assembly complete — summary written to {summary_path}")
    state["final_output"] = str(summary_path)
    return state


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_architect(state: CognivCrewState) -> str:
    approved = state.get("architect_approved", False)
    iterations = state.get("architect_iteration", 0)
    if approved or iterations >= cfg.MAX_ARCHITECT_ITERATIONS:
        logger.info(
            f"Architect routing → designer "
            f"(approved={approved}, iterations={iterations})"
        )
        return "designer"
    logger.info(
        f"Architect routing → architect again "
        f"(approved={approved}, iterations={iterations})"
    )
    return "architect"


def route_qa(state: CognivCrewState) -> str:
    verdict = state.get("qa_verdict", "")
    iteration = state.get("iteration", 0)
    if "VERDICT: FAIL" in verdict and iteration < cfg.MAX_ITERATIONS:
        logger.warning(
            f"QA FAIL — routing back to engineer "
            f"(iteration {iteration}/{cfg.MAX_ITERATIONS})"
        )
        return "engineer"
    return "final"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

builder = StateGraph(CognivCrewState)

builder.add_node("ceo", ceo_node)
builder.add_node("pm", pm_node)
builder.add_node("architect", architect_node)
builder.add_node("designer", designer_node)
builder.add_node("engineer", engineer_node)
builder.add_node("qa", qa_node)
builder.add_node("final", final_node)

builder.add_edge(START, "ceo")
builder.add_edge("ceo", "pm")
builder.add_edge("pm", "architect")
# Step 4 — Human Approval Gate: interrupt fires after architect; route_architect
# reads architect_approved / architect_iteration to decide next node.
builder.add_conditional_edges(
    "architect",
    route_architect,
    {"architect": "architect", "designer": "designer"},
)
builder.add_edge("designer", "engineer")
builder.add_edge("engineer", "qa")
builder.add_conditional_edges(
    "qa",
    route_qa,
    {"engineer": "engineer", "final": "final"},
)
builder.add_edge("final", END)

app = builder.compile(checkpointer=checkpointer, interrupt_after=["architect"])
