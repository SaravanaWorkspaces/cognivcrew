from pathlib import Path

from langgraph.graph import END, START, StateGraph
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from agents.ceo import ceo_node
from agents.designer import designer_node
from agents.engineer import engineer_node
from agents.pm import pm_node
from agents.qa import qa_node
from config import cfg
from graph.state import CognivCrewState

console = Console()

_DELIVERABLES = [
    ("01_product_spec.md", "Product specification — epics, stories, and acceptance criteria"),
    ("02_design_brief.md", "Design brief — user journeys, IA, and interaction patterns"),
    ("03_implementation_plan.md", "Implementation plan — architecture, data models, and API design"),
    ("04_qa_report.md", "QA report — traceability matrix, issue log, and final verdict"),
]


def final_node(state: CognivCrewState) -> CognivCrewState:
    logger.info("Final assembly node starting")
    output_dir = Path(state["output_dir"])

    summary_lines = [
        f"# CognivCrew Project Summary\n",
        f"**User Request:** {state['user_request']}\n",
        f"**Output Directory:** {output_dir}\n",
        f"**QA Iterations:** {state.get('iteration', 0)}\n",
        "\n## Deliverables\n",
    ]
    for filename, description in _DELIVERABLES:
        file_path = output_dir / filename
        status = "✓" if file_path.exists() else "✗"
        summary_lines.append(f"- {status} `{filename}` — {description}\n")

    summary_path = output_dir / "00_project_summary.md"
    summary_path.write_text("".join(summary_lines))

    tree = Tree(f"[bold white]{output_dir}[/bold white]")
    all_files = sorted(output_dir.iterdir())
    for f in all_files:
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


def route_qa(state: CognivCrewState) -> str:
    verdict = state.get("qa_verdict", "")
    iteration = state.get("iteration", 0)
    if "VERDICT: FAIL" in verdict and iteration < cfg.MAX_ITERATIONS:
        logger.warning(f"QA FAIL — routing back to engineer (iteration {iteration}/{cfg.MAX_ITERATIONS})")
        return "engineer"
    return "final"


builder = StateGraph(CognivCrewState)

builder.add_node("ceo", ceo_node)
builder.add_node("pm", pm_node)
builder.add_node("designer", designer_node)
builder.add_node("engineer", engineer_node)
builder.add_node("qa", qa_node)
builder.add_node("final", final_node)

builder.add_edge(START, "ceo")
builder.add_edge("ceo", "pm")
builder.add_edge("pm", "designer")
builder.add_edge("designer", "engineer")
builder.add_edge("engineer", "qa")
builder.add_conditional_edges("qa", route_qa, {"engineer": "engineer", "final": "final"})
builder.add_edge("final", END)

app = builder.compile()
