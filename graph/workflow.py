from langgraph.graph import StateGraph, START, END

from agents.ceo import ceo_node
from agents.pm import pm_node
from graph.state import CognivCrewState


def designer_node(state: CognivCrewState) -> CognivCrewState:
    return state


def engineer_node(state: CognivCrewState) -> CognivCrewState:
    return state


def qa_node(state: CognivCrewState) -> CognivCrewState:
    return state


def route_qa(state: CognivCrewState) -> str:
    if state.get("qa_verdict") == "FAIL" and state.get("iteration", 0) < 3:
        return "engineer"
    return END


builder = StateGraph(CognivCrewState)

builder.add_node("ceo", ceo_node)
builder.add_node("pm", pm_node)
builder.add_node("designer", designer_node)
builder.add_node("engineer", engineer_node)
builder.add_node("qa", qa_node)

builder.add_edge(START, "ceo")
builder.add_edge("ceo", "pm")
builder.add_edge("pm", "designer")
builder.add_edge("designer", "engineer")
builder.add_edge("engineer", "qa")

builder.add_conditional_edges("qa", route_qa)

app = builder.compile()
