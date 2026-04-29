from typing import TypedDict


class CognivCrewState(TypedDict, total=False):
    user_request: str
    strategy: str
    product_spec: str
    architect_brief: str
    human_feedback: str
    architect_approved: bool
    architect_iteration: int
    design_brief: str
    implementation_plan: str
    qa_verdict: str
    qa_feedback: str
    iteration: int
    final_output: str
    output_dir: str


def default_state() -> CognivCrewState:
    return CognivCrewState(
        user_request="",
        strategy="",
        product_spec="",
        architect_brief="",
        human_feedback="",
        architect_approved=False,
        architect_iteration=0,
        design_brief="",
        implementation_plan="",
        qa_verdict="",
        qa_feedback="",
        iteration=0,
        final_output="",
        output_dir="",
    )
