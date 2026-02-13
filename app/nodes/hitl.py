from langgraph.types import interrupt

from app.core.types import GraphState


def node_hitl_wait_user_input():
    def _run(state: GraphState) -> GraphState:
        question = str(state.get("followup_question", "")).strip()
        payload = {"type": "HITL", "question": question}

        # Pause execution and wait for external (human) input.
        answer = interrupt(payload)
        return {"followup_answer": str(answer).strip()}

    return _run
