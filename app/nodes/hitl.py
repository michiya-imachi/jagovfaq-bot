from typing import Any, Dict, List

from langgraph.types import interrupt

from app.core.types import GraphState


def node_hitl_wait_user_input():
    def _run(state: GraphState) -> GraphState:
        question = str(state.get("clarifying_question", "")).strip()
        payload = {
            "type": "HITL",
            "question": question,
        }

        # Pause execution and wait for external (human) input.
        answer = interrupt(payload)
        return {
            "clarifying_answer": str(answer).strip(),
        }

    return _run


def node_apply_followup():
    def _run(state: GraphState) -> GraphState:
        base = str(
            state.get("original_user_query") or state.get("user_query") or ""
        ).strip()
        q = str(state.get("clarifying_question", "")).strip()
        a = str(state.get("clarifying_answer", "")).strip()

        history: List[Dict[str, str]] = list(state.get("clarifications", []) or [])

        turn_count = int(state.get("turn_count", 0))
        if q and a:
            history.append({"question": q, "answer": a})
            turn_count += 1

        # Build an augmented query that keeps Q/A context explicit.
        lines: List[str] = []
        if base:
            lines.append(base)

        for qa in history:
            qq = str(qa.get("question", "")).strip()
            aa = str(qa.get("answer", "")).strip()
            if not qq or not aa:
                continue
            lines.append(f"追加確認: {qq}")
            lines.append(f"回答: {aa}")

        augmented = "\n".join(lines).strip()

        # Use the augmented text both for retrieval and for LLM-facing prompts.
        return {
            "clarifications": history,
            "search_query": augmented,
            "user_query": augmented,
            "turn_count": turn_count,
            "need_clarification": False,
            "clarifying_answer": "",
            "clarifying_question": "",
            "next_node": "",
        }

    return _run
