import logging
from typing import Any

from app.core.types import GraphState


logger = logging.getLogger(__name__)


def shorten_text(text: str, max_len: int) -> str:
    # Truncate for log readability.
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def make_decide_next_action(top_n: int = 1, candidate_source: str = "retrieved"):
    def _run(state: GraphState) -> GraphState:
        source_key = str(candidate_source or "retrieved")
        source_items = state.get(source_key, []) or []
        candidates = source_items[: max(1, int(top_n))]

        turn_count = int(state.get("turn_count", 0))
        max_turns = int(state.get("max_turns", 2))
        active_retrievers = [str(v) for v in (state.get("active_retrievers", []) or [])]

        if not candidates:
            need = True
        else:
            any_passed = any(bool(r.get("passed_any", False)) for r in candidates)
            any_multi = any(
                bool(r.get("has_multiple_sources", False)) for r in candidates
            )

            if len(active_retrievers) == 1 and active_retrievers[0] == "bm25":
                need = False
            elif len(active_retrievers) <= 1:
                need = not any_passed
            else:
                need = not (any_passed or any_multi)

        if (not candidates or need) and turn_count >= max_turns:
            next_node = "fallback"
            next_node_reason = "max_turns_reached"
        elif need:
            next_node = "followup_question"
            next_node_reason = "need_followup"
        else:
            next_node = "answer"
            next_node_reason = "enough_evidence"

        query = shorten_text(state.get("search_query", ""), 120)
        logger.debug(
            "[decide] source=%s top_n=%d turn=%d/%d need_followup=%s",
            source_key,
            max(1, int(top_n)),
            turn_count,
            max_turns,
            str(need),
        )
        logger.debug(
            "[decide] active_retrievers=%s",
            ",".join(active_retrievers),
        )
        logger.debug(
            '[decide] next_node=%s reason=%s search_query="%s"',
            next_node,
            next_node_reason,
            query,
        )

        return {
            "need_followup": bool(need),
            "next_node": str(next_node),
            "next_node_reason": str(next_node_reason),
        }

    return _run


def node_decide_next_action(state: GraphState) -> GraphState:
    # Keep backward compatibility for direct imports/tests.
    return make_decide_next_action(top_n=1, candidate_source="retrieved")(state)
