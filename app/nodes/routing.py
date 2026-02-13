import logging
import os
from typing import Any, Dict, List

from app.core.config import get_env_float
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


def make_decide_next_action(
    top_n: int = 1, candidate_source: str = "retrieved"
):
    def _run(state: GraphState) -> GraphState:
        source_key = str(candidate_source or "retrieved")
        source_items = state.get(source_key, []) or []
        candidates = source_items[: max(1, int(top_n))]

        turn_count = int(state.get("turn_count", 0))
        max_turns = int(state.get("max_turns", 2))

        run_bm25 = bool(state.get("run_bm25", True))
        run_vec = bool(state.get("run_vec", True))

        vec_threshold = get_env_float("VEC_THRESHOLD", 0.35)
        vec_strong_threshold = get_env_float("VEC_STRONG_THRESHOLD", 0.45)

        if not candidates:
            need = True
        else:
            vec_values = [
                float(r["vec_raw"])
                for r in candidates
                if isinstance(r.get("vec_raw"), (int, float))
            ]
            any_vec_ok = any(v >= vec_threshold for v in vec_values)
            any_vec_strong = any(v >= vec_strong_threshold for v in vec_values)
            any_both = any(bool(r.get("has_both", False)) for r in candidates)
            any_both_and_vec_ok = any(
                bool(r.get("has_both", False))
                and isinstance(r.get("vec_raw"), (int, float))
                and float(r["vec_raw"]) >= vec_threshold
                for r in candidates
            )

            if run_bm25 and (not run_vec):
                need = False
            elif run_vec and (not run_bm25):
                need = not bool(any_vec_ok)
            else:
                if any_both_and_vec_ok:
                    need = False
                elif any_vec_strong:
                    need = False
                elif any_both and any_vec_strong:
                    need = False
                else:
                    need = True

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
