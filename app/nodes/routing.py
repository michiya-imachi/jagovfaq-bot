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


def node_decide_next_action(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved", []) or []
    turn_count = int(state.get("turn_count", 0))
    max_turns = int(state.get("max_turns", 2))

    run_bm25 = bool(state.get("run_bm25", True))
    run_vec = bool(state.get("run_vec", True))

    vec_threshold = get_env_float("VEC_THRESHOLD", 0.35)
    vec_strong_threshold = get_env_float("VEC_STRONG_THRESHOLD", 0.45)

    if not retrieved:
        need = True
    else:
        top = retrieved[0]
        top_vec_raw = top.get("vec_raw")

        top_vec_ok = (
            isinstance(top_vec_raw, (int, float))
            and float(top_vec_raw) >= vec_threshold
        )
        top_vec_strong = (
            isinstance(top_vec_raw, (int, float))
            and float(top_vec_raw) >= vec_strong_threshold
        )

        if run_bm25 and (not run_vec):
            need = False
        elif run_vec and (not run_bm25):
            need = not bool(top_vec_ok)
        else:
            top_has_both = bool(top.get("has_both", False))
            any_both_top3 = any(bool(r.get("has_both", False)) for r in retrieved[:3])
            any_vec_strong_top3 = any(
                isinstance(r.get("vec_raw"), (int, float))
                and float(r["vec_raw"]) >= vec_strong_threshold
                for r in retrieved[:3]
            )

            if top_has_both and top_vec_ok:
                need = False
            elif top_vec_strong:
                need = False
            elif any_both_top3 and any_vec_strong_top3:
                need = False
            else:
                need = True

    if (not retrieved or need) and turn_count >= max_turns:
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
        "[decide] turn=%d/%d need_followup=%s",
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
