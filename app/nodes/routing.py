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


def log_decision_debug(
    state: GraphState,
    retrieved: List[Dict[str, Any]],
    need_clarification: bool,
    next_node: str,
    turn_count: int,
    max_turns: int,
) -> None:
    try:
        topk = int(os.getenv("ROUTE_LOG_TOPK", "5"))
    except Exception:
        topk = 5
    if topk <= 0:
        return

    user_query = shorten_text(
        state.get("search_query") or state.get("user_query", ""), 120
    )
    cand_count = len(retrieved)

    top = retrieved[0] if retrieved else None
    top_score = top.get("score") if isinstance(top, dict) else None
    top_sources = ",".join(top.get("sources", [])) if isinstance(top, dict) else ""
    top_vec_raw = top.get("vec_raw") if isinstance(top, dict) else None

    top_score_str = (
        f"{float(top_score):.4f}" if isinstance(top_score, (int, float)) else "None"
    )
    top_vec_str = (
        f"{float(top_vec_raw):.3f}" if isinstance(top_vec_raw, (int, float)) else "None"
    )

    plan = f"bm25={bool(state.get('run_bm25', True))} vec={bool(state.get('run_vec', True))}"
    plan_reason = shorten_text(state.get("retrieval_plan_reason", ""), 40)

    logger.debug(
        f"[decision] turn={turn_count}/{max_turns} "
        f"need_clarification={need_clarification} next_node={next_node} "
        f"plan={plan} plan_reason={plan_reason} "
        f"candidates={cand_count} top_score={top_score_str} top_sources={top_sources} top_vec_raw={top_vec_str} "
        f'search_query="{user_query}"'
    )


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

    # IMPORTANT:
    # Return the destination node name directly to avoid extra branch labels like "clarify".
    if (not retrieved or need) and turn_count >= max_turns:
        next_node = "fallback"
    elif need:
        next_node = "followup_question"
    else:
        next_node = "answer"

    log_decision_debug(
        state=state,
        retrieved=retrieved,
        need_clarification=need,
        next_node=next_node,
        turn_count=turn_count,
        max_turns=max_turns,
    )

    return {
        "need_clarification": bool(need),
        "next_node": str(next_node),
    }
