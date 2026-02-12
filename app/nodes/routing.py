import os
import sys
from typing import Any, Dict, List

from app.core.config import get_env_float
from app.core.types import GraphState


def shorten_text(text: str, max_len: int) -> str:
    # Truncate for log readability.
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def log_retrieve_route_debug(
    state: GraphState,
    retrieved: List[Dict[str, Any]],
    need_clarification: bool,
    turn_count: int,
    max_turns: int,
) -> None:
    # Debug log for routing and retrieved references. Print to stderr to avoid
    # mixing with streamed stdout output in the answer node.
    try:
        topk = int(os.getenv("ROUTE_LOG_TOPK", "5"))
    except Exception:
        topk = 5
    if topk <= 0:
        return

    user_query = shorten_text(state.get("user_query", ""), 120)
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

    print(
        f"[retrieve-route-debug] turn={turn_count}/{max_turns} "
        f"need_clarification={need_clarification} "
        f"plan={plan} plan_reason={plan_reason} "
        f"candidates={cand_count} top_score={top_score_str} top_sources={top_sources} top_vec_raw={top_vec_str} "
        f'user_query="{user_query}"',
        file=sys.stderr,
        flush=True,
    )

    for i, r in enumerate(retrieved[:topk], start=1):
        it = r.get("item", {}) or {}
        rid = r.get("id", "")
        score = r.get("score", None)
        sources = ",".join(r.get("sources", []))
        bm25_rank = r.get("bm25_rank", None)
        vec_rank = r.get("vec_rank", None)
        vec_raw = r.get("vec_raw", None)

        q = shorten_text(it.get("question", ""), 80)
        url = shorten_text(it.get("url", ""), 160)

        score_str = f"{float(score):.4f}" if isinstance(score, (int, float)) else "None"
        bm25_rank_str = str(bm25_rank) if isinstance(bm25_rank, int) else "-"
        vec_rank_str = str(vec_rank) if isinstance(vec_rank, int) else "-"
        vec_raw_str = (
            f"{float(vec_raw):.3f}" if isinstance(vec_raw, (int, float)) else "-"
        )

        print(
            f"[retrieve-route-debug] #{i} id={rid} score={score_str} sources={sources} bm25_rank={bm25_rank_str} vec_rank={vec_rank_str} vec_raw={vec_raw_str} "
            f'Q="{q}" url="{url}"',
            file=sys.stderr,
            flush=True,
        )


def node_retrieve_route(state: GraphState) -> GraphState:
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

        # If the plan disables one retriever, relax the confidence heuristic accordingly.
        if run_bm25 and (not run_vec):
            # BM25-only mode: answer as long as we have candidates.
            need = False
        elif run_vec and (not run_bm25):
            # Vector-only mode: require at least a basic similarity threshold.
            need = not bool(top_vec_ok)
        else:
            top_has_both = bool(top.get("has_both", False))

            any_both_top3 = any(bool(r.get("has_both", False)) for r in retrieved[:3])
            any_vec_strong_top3 = any(
                isinstance(r.get("vec_raw"), (int, float))
                and float(r["vec_raw"]) >= vec_strong_threshold
                for r in retrieved[:3]
            )

            # Conservative confidence heuristic (hybrid mode):
            # - If the top candidate is supported by both retrievers and vector similarity is at least OK -> answer.
            # - If vector similarity is strong -> answer.
            # - If both sources agree among top-3 and vector is strong in top-3 -> answer.
            # Otherwise -> ask clarification.
            if top_has_both and top_vec_ok:
                need = False
            elif top_vec_strong:
                need = False
            elif any_both_top3 and any_vec_strong_top3:
                need = False
            else:
                need = True

    if turn_count >= max_turns:
        need = False

    log_retrieve_route_debug(
        state=state,
        retrieved=retrieved,
        need_clarification=need,
        turn_count=turn_count,
        max_turns=max_turns,
    )

    return {**state, "need_clarification": need}
