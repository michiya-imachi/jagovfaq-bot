import logging
import os
from typing import Any, Dict, List, Tuple

from app.core.config import get_env_float, get_env_int
from app.core.store import IndexedStore
from app.core.types import GraphState


logger = logging.getLogger(__name__)


def shorten_text(text: Any, max_len: int) -> str:
    # Truncate for log readability.
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def sorted_items_desc(m: Dict[int, float]) -> List[Tuple[int, float]]:
    return sorted(m.items(), key=lambda x: x[1], reverse=True)


def node_retrieval_router():
    def _run(state: GraphState) -> GraphState:
        # Decide which retrievers to run based on env var RETRIEVAL_MODE.
        mode = str(os.getenv("RETRIEVAL_MODE", "")).strip().lower()

        if mode in {"bm25", "keyword"}:
            run_bm25, run_vec, reason = True, False, f"env:{mode}"
        elif mode in {"vec", "vector"}:
            run_bm25, run_vec, reason = False, True, f"env:{mode}"
        else:
            run_bm25, run_vec, reason = (
                True,
                True,
                ("env:both" if mode in {"both", "hybrid"} else "default_both"),
            )

        if run_bm25 and run_vec:
            plan = "bm25+vec"
        elif run_bm25:
            plan = "bm25_only"
        elif run_vec:
            plan = "vec_only"
        else:
            plan = "none"

        user_query = shorten_text(
            state.get("search_query") or state.get("user_query", ""), 120
        )
        logger.info(
            '[retrieval-router] mode="%s" plan=%s reason=%s search_query="%s"',
            mode,
            plan,
            reason,
            user_query,
        )

        # Return only updated keys to avoid conflicts.
        return {
            "run_bm25": bool(run_bm25),
            "run_vec": bool(run_vec),
            "retrieval_plan_reason": str(reason),
        }

    return _run


def node_retrieve_bm25(store: IndexedStore):
    def _run(state: GraphState) -> GraphState:
        if not bool(state.get("run_bm25", True)):
            # Return empty results to avoid stale data in iterative runs.
            return {
                "bm25_retrieved": [],
                "bm25_count": 0,
            }

        query = str(state.get("search_query") or state.get("user_query") or "").strip()
        topk = max(1, get_env_int("BM25_TOPK", 10))

        bm25_scores = store.bm25_search(query, top_n=topk)
        ranked = sorted_items_desc(bm25_scores)

        out: List[Dict[str, Any]] = []
        for rank, (rid, score) in enumerate(ranked, start=1):
            it = store.meta_by_id.get(int(rid))
            if not it:
                continue
            out.append(
                {
                    "id": int(rid),
                    "item": it,
                    "bm25_raw": float(score),
                    "bm25_rank": int(rank),
                }
            )

        # Return only updated keys to avoid conflicts in parallel execution.
        return {
            "bm25_retrieved": out,
            "bm25_count": len(out),
        }

    return _run


def node_retrieve_vec_threshold(store: IndexedStore):
    def _run(state: GraphState) -> GraphState:
        if not bool(state.get("run_vec", True)):
            # Return empty results to avoid stale data in iterative runs.
            return {
                "vec_retrieved": [],
                "vec_count": 0,
                "vec_pass_count": 0,
            }

        query = str(state.get("search_query") or state.get("user_query") or "").strip()

        search_topn = max(1, get_env_int("VEC_SEARCH_TOPN", 200))
        threshold = get_env_float("VEC_THRESHOLD", 0.35)
        max_keep = max(1, get_env_int("VEC_MAX_KEEP", 30))
        fallback_topk = max(1, get_env_int("VEC_FALLBACK_TOPK", 30))

        vec_scores = store.vec_search(query, top_n=search_topn)
        ranked = sorted_items_desc(vec_scores)

        passed: List[Tuple[int, float, int]] = []
        for rank, (rid, score) in enumerate(ranked, start=1):
            if float(score) >= threshold:
                passed.append((int(rid), float(score), int(rank)))
                if len(passed) >= max_keep:
                    break

        use_fallback = len(passed) == 0
        picked: List[Tuple[int, float, int]] = []
        if not use_fallback:
            picked = passed
        else:
            # Fallback: keep top results to avoid empty candidate set.
            for rank, (rid, score) in enumerate(ranked[:fallback_topk], start=1):
                picked.append((int(rid), float(score), int(rank)))
                if len(picked) >= max_keep:
                    break

        out: List[Dict[str, Any]] = []
        for rid, score, rank in picked:
            it = store.meta_by_id.get(int(rid))
            if not it:
                continue
            out.append(
                {
                    "id": int(rid),
                    "item": it,
                    "vec_raw": float(score),
                    "vec_rank": int(rank),
                    "vec_pass_threshold": (not use_fallback)
                    and (float(score) >= threshold),
                }
            )

        pass_count = sum(1 for r in out if r.get("vec_pass_threshold", False))

        # Return only updated keys to avoid conflicts in parallel execution.
        return {
            "vec_retrieved": out,
            "vec_count": len(out),
            "vec_pass_count": int(pass_count),
        }

    return _run
