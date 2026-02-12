import os
from typing import Any, Dict, List, Tuple

import numpy as np

from app.core.config import get_env_float, get_env_int
from app.core.store import IndexedStore
from app.core.types import GraphState, MetaItem


def sorted_items_desc(m: Dict[int, float]) -> List[Tuple[int, float]]:
    return sorted(m.items(), key=lambda x: x[1], reverse=True)


def node_retrieval_router():
    def _run(state: GraphState) -> GraphState:
        # This node decides which retrievers to run.
        # Default: run both. You can override via env var RETRIEVAL_MODE.
        # Supported values: "both", "bm25", "vec"
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

        # Return only updated keys to avoid conflicts.
        return {
            "run_bm25": bool(run_bm25),
            "run_vec": bool(run_vec),
            "retrieval_plan_reason": str(reason),
        }

    return _run


# --- Legacy fusion retriever (kept for reference) --------------------------------
def rank_candidates_fusion(
    bm25_scores: Dict[int, float],
    vec_scores: Dict[int, float],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> List[Dict[str, Any]]:
    # Fusion strategy (replaceable later with other rerankers).
    all_ids = set(bm25_scores.keys()) | set(vec_scores.keys())

    def norm_map(m: Dict[int, float]) -> Dict[int, float]:
        if not m:
            return {}
        vals = np.array(list(m.values()), dtype=np.float32)
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax - vmin < 1e-8:
            return {k: 0.0 for k in m.keys()}
        return {k: (v - vmin) / (vmax - vmin) for k, v in m.items()}

    bm25_n = norm_map(bm25_scores)
    vec_n = norm_map(vec_scores)

    candidates: List[Dict[str, Any]] = []
    for rid in all_ids:
        b = float(bm25_n.get(rid, 0.0))
        v = float(vec_n.get(rid, 0.0))
        fused = alpha * b + beta * v
        candidates.append(
            {
                "id": int(rid),
                "fused_score": float(fused),
                "bm25_raw": float(bm25_scores.get(rid, 0.0)),
                "vec_raw": float(vec_scores.get(rid, 0.0)),
            }
        )

    candidates.sort(key=lambda x: x["fused_score"], reverse=True)
    return candidates


def retrieve_hybrid(
    store: IndexedStore,
    query: str,
    bm25_top_n: int = 50,
    vec_top_n: int = 50,
    final_top_k: int = 8,
) -> List[Dict[str, Any]]:
    bm25_scores = store.bm25_search(query, top_n=bm25_top_n)
    vec_scores = store.vec_search(query, top_n=vec_top_n)

    ranked = rank_candidates_fusion(bm25_scores, vec_scores, alpha=1.0, beta=1.0)[
        :final_top_k
    ]

    results: List[Dict[str, Any]] = []
    for c in ranked:
        it = store.meta_by_id.get(c["id"])
        if not it:
            continue
        results.append(
            {
                "item": it,
                "score": c["fused_score"],
                "bm25_raw": c["bm25_raw"],
                "vec_raw": c["vec_raw"],
            }
        )
    return results


def node_retrieve(store: IndexedStore):
    # Legacy single-node hybrid retriever (unused in the new graph).
    def _run(state: GraphState) -> GraphState:
        query = state["user_query"].strip()
        retrieved = retrieve_hybrid(store, query)
        return {**state, "retrieved": retrieved}

    return _run


# --- New retrieval pipeline (BM25 node + Vector node + Organizer node) -----------
def node_retrieve_bm25(store: IndexedStore):
    def _run(state: GraphState) -> GraphState:
        if not bool(state.get("run_bm25", True)):
            # Return empty results to avoid stale data in iterative runs.
            return {
                "bm25_retrieved": [],
                "bm25_count": 0,
            }

        query = state["user_query"].strip()
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

        query = state["user_query"].strip()

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


def node_organize_candidates():
    def _run(state: GraphState) -> GraphState:
        bm25 = state.get("bm25_retrieved", []) or []
        vec = state.get("vec_retrieved", []) or []

        rrf_k = max(1, get_env_int("RRF_K", 60))
        final_topk = max(1, get_env_int("FINAL_TOPK", 20))

        by_id: Dict[int, Dict[str, Any]] = {}

        for r in bm25:
            rid = int(r["id"])
            it: MetaItem = r["item"]
            by_id[rid] = {
                "id": rid,
                "item": it,
                "bm25_raw": r.get("bm25_raw"),
                "bm25_rank": r.get("bm25_rank"),
                "vec_raw": None,
                "vec_rank": None,
                "vec_pass_threshold": False,
                "sources": ["bm25"],
            }

        for r in vec:
            rid = int(r["id"])
            it: MetaItem = r["item"]
            if rid not in by_id:
                by_id[rid] = {
                    "id": rid,
                    "item": it,
                    "bm25_raw": None,
                    "bm25_rank": None,
                    "vec_raw": r.get("vec_raw"),
                    "vec_rank": r.get("vec_rank"),
                    "vec_pass_threshold": bool(r.get("vec_pass_threshold", False)),
                    "sources": ["vec"],
                }
            else:
                by_id[rid]["vec_raw"] = r.get("vec_raw")
                by_id[rid]["vec_rank"] = r.get("vec_rank")
                by_id[rid]["vec_pass_threshold"] = bool(
                    r.get("vec_pass_threshold", False)
                )
                if "vec" not in by_id[rid]["sources"]:
                    by_id[rid]["sources"].append("vec")

        organized: List[Dict[str, Any]] = []
        for _, r in by_id.items():
            bm25_rank = r.get("bm25_rank")
            vec_rank = r.get("vec_rank")

            score = 0.0
            if isinstance(bm25_rank, int) and bm25_rank > 0:
                score += 1.0 / float(rrf_k + bm25_rank)
            if isinstance(vec_rank, int) and vec_rank > 0:
                score += 1.0 / float(rrf_k + vec_rank)

            has_both = ("bm25" in r.get("sources", [])) and (
                "vec" in r.get("sources", [])
            )

            organized.append(
                {
                    **r,
                    "has_both": bool(has_both),
                    "score": float(score),
                }
            )

        organized.sort(key=lambda x: x["score"], reverse=True)
        organized = organized[:final_topk]

        return {
            **state,
            "retrieved": organized,
        }

    return _run
