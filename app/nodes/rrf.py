import logging
from typing import Any, Dict, List

from app.core.config import get_env_int
from app.core.types import GraphState, MetaItem


logger = logging.getLogger(__name__)


def shorten_text(text: Any, max_len: int) -> str:
    # Truncate for log readability.
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def node_rrf_rank():
    def _run(state: GraphState) -> GraphState:
        # Merge by id -> compute RRF score -> sort ALL candidates -> log ALL at INFO.
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

            bm25_contrib = 0.0
            vec_contrib = 0.0

            if isinstance(bm25_rank, int) and bm25_rank > 0:
                bm25_contrib = 1.0 / float(rrf_k + bm25_rank)
            if isinstance(vec_rank, int) and vec_rank > 0:
                vec_contrib = 1.0 / float(rrf_k + vec_rank)

            score = float(bm25_contrib + vec_contrib)

            has_both = ("bm25" in r.get("sources", [])) and (
                "vec" in r.get("sources", [])
            )

            organized.append(
                {
                    **r,
                    "has_both": bool(has_both),
                    "score": score,
                    "bm25_contrib": float(bm25_contrib),
                    "vec_contrib": float(vec_contrib),
                }
            )

        organized.sort(key=lambda x: x["score"], reverse=True)

        for i, r in enumerate(organized, start=1):
            r["final_rank"] = int(i)
            r["keep"] = bool(i <= final_topk)

        run_bm25 = bool(state.get("run_bm25", True))
        run_vec = bool(state.get("run_vec", True))
        reason = shorten_text(state.get("retrieval_plan_reason", ""), 60)

        bm25_count = int(state.get("bm25_count", len(bm25)))
        vec_count = int(state.get("vec_count", len(vec)))
        vec_pass = int(state.get("vec_pass_count", 0))

        user_query = shorten_text(state.get("user_query", ""), 120)

        logger.info(
            '[rrf-rank] plan=bm25=%s vec=%s reason=%s bm25_count=%d vec_count=%d vec_pass=%d rrf_k=%d final_topk=%d merged=%d user_query="%s"',
            run_bm25,
            run_vec,
            reason,
            bm25_count,
            vec_count,
            vec_pass,
            rrf_k,
            final_topk,
            len(organized),
            user_query,
        )

        for r in organized:
            final_rank = int(r.get("final_rank", 0))
            keep = bool(r.get("keep", False))
            rid = r.get("id", "")

            score = r.get("score", None)
            bm25_contrib = r.get("bm25_contrib", None)
            vec_contrib = r.get("vec_contrib", None)

            score_str = (
                f"{float(score):.6f}" if isinstance(score, (int, float)) else "None"
            )
            bm25_contrib_str = (
                f"{float(bm25_contrib):.6f}"
                if isinstance(bm25_contrib, (int, float))
                else "-"
            )
            vec_contrib_str = (
                f"{float(vec_contrib):.6f}"
                if isinstance(vec_contrib, (int, float))
                else "-"
            )

            sources = ",".join(r.get("sources", []))

            bm25_rank = r.get("bm25_rank", None)
            vec_rank = r.get("vec_rank", None)
            bm25_raw = r.get("bm25_raw", None)
            vec_raw = r.get("vec_raw", None)
            vec_pass_threshold = bool(r.get("vec_pass_threshold", False))

            bm25_rank_str = str(bm25_rank) if isinstance(bm25_rank, int) else "-"
            vec_rank_str = str(vec_rank) if isinstance(vec_rank, int) else "-"
            bm25_raw_str = (
                f"{float(bm25_raw):.4f}" if isinstance(bm25_raw, (int, float)) else "-"
            )
            vec_raw_str = (
                f"{float(vec_raw):.4f}" if isinstance(vec_raw, (int, float)) else "-"
            )

            it = r.get("item", {}) or {}
            q = shorten_text(it.get("question", ""), 80)
            url = shorten_text(it.get("url", ""), 160)

            logger.info(
                '[rrf-rank] #%d keep=%s id=%s score=%s bm25_contrib=%s vec_contrib=%s sources=%s bm25_rank=%s bm25_raw=%s vec_rank=%s vec_raw=%s vec_pass=%s Q="%s" url="%s"',
                final_rank,
                str(keep),
                str(rid),
                score_str,
                bm25_contrib_str,
                vec_contrib_str,
                sources,
                bm25_rank_str,
                bm25_raw_str,
                vec_rank_str,
                vec_raw_str,
                str(vec_pass_threshold),
                q,
                url,
            )

        return {"rrf_ranked_all": organized}

    return _run
