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
        # Merge by id -> compute RRF score -> sort ALL candidates.
        results_by_source = state.get("retrieval_results_by_source", {}) or {}

        rrf_k = max(1, get_env_int("RRF_K", 60))
        final_topk = max(1, get_env_int("FINAL_TOPK", 20))

        by_id: Dict[int, Dict[str, Any]] = {}
        for source, rows in results_by_source.items():
            for row in rows:
                rid = int(row["id"])
                it: MetaItem = row["item"]
                rank = row.get("rank", None)
                raw_score = row.get("raw_score", None)
                passed = row.get("passed", None)
                features = row.get("features", {}) or {}

                if rid not in by_id:
                    by_id[rid] = {
                        "id": rid,
                        "item": it,
                        "sources": [],
                        "source_details": {},
                        "source_contrib": {},
                    }

                if source not in by_id[rid]["sources"]:
                    by_id[rid]["sources"].append(source)

                safe_rank = int(rank) if isinstance(rank, int) and rank > 0 else None
                safe_raw = (
                    float(raw_score)
                    if isinstance(raw_score, (int, float))
                    else None
                )
                safe_passed = bool(passed) if isinstance(passed, bool) else None
                by_id[rid]["source_details"][source] = {
                    "rank": safe_rank,
                    "raw_score": safe_raw,
                    "passed": safe_passed,
                    "features": dict(features),
                }

                contrib = 0.0
                if isinstance(safe_rank, int) and safe_rank > 0:
                    contrib = 1.0 / float(rrf_k + safe_rank)
                by_id[rid]["source_contrib"][source] = float(contrib)

        organized: List[Dict[str, Any]] = []
        for _, row in by_id.items():
            contrib_map = row.get("source_contrib", {}) or {}
            detail_map = row.get("source_details", {}) or {}
            score = float(sum(float(v) for v in contrib_map.values()))
            passed_any = any(
                detail.get("passed") is True for detail in detail_map.values()
            )
            has_multiple_sources = len(row.get("sources", [])) >= 2

            organized.append(
                {
                    **row,
                    "score": score,
                    "passed_any": bool(passed_any),
                    "has_multiple_sources": bool(has_multiple_sources),
                }
            )

        organized.sort(key=lambda x: x["score"], reverse=True)

        for i, r in enumerate(organized, start=1):
            r["final_rank"] = int(i)
            r["keep"] = bool(i <= final_topk)

        active = state.get("active_retrievers", []) or []
        reason = shorten_text(state.get("retrieval_plan_reason", ""), 60)
        counts = state.get("retrieval_counts", {}) or {}

        search_query = shorten_text(state.get("search_query", ""), 120)

        logger.info(
            '[rrf-rank] active=%s reason=%s counts=%s rrf_k=%d final_topk=%d merged=%d search_query="%s"',
            ",".join([str(v) for v in active]),
            reason,
            str(counts),
            rrf_k,
            final_topk,
            len(organized),
            search_query,
        )

        for r in organized:
            final_rank = int(r.get("final_rank", 0))
            keep = bool(r.get("keep", False))
            rid = int(r.get("id", -1))
            score = float(r.get("score", 0.0))
            sources = ",".join([str(v) for v in r.get("sources", [])])
            passed_any = bool(r.get("passed_any", False))
            multi = bool(r.get("has_multiple_sources", False))
            contrib_text = ",".join(
                f"{k}:{float(v):.6f}"
                for k, v in (r.get("source_contrib", {}) or {}).items()
            )
            details = r.get("source_details", {}) or {}
            detail_text = ",".join(
                f"{source}(rank={detail.get('rank')} raw={detail.get('raw_score')} passed={detail.get('passed')})"
                for source, detail in details.items()
            )
            item = r.get("item", {}) or {}
            q = shorten_text(item.get("question", ""), 80)
            url = shorten_text(item.get("url", ""), 160)

            logger.info(
                '[rrf-rank] #%d keep=%s id=%d score=%.6f sources=%s passed_any=%s multi=%s contribs="%s" details="%s" Q="%s" url="%s"',
                final_rank,
                str(keep),
                rid,
                score,
                sources,
                str(passed_any),
                str(multi),
                contrib_text,
                detail_text,
                q,
                url,
            )

        return {"merged_candidates_all": organized}

    return _run
