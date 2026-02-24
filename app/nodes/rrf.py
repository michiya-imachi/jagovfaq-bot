import logging
from typing import Any, Dict, List, Optional

from app.core.config import get_env_int
from app.core.types import GraphState


logger = logging.getLogger(__name__)


def shorten_text(text: Any, max_len: int) -> str:
    # Truncate for log readability.
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "..."


def _safe_rank(value: Any) -> Optional[int]:
    if isinstance(value, int) and value > 0:
        return int(value)
    return None


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def node_rrf_rank():
    def _run(state: GraphState) -> GraphState:
        # Merge by id -> compute RRF score -> sort all candidates.
        bm25 = state.get("bm25_retrieved", []) or []
        vec = state.get("vec_retrieved", []) or []

        rrf_k = max(1, get_env_int("RRF_K", 60))
        final_topk = max(1, get_env_int("FINAL_TOPK", 20))

        by_id: Dict[int, Dict[str, Any]] = {}

        def _add_source(
            rid: int,
            source: str,
            rank_value: Any,
            raw_value: Any,
            passed_value: Any,
            features: Dict[str, Any],
        ) -> None:
            if rid not in by_id:
                by_id[rid] = {
                    "id": rid,
                    "sources": [],
                    "source_details": {},
                    "source_contrib": {},
                }

            row = by_id[rid]
            if source not in row["sources"]:
                row["sources"].append(source)

            safe_rank = _safe_rank(rank_value)
            safe_raw = _safe_float(raw_value)
            safe_passed = (
                bool(passed_value) if isinstance(passed_value, bool) else None
            )

            row["source_details"][source] = {
                "rank": safe_rank,
                "raw_score": safe_raw,
                "passed": safe_passed,
                "features": dict(features),
            }

            contrib = 0.0
            if safe_rank is not None:
                contrib = 1.0 / float(rrf_k + safe_rank)
            row["source_contrib"][source] = float(contrib)

        for row in bm25:
            try:
                rid = int(row["id"])
            except (TypeError, ValueError, KeyError):
                continue

            raw = _safe_float(row.get("bm25_raw"))
            _add_source(
                rid=rid,
                source="bm25",
                rank_value=row.get("bm25_rank"),
                raw_value=raw,
                passed_value=None,
                features={"bm25_raw": raw},
            )

        for row in vec:
            try:
                rid = int(row["id"])
            except (TypeError, ValueError, KeyError):
                continue

            raw = _safe_float(row.get("vec_raw"))
            passed = bool(row.get("vec_pass_threshold", False))
            _add_source(
                rid=rid,
                source="vec",
                rank_value=row.get("vec_rank"),
                raw_value=raw,
                passed_value=passed,
                features={
                    "vec_raw": raw,
                    "vec_pass_threshold": passed,
                },
            )

        organized: List[Dict[str, Any]] = []
        for _, merged in by_id.items():
            contrib_map = merged.get("source_contrib", {}) or {}
            detail_map = merged.get("source_details", {}) or {}
            rrf_score = float(sum(float(v) for v in contrib_map.values()))
            passed_any = any(
                detail.get("passed") is True for detail in detail_map.values()
            )
            has_multiple_sources = len(merged.get("sources", [])) >= 2

            organized.append(
                {
                    **merged,
                    "rrf_score": rrf_score,
                    "passed_any": bool(passed_any),
                    "has_multiple_sources": bool(has_multiple_sources),
                }
            )

        organized.sort(key=lambda x: x["rrf_score"], reverse=True)

        for i, row in enumerate(organized, start=1):
            row["final_rank"] = int(i)
            row["keep"] = bool(i <= final_topk)

        active = state.get("active_retrievers", []) or []
        reason = shorten_text(state.get("retrieval_plan_reason", ""), 60)
        bm25_count = int(state.get("bm25_count", len(bm25)))
        vec_count = int(state.get("vec_count", len(vec)))
        vec_pass_count = int(state.get("vec_pass_count", 0))
        search_query = shorten_text(state.get("search_query", ""), 120)

        logger.info(
            '[rrf-rank] active=%s reason=%s bm25_count=%d vec_count=%d vec_pass=%d rrf_k=%d final_topk=%d merged=%d search_query="%s"',
            ",".join([str(v) for v in active]),
            reason,
            bm25_count,
            vec_count,
            vec_pass_count,
            rrf_k,
            final_topk,
            len(organized),
            search_query,
        )

        for row in organized:
            final_rank = int(row.get("final_rank", 0))
            keep = bool(row.get("keep", False))
            rid = int(row.get("id", -1))
            rrf_score = float(row.get("rrf_score", 0.0))
            sources = ",".join([str(v) for v in row.get("sources", [])])

            logger.info(
                '[rrf-rank] #%d keep=%s id=%d source=%s rrf_score=%.6f sim_score=n/a',
                final_rank,
                str(keep),
                rid,
                sources,
                rrf_score,
            )

        return {"merged_candidates_all": organized}

    return _run
