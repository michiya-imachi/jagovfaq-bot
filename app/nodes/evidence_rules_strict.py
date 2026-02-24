import logging
from typing import Any, Dict, List

from app.core.store import IndexedStore
from app.core.types import GraphState


logger = logging.getLogger(__name__)
_VEC_SIM_THRESHOLD = 0.4
_THRESHOLD_LABEL = str(_VEC_SIM_THRESHOLD).rstrip("0").rstrip(".")
_REASON_GE = f"vecsim_max_ge_{_THRESHOLD_LABEL}"
_REASON_LT = f"vecsim_max_lt_{_THRESHOLD_LABEL}"


def node_evidence_rules_strict(
    store: IndexedStore,
    candidate_source: str = "retrieved",
):
    def _run(state: GraphState) -> GraphState:
        source_key = str(candidate_source or "retrieved")
        candidates: List[Dict[str, Any]] = state.get(source_key, []) or []
        candidate_ids: List[int] = []
        for row in candidates:
            if not isinstance(row, dict):
                continue
            try:
                candidate_ids.append(int(row["id"]))
            except (TypeError, ValueError, KeyError) as e:
                raise ValueError(f"evidence_candidate_id_invalid: {e}") from e

        max_sim: float | None = None
        sim_map: Dict[int, float] = {}
        if not candidate_ids:
            level = "none"
            reason = "no_candidates"
        else:
            query = str(
                state.get("search_query") or state.get("original_user_query") or ""
            ).strip()
            sim_map = store.query_doc_similarities(query=query, rec_ids=candidate_ids)
            max_sim = max(float(v) for v in sim_map.values())
            if max_sim >= _VEC_SIM_THRESHOLD:
                level = "high"
                reason = _REASON_GE
            else:
                level = "low"
                reason = _REASON_LT

        enriched_candidates: List[Dict[str, Any]] = []
        for row in candidates:
            if not isinstance(row, dict):
                continue
            rid = int(row["id"])
            raw_sources = row.get("sources", []) or []
            if isinstance(raw_sources, list):
                source = ",".join(str(v) for v in raw_sources)
            else:
                source = str(raw_sources)
            raw_rrf = row.get("rrf_score")
            if isinstance(raw_rrf, (int, float)):
                rrf_score = float(raw_rrf)
            else:
                contrib_map = row.get("source_contrib", {}) or {}
                rrf_score = float(sum(float(v) for v in contrib_map.values()))
            sim_score = (
                float(sim_map[rid]) if rid in sim_map else None
            )
            logger.info(
                "id=%d source=%s rrf_score=%.6f sim_score=%s",
                rid,
                source,
                rrf_score,
                f"{sim_score:.6f}" if isinstance(sim_score, float) else "n/a",
            )
            enriched = {
                **row,
                "source": source,
                "rrf_score": rrf_score,
            }
            if isinstance(sim_score, float):
                enriched["sim_score"] = sim_score
            enriched_candidates.append(enriched)

        logger.info(
            "source=%s candidates=%d threshold=%.2f max_sim=%s level=%s reason=%s",
            source_key,
            len(candidates),
            _VEC_SIM_THRESHOLD,
            f"{max_sim:.6f}" if isinstance(max_sim, float) else "n/a",
            level,
            reason,
        )

        return {
            source_key: enriched_candidates,
            "evidence_rules_level": str(level),
            "evidence_rules_reason": str(reason),
        }

    return _run
