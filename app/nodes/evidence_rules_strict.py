import logging
from typing import Any, Dict, List

from app.core.config import get_env_int
from app.core.types import GraphState


logger = logging.getLogger(__name__)


def node_evidence_rules_strict(candidate_source: str = "merged_candidates_all"):
    def _run(state: GraphState) -> GraphState:
        source_key = str(candidate_source or "merged_candidates_all")
        all_candidates: List[Dict[str, Any]] = state.get(source_key, []) or []
        top_n = max(1, get_env_int("EVIDENCE_TOPN", 10))
        candidates = list(all_candidates[:top_n])

        if not candidates:
            level = "none"
            reason = "no_candidates"
        else:
            top1 = candidates[0] if isinstance(candidates[0], dict) else {}
            passed_any = bool(top1.get("passed_any", False))
            has_multiple_sources = bool(top1.get("has_multiple_sources", False))
            if passed_any and has_multiple_sources:
                level = "high"
                reason = "top1_passed_and_multi"
            else:
                level = "low"
                reason = "not_strong_enough"

        logger.info(
            "[evidence-rules-strict] source=%s topn=%d candidates=%d level=%s reason=%s",
            source_key,
            top_n,
            len(candidates),
            level,
            reason,
        )

        return {
            "evidence_rules_level": str(level),
            "evidence_rules_reason": str(reason),
        }

    return _run
