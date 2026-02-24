import logging
from typing import Any, Dict, List

from app.core.config import get_env_int
from app.core.types import GraphState


logger = logging.getLogger(__name__)


def node_evidence_assess(candidate_source: str = "merged_candidates_all"):
    def _run(state: GraphState) -> GraphState:
        source_key = str(candidate_source or "merged_candidates_all")
        all_candidates: List[Dict[str, Any]] = state.get(source_key, []) or []
        top_n = max(1, get_env_int("EVIDENCE_TOPN", 10))
        candidates = list(all_candidates[:top_n])
        turn_count = int(state.get("turn_count", 0))

        if not candidates:
            level = "none"
            reason = "no_candidates"
        else:
            any_passed = any(bool(r.get("passed_any", False)) for r in candidates)
            any_multi = any(
                bool(r.get("has_multiple_sources", False)) for r in candidates
            )
            if any_passed or any_multi:
                level = "high"
                if any_passed and any_multi:
                    reason = "found_passed_and_multi"
                elif any_passed:
                    reason = "found_passed_any"
                else:
                    reason = "found_has_multiple_sources"
            else:
                level = "low"
                reason = "no_passed_or_multi"

        # Low/none evidence should be eligible for web check from the first turn.
        web_needed = level in {"none", "low"}

        logger.info(
            "[evidence-assess] source=%s topn=%d candidates=%d level=%s reason=%s turn_count=%d web_needed=%s",
            source_key,
            top_n,
            len(candidates),
            level,
            reason,
            turn_count,
            str(web_needed),
        )

        return {
            "local_evidence_level": level,
            "local_evidence_reason": reason,
            "web_needed": bool(web_needed),
        }

    return _run
