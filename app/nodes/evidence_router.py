import logging

from app.core.types import GraphState


logger = logging.getLogger(__name__)


def node_evidence_router():
    def _run(state: GraphState) -> GraphState:
        level = str(state.get("evidence_rules_level", "") or "").strip().lower()
        if level == "high":
            run_evidence_llm = False
            reason = "rules_high_skip_llm"
        else:
            run_evidence_llm = True
            reason = "rules_not_high_run_llm"

        logger.info(
            "[evidence-router] rules_level=%s run_evidence_llm=%s reason=%s",
            level,
            str(run_evidence_llm),
            reason,
        )

        return {
            "run_evidence_llm": bool(run_evidence_llm),
            "evidence_route_reason": str(reason),
        }

    return _run
