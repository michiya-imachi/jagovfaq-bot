import logging

from app.core.types import GraphState


logger = logging.getLogger(__name__)

_ALLOWED_LEVELS = {"high", "low", "none"}
_ALLOWED_ACTIONS = {"answer", "followup", "web"}


def node_evidence_finalize():
    def _run(state: GraphState) -> GraphState:
        rules_level = str(state.get("evidence_rules_level", "") or "").strip().lower()
        rules_reason = str(state.get("evidence_rules_reason", "") or "").strip()
        if rules_level not in _ALLOWED_LEVELS:
            rules_level = "low"

        run_evidence_llm = bool(state.get("run_evidence_llm", False))
        llm_level = str(state.get("evidence_llm_level", "") or "").strip().lower()
        llm_action = str(state.get("evidence_llm_action", "") or "").strip().lower()
        llm_reason = str(state.get("evidence_llm_reason", "") or "").strip()
        llm_error = str(state.get("evidence_llm_error", "") or "").strip()

        if not run_evidence_llm:
            local_evidence_level = "high"
            local_evidence_reason = f"rules_strict_high:{rules_reason}"
            evidence_action_hint = "answer"
            web_needed = False
            branch = "rules_high"
        elif (
            not llm_error
            and llm_level in _ALLOWED_LEVELS
            and llm_action in _ALLOWED_ACTIONS
            and llm_reason
        ):
            local_evidence_level = llm_level
            local_evidence_reason = f"llm:{llm_reason}"
            evidence_action_hint = llm_action
            web_needed = llm_action == "web"
            branch = "llm_success"
        else:
            local_evidence_level = rules_level
            local_evidence_reason = f"llm_failed_fallback_rules:{llm_error or 'unknown_error'}"
            evidence_action_hint = "followup"
            web_needed = local_evidence_level in {"low", "none"}
            branch = "llm_failed_fallback"

        logger.info(
            "[evidence-finalize] branch=%s run_evidence_llm=%s local_level=%s action_hint=%s web_needed=%s",
            branch,
            str(run_evidence_llm),
            local_evidence_level,
            evidence_action_hint,
            str(web_needed),
        )

        return {
            "local_evidence_level": str(local_evidence_level),
            "local_evidence_reason": str(local_evidence_reason),
            "evidence_action_hint": str(evidence_action_hint),
            "web_needed": bool(web_needed),
        }

    return _run
