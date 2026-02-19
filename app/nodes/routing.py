import logging
from typing import Any, Dict, List

from app.core.types import GraphState


logger = logging.getLogger(__name__)


def shorten_text(text: str, max_len: int) -> str:
    # Truncate for log readability.
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "..."


def _infer_level_from_candidates(
    state: GraphState, top_n: int, candidate_source: str
) -> str:
    source_key = str(candidate_source or "merged_candidates_all")
    source_items = state.get(source_key, []) or []
    candidates: List[Dict[str, Any]] = source_items[: max(1, int(top_n))]
    if not candidates:
        return "none"

    any_passed = any(bool(r.get("passed_any", False)) for r in candidates)
    any_multi = any(bool(r.get("has_multiple_sources", False)) for r in candidates)
    return "high" if (any_passed or any_multi) else "low"


def make_decide_next_action(top_n: int = 1, candidate_source: str = "merged_candidates_all"):
    def _run(state: GraphState) -> GraphState:
        turn_count = int(state.get("turn_count", 0))
        max_turns = int(state.get("max_turns", 2))
        web_search_attempted = bool(state.get("web_search_attempted", False))
        web_permission_asked = bool(state.get("web_permission_asked", False))
        can_ask_web_permission = (not web_search_attempted) and (not web_permission_asked)

        raw_level = str(state.get("local_evidence_level", "") or "").strip().lower()
        if raw_level in {"high", "low", "none"}:
            level = raw_level
        else:
            level = _infer_level_from_candidates(state, top_n, candidate_source)

        web_needed_state = state.get("web_needed", None)
        if isinstance(web_needed_state, bool):
            web_needed = bool(web_needed_state)
        else:
            web_needed = level in {"none", "low"} and turn_count >= 1

        forced_multi_turn_web_check = turn_count >= 2 and can_ask_web_permission

        if level == "high":
            next_node = "answer"
            next_node_reason = "high_evidence_local_answer"
            need = False
        elif forced_multi_turn_web_check:
            next_node = "web_permission"
            next_node_reason = "force_web_check_multi_turn"
            need = False
        elif web_needed and can_ask_web_permission:
            next_node = "web_permission"
            next_node_reason = "need_web_permission"
            need = False
        elif turn_count >= max_turns:
            next_node = "fallback"
            next_node_reason = "max_turns_reached"
            need = True
        else:
            next_node = "followup_question"
            next_node_reason = "need_followup"
            need = True

        query = shorten_text(state.get("search_query", ""), 120)
        logger.debug(
            "[decide] level=%s web_needed=%s turn=%d/%d need_followup=%s asked=%s attempted=%s forced=%s",
            level,
            str(web_needed),
            turn_count,
            max_turns,
            str(need),
            str(web_permission_asked),
            str(web_search_attempted),
            str(forced_multi_turn_web_check),
        )
        logger.debug(
            '[decide] next_node=%s reason=%s search_query="%s"',
            next_node,
            next_node_reason,
            query,
        )
        logger.info(
            "[decide] turn_count=%d web_permission_asked=%s web_search_attempted=%s forced_multi_turn_web_check=%s",
            turn_count,
            str(web_permission_asked),
            str(web_search_attempted),
            str(forced_multi_turn_web_check),
        )

        return {
            "need_followup": bool(need),
            "next_node": str(next_node),
            "next_node_reason": str(next_node_reason),
        }

    return _run


def node_decide_next_action(state: GraphState) -> GraphState:
    # Keep backward compatibility for direct imports/tests.
    return make_decide_next_action(top_n=1, candidate_source="merged_candidates_all")(state)
