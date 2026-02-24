import logging

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


def make_response_router():
    def _run(state: GraphState) -> GraphState:
        turn_count = int(state.get("turn_count", 0))
        max_turns = int(state.get("max_turns", 2))
        web_search_attempted = bool(state.get("web_search_attempted", False))
        hitl_permission_web_asked = bool(
            state.get("hitl_permission_web_asked", False)
        )
        web_search_declined = bool(state.get("web_search_declined", False))
        can_ask_web_permission = (
            (not web_search_attempted)
            and (not hitl_permission_web_asked)
            and (not web_search_declined)
        )

        raw_hint = str(state.get("evidence_action_hint", "") or "").strip().lower()
        if raw_hint in {"answer", "followup", "web"}:
            hint = raw_hint
            hint_valid = True
        else:
            hint = "followup"
            hint_valid = False

        if hint == "answer":
            next_node = "answer"
            next_node_reason = "hint_answer"
            need_followup = False
        elif hint == "web":
            if can_ask_web_permission:
                next_node = "hitl_permission_web"
                next_node_reason = "hint_web_need_permission"
            else:
                next_node = "answer"
                next_node_reason = "hint_web_cannot_ask_answer"
            need_followup = False
        else:
            if turn_count < max_turns:
                next_node = "followup_question"
                next_node_reason = "hint_followup" if hint_valid else "hint_invalid_followup"
                need_followup = True
            else:
                next_node = "fallback"
                next_node_reason = "hint_followup_max_turns"
                need_followup = True

        query = shorten_text(state.get("search_query", ""), 120)
        logger.debug(
            "[response-router] hint=%s hint_valid=%s turn=%d/%d need_followup=%s asked=%s attempted=%s declined=%s",
            hint,
            str(hint_valid),
            turn_count,
            max_turns,
            str(need_followup),
            str(hitl_permission_web_asked),
            str(web_search_attempted),
            str(web_search_declined),
        )
        logger.debug(
            '[response-router] next_node=%s reason=%s search_query="%s"',
            next_node,
            next_node_reason,
            query,
        )
        logger.info(
            "[response-router] hint=%s turn_count=%d hitl_permission_web_asked=%s web_search_attempted=%s web_search_declined=%s",
            hint,
            turn_count,
            str(hitl_permission_web_asked),
            str(web_search_attempted),
            str(web_search_declined),
        )

        return {
            "need_followup": bool(need_followup),
            "next_node": str(next_node),
            "next_node_reason": str(next_node_reason),
        }

    return _run


def node_response_router(state: GraphState) -> GraphState:
    return make_response_router()(state)
