import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.types import GraphState
from app.prompts.loader import PromptLoader


logger = logging.getLogger(__name__)


def _shorten_text(text: Any, max_len: int = 160) -> str:
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "..."


def node_standalone_question(llm: Any, prompts: PromptLoader):
    def _run(state: GraphState) -> GraphState:
        original = str(state.get("original_user_query", "")).strip()
        current = str(state.get("search_query", "")).strip() or original

        followup_q = str(state.get("followup_question", "")).strip()
        followup_a = str(state.get("followup_answer", "")).strip()
        has_followup_answer = bool(followup_a)
        reason = "with_followup" if has_followup_answer else "first_pass"
        if has_followup_answer and not followup_q:
            reason = "with_followup_missing_question"

        prompt_pair = prompts.render_pair(
            "standalone_question",
            original_user_query=original,
            current_search_query=current,
            followup_question=followup_q,
            followup_answer=followup_a,
        )
        messages = [
            SystemMessage(content=prompt_pair["system"]),
            HumanMessage(content=prompt_pair["user"]),
        ]

        msg = llm.invoke(messages)
        rewritten = str(getattr(msg, "content", "")).strip()
        fallback_used = not bool(rewritten)
        new_q = rewritten or current

        if has_followup_answer:
            turn_count = int(state.get("turn_count", 0)) + 1
            logger.info(
                '[standalone-question] rewrite=true reason=%s fallback_used=%s turn_count=%d before="%s" after="%s"',
                reason,
                str(fallback_used).lower(),
                turn_count,
                _shorten_text(current),
                _shorten_text(new_q),
            )
            return {
                "search_query": new_q,
                "turn_count": turn_count,
                "followup_question": "",
                "followup_answer": "",
            }

        logger.info(
            '[standalone-question] rewrite=true reason=%s fallback_used=%s before="%s" after="%s"',
            reason,
            str(fallback_used).lower(),
            _shorten_text(current),
            _shorten_text(new_q),
        )
        return {"search_query": new_q}

    return _run
