import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_env_float, get_env_int, get_env_str
from app.core.types import GraphState
from app.prompts.loader import PromptLoader


logger = logging.getLogger(__name__)

_ALLOWED_LEVELS = {"high", "low", "none"}
_ALLOWED_ACTIONS = {"answer", "followup", "web"}
_ANSWER_MAX_CHARS = 800


def _shorten_text(text: Any, max_len: int) -> str:
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        rows: List[str] = []
        for part in content:
            if isinstance(part, str):
                rows.append(part)
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    rows.append(text)
                    continue
                value = part.get("value")
                if isinstance(value, str):
                    rows.append(value)
                    continue
        return "\n".join(rows).strip()
    return str(content or "").strip()


def _resolve_judge_llm(base_llm: Any, model: str, timeout_s: float) -> Tuple[Any, str]:
    base_model = str(getattr(base_llm, "model_name", "") or "").strip()
    base_timeout = getattr(base_llm, "request_timeout", None)
    if not base_model:
        # Tests can inject a fake llm object that only supports invoke().
        return base_llm, "injected"

    timeout_match = False
    if isinstance(base_timeout, (int, float)):
        timeout_match = float(base_timeout) == float(timeout_s)
    if base_model == model and timeout_match:
        return base_llm, "base"

    return ChatOpenAI(model=model, temperature=0, timeout=timeout_s), "runtime_new"


def _format_candidates_for_prompt(
    all_candidates: List[Dict[str, Any]],
    top_n: int,
) -> str:
    rows: List[str] = []
    for index, row in enumerate(all_candidates[:top_n], start=1):
        if not isinstance(row, dict):
            continue

        item = row.get("item", {}) or {}
        if not isinstance(item, dict):
            item = {}

        final_rank = row.get("final_rank")
        if not isinstance(final_rank, int) or final_rank <= 0:
            final_rank = index

        score = row.get("score", 0.0)
        if isinstance(score, (int, float)):
            score_text = f"{float(score):.6f}"
        else:
            score_text = "0.000000"

        raw_sources = row.get("sources", []) or []
        if isinstance(raw_sources, list):
            sources = ",".join(str(v) for v in raw_sources)
        else:
            sources = str(raw_sources)

        passed_any = bool(row.get("passed_any", False))
        has_multiple_sources = bool(row.get("has_multiple_sources", False))

        question = _shorten_text(item.get("question", ""), 240)
        answer = _shorten_text(item.get("answer", ""), _ANSWER_MAX_CHARS)
        url = _shorten_text(item.get("url", ""), 240)

        rows.append(
            "\n".join(
                [
                    f"[candidate_{index}]",
                    f"final_rank: {final_rank}",
                    f"score: {score_text}",
                    f"sources: {sources}",
                    f"passed_any: {str(passed_any).lower()}",
                    f"has_multiple_sources: {str(has_multiple_sources).lower()}",
                    f"question: {question}",
                    f"answer: {answer}",
                    f"url: {url}",
                ]
            )
        )

    if not rows:
        return "(no candidates)"
    return "\n\n".join(rows)


def _parse_result_json(raw_text: str) -> Tuple[str, str, str]:
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("root_must_be_object")

    level = str(parsed.get("level", "") or "").strip().lower()
    action = str(parsed.get("action", "") or "").strip().lower()
    reason = str(parsed.get("reason", "") or "").strip()

    if level not in _ALLOWED_LEVELS:
        raise ValueError(f"invalid_level:{level}")
    if action not in _ALLOWED_ACTIONS:
        raise ValueError(f"invalid_action:{action}")
    if not reason:
        raise ValueError("empty_reason")

    return level, action, reason


def node_evidence_llm_judge(
    llm: Any,
    prompts: PromptLoader,
    candidate_source: str = "merged_candidates_all",
):
    def _run(state: GraphState) -> GraphState:
        if not bool(state.get("run_evidence_llm", False)):
            logger.info("[evidence-llm] skipped run_evidence_llm=false")
            return {}

        source_key = str(candidate_source or "merged_candidates_all")
        all_candidates: List[Dict[str, Any]] = state.get(source_key, []) or []
        top_n = max(1, get_env_int("EVIDENCE_LLM_TOPN", 5))
        timeout_s = max(1.0, get_env_float("EVIDENCE_LLM_TIMEOUT_S", 25.0))
        model = get_env_str(
            "EVIDENCE_LLM_MODEL",
            get_env_str("OPENAI_MODEL", "gpt-5-mini"),
        )

        search_query = str(
            state.get("search_query") or state.get("original_user_query") or ""
        ).strip()
        candidates_text = _format_candidates_for_prompt(all_candidates, top_n)
        prompt_pair = prompts.render_pair(
            "evidence_judge",
            search_query=search_query,
            candidates_text=candidates_text,
        )
        messages = [
            SystemMessage(content=prompt_pair["system"]),
            HumanMessage(content=prompt_pair["user"]),
        ]

        try:
            judge_llm, llm_origin = _resolve_judge_llm(llm, model=model, timeout_s=timeout_s)
            msg = judge_llm.invoke(messages)
            raw_text = _content_to_text(getattr(msg, "content", ""))
            level, action, reason = _parse_result_json(raw_text)

            logger.info(
                "[evidence-llm] source=%s topn=%d candidates=%d model=%s timeout_s=%.1f llm_origin=%s success=true level=%s action=%s reason=%s",
                source_key,
                top_n,
                len(all_candidates[:top_n]),
                model,
                timeout_s,
                llm_origin,
                level,
                action,
                _shorten_text(reason, 80),
            )
            return {
                "evidence_llm_level": str(level),
                "evidence_llm_action": str(action),
                "evidence_llm_reason": str(reason),
                "evidence_llm_error": "",
            }
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.warning(
                "[evidence-llm] source=%s topn=%d candidates=%d model=%s timeout_s=%.1f success=false error=%s",
                source_key,
                top_n,
                len(all_candidates[:top_n]),
                model,
                timeout_s,
                _shorten_text(err, 240),
            )
            return {
                "evidence_llm_level": "",
                "evidence_llm_action": "",
                "evidence_llm_reason": "",
                "evidence_llm_error": err,
            }

    return _run
