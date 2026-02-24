import logging
import sys
import time
from typing import Any, Dict, List

from langgraph.types import interrupt

from app.core.config import get_env_float, get_env_int, get_env_str
from app.core.types import GraphState


logger = logging.getLogger(__name__)

_YES_WORDS = {"はい", "yes", "y", "ok", "okay", "true", "1"}
_NO_WORDS = {"いいえ", "no", "n", "false", "0"}


def _shorten_text(text: Any, max_len: int) -> str:
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "..."


def _normalize_yes_no(answer: Any) -> bool:
    raw = str(answer or "").strip().lower()
    if raw in _YES_WORDS:
        return True
    if raw in _NO_WORDS:
        return False
    return False


def _iter_output_items(response: Any) -> List[Any]:
    if response is None:
        return []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        return output
    if output is None:
        return []
    return list(output)


def _collect_sources_from_response(response: Any) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    seen_urls = set()
    titles_by_url: Dict[str, str] = {}

    for item in _iter_output_items(response):
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for part in content:
            if getattr(part, "type", None) != "output_text":
                continue
            annotations = getattr(part, "annotations", None) or []
            for ann in annotations:
                if getattr(ann, "type", None) != "url_citation":
                    continue
                url = str(getattr(ann, "url", "") or "").strip()
                title = str(getattr(ann, "title", "") or "").strip()
                if url and title and url not in titles_by_url:
                    titles_by_url[url] = title

    for item in _iter_output_items(response):
        item_type = getattr(item, "type", None)
        if item_type != "web_search_call":
            continue

        action = getattr(item, "action", None)
        action_type = getattr(action, "type", None)
        if action_type != "search":
            continue

        action_sources = getattr(action, "sources", None) or []
        for source in action_sources:
            url = str(getattr(source, "url", "") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                {"title": titles_by_url.get(url, ""), "url": url, "snippet": ""}
            )

    return sources


def node_web_permission():
    def _run(state: GraphState) -> GraphState:
        question = (
            "ローカルFAQだけでは根拠が不足しました。"
            "Web検索して確認してもよろしいですか？（はい/いいえ）"
        )
        payload = {"type": "HITL_PERMISSION_WEB", "question": question}
        raw_answer = interrupt(payload)
        answer_text = str(raw_answer or "").strip()
        allowed = _normalize_yes_no(answer_text)
        declined = not allowed

        logger.info(
            '[web-permission] question="%s" answer="%s" allowed=%s declined=%s',
            _shorten_text(question, 120),
            _shorten_text(answer_text, 80),
            str(allowed),
            str(declined),
        )

        return {
            "hitl_permission_web_question": question,
            "hitl_permission_web_answer": answer_text,
            "hitl_permission_web_asked": True,
            "web_search_allowed": bool(allowed),
            "web_search_declined": bool(declined),
        }

    return _run


def node_web_search(openai_client: Any):
    def _run(state: GraphState) -> GraphState:
        allowed = bool(state.get("web_search_allowed", False))
        attempted = bool(state.get("web_search_attempted", False))
        if not allowed or attempted:
            logger.info(
                "[web-search] skipped allowed=%s attempted=%s",
                str(allowed),
                str(attempted),
            )
            return {}

        web_query = str(
            state.get("search_query") or state.get("original_user_query") or ""
        ).strip()
        if not web_query:
            return {
                "web_query": "",
                "web_results": [],
                "web_search_error": "empty_web_query",
                "web_search_attempted": True,
            }

        model = get_env_str(
            "OPENAI_WEB_SEARCH_MODEL",
            get_env_str("OPENAI_MODEL", "gpt-5-mini"),
        )
        timeout_s = max(1.0, get_env_float("WEB_SEARCH_TIMEOUT_S", 45.0))
        max_retries = max(0, min(5, get_env_int("WEB_SEARCH_MAX_RETRIES", 1)))
        top_k = max(1, get_env_int("WEB_TOPK", 5))
        snippet_maxlen = max(1, get_env_int("WEB_SNIPPET_MAXLEN", 240))

        web_client = openai_client
        with_options = getattr(openai_client, "with_options", None)
        if callable(with_options):
            try:
                web_client = with_options(max_retries=max_retries)
            except Exception:
                web_client = openai_client

        sys.stdout.write("Web検索中...\n")
        sys.stdout.flush()
        started = time.perf_counter()
        try:
            response = web_client.responses.create(
                model=model,
                input=web_query,
                tools=[{"type": "web_search"}],
                tool_choice="required",
                include=["web_search_call.action.sources"],
                timeout=timeout_s,
            )
            rows = _collect_sources_from_response(response)
            web_results = []
            for row in rows[:top_k]:
                snippet = str(row.get("snippet", "") or "")
                web_results.append(
                    {
                        "title": str(row.get("title", "") or ""),
                        "url": str(row.get("url", "") or ""),
                        "snippet": snippet[:snippet_maxlen],
                    }
                )

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                '[web-search] model=%s query="%s" result_count=%d timeout_s=%.1f max_retries=%d elapsed_ms=%d attempted=true error=false',
                model,
                _shorten_text(web_query, 120),
                len(web_results),
                timeout_s,
                max_retries,
                elapsed_ms,
            )
            return {
                "web_query": web_query,
                "web_results": web_results,
                "web_search_error": "",
                "web_search_attempted": True,
            }
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(
                '[web-search] model=%s query="%s" result_count=0 timeout_s=%.1f max_retries=%d elapsed_ms=%d attempted=true error=true detail="%s"',
                model,
                _shorten_text(web_query, 120),
                timeout_s,
                max_retries,
                elapsed_ms,
                _shorten_text(err, 240),
            )
            return {
                "web_query": web_query,
                "web_results": [],
                "web_search_error": err,
                "web_search_attempted": True,
            }

    return _run
