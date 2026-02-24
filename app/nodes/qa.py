import logging
import sys
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.types import GraphState, MetaItem
from app.prompts.loader import PromptLoader


logger = logging.getLogger(__name__)


def _shorten_text(text: Any, max_len: int = 120) -> str:
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "..."


def _dedupe_citations(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        url = str(item.get("url", "") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "url": url,
                "question": str(item.get("question", "") or "").strip(),
            }
        )
    return out


def _build_local_sources(retrieved: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    sources_text: List[str] = []
    citations: List[Dict[str, str]] = []

    for r in retrieved:
        it: MetaItem = r["item"]
        score = float(r.get("score", 0.0))
        sources = ",".join([str(v) for v in r.get("sources", [])])
        details = r.get("source_details", {}) or {}
        detail_text = ", ".join(
            f"{source}(rank={detail.get('rank')} raw={detail.get('raw_score')} passed={detail.get('passed')})"
            for source, detail in details.items()
        )

        sources_text.append(
            f"[rrf_score={score:.4f} sources={sources} details={detail_text}]\n"
            f"Q: {it['question']}\nA: {it['answer']}\nURL: {it['url']}\n"
        )
        citations.append({"url": it["url"], "question": it["question"]})

    return "\n".join(sources_text), citations


def _build_web_sources(web_results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    rows: List[str] = []
    citations: List[Dict[str, str]] = []

    for row in web_results:
        url = str(row.get("url", "") or "").strip()
        if not url:
            continue
        title = str(row.get("title", "") or "").strip()
        snippet = str(row.get("snippet", "") or "").strip()

        rows.append(
            f"[web]\n"
            f"Title: {title}\n"
            f"Snippet: {snippet}\n"
            f"URL: {url}\n"
        )
        citations.append({"url": url, "question": title})

    return "\n".join(rows), citations


def _build_warning_lines(state: GraphState, web_results: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    level = str(state.get("local_evidence_level", "") or "").strip().lower()
    if level in {"none", "low"}:
        lines.append("注意: ローカルFAQ候補が弱いため、確度は高くありません。")

    if bool(state.get("web_search_declined", False)):
        lines.append("注意: Web検索は実行していません（ユーザーの選択）。")

    web_error = str(state.get("web_search_error", "") or "").strip()
    if web_error:
        lines.append("注意: Web検索が失敗したため、ローカル情報のみで回答します。")

    attempted = bool(state.get("web_search_attempted", False))
    if attempted and not web_results and not web_error:
        lines.append("注意: Web検索でも確証が得られませんでした。")

    return lines


def node_followup_question(llm: Any, prompts: PromptLoader):
    def _run(state: GraphState) -> GraphState:
        retrieved = state.get("retrieved", []) or []
        user_query = str(
            state.get("search_query") or state.get("original_user_query", "")
        ).strip()

        snippets = []
        for r in retrieved[:3]:
            it: MetaItem = r["item"]
            sources = ",".join([str(v) for v in r.get("sources", [])])
            score = float(r.get("score", 0.0))
            details = r.get("source_details", {}) or {}
            detail_text = ", ".join(
                f"{source}(rank={detail.get('rank')} raw={detail.get('raw_score')} passed={detail.get('passed')})"
                for source, detail in details.items()
            )

            snippets.append(
                f"- sources: {sources} score: {score:.4f} details: {detail_text}\n"
                f"  Q: {it['question']}\n"
                f"  A: {it['answer'][:120]}...\n"
                f"  URL: {it['url']}"
            )
        context = "\n".join(snippets) if snippets else "(no candidates)"

        prompt_pair = prompts.render_pair(
            "ask_clarification",
            user_query=user_query,
            context=context,
        )
        messages = [
            SystemMessage(content=prompt_pair["system"]),
            HumanMessage(content=prompt_pair["user"]),
        ]

        msg = llm.invoke(messages)
        q = str(msg.content).strip()
        return {
            "followup_question": q,
        }

    return _run


def node_generate_answer_stream(llm: Any, prompts: PromptLoader):
    def _run(state: GraphState) -> GraphState:
        user_query = str(
            state.get("search_query") or state.get("original_user_query", "")
        ).strip()
        retrieved = (state.get("retrieved", []) or [])[:5]
        web_results = state.get("web_results", []) or []

        local_sources_text, local_citations = _build_local_sources(retrieved)
        web_sources_text, web_citations = _build_web_sources(web_results)
        citations = _dedupe_citations(local_citations + web_citations)
        warning_lines = _build_warning_lines(state, web_results)
        warning_notes = "\n".join(warning_lines)

        prompt_name = "generate_answer_with_web" if web_results else "generate_answer"
        if prompt_name == "generate_answer_with_web":
            prompt_pair = prompts.render_pair(
                prompt_name,
                user_query=user_query,
                local_sources_text=local_sources_text,
                web_sources_text=web_sources_text,
                warning_notes=warning_notes,
            )
        else:
            prompt_pair = prompts.render_pair(
                prompt_name,
                user_query=user_query,
                sources_text=local_sources_text,
                warning_notes=warning_notes,
            )

        messages = [
            SystemMessage(content=prompt_pair["system"]),
            HumanMessage(content=prompt_pair["user"]),
        ]

        sys.stdout.flush()
        if warning_lines:
            for line in warning_lines:
                sys.stdout.write(line + "\n")
            sys.stdout.write("\n")
            sys.stdout.flush()

        full: List[str] = []
        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", "")
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                full.append(text)
        sys.stdout.write("\n\n")
        sys.stdout.flush()

        generated = "".join(full).strip()
        if warning_lines:
            answer = ("\n".join(warning_lines) + "\n\n" + generated).strip()
        else:
            answer = generated

        logger.info(
            "[answer] local_evidence_level=%s web_results=%d declined=%s web_error=%s citations=%d",
            str(state.get("local_evidence_level", "")),
            len(web_results),
            str(bool(state.get("web_search_declined", False))),
            str(bool(state.get("web_search_error", ""))),
            len(citations),
        )

        return {
            "answer": answer,
            "citations": citations,
        }

    return _run


def node_fallback(state: GraphState) -> GraphState:
    reason = str(state.get("next_node_reason", "")).strip()
    note = ""
    if reason in {"max_turns_reached", "hint_followup_max_turns"}:
        note = "\n(注記) 追加確認の上限回数に達したため、ここで終了しました。"

    msg = (
        "すみません。該当しそうなFAQが見つかりませんでした。\n"
        f"{note}\n\n"
    )
    print(msg + "\n")
    return {"answer": msg}
