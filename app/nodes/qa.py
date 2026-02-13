import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.types import GraphState, MetaItem
from app.prompts.loader import PromptLoader


def node_followup_question(llm: Any, prompts: PromptLoader):
    def _run(state: GraphState) -> GraphState:
        retrieved = state.get("retrieved", []) or []
        user_query = str(
            state.get("search_query") or state.get("original_user_query", "")
        ).strip()

        snippets = []
        for r in retrieved[:3]:
            it: MetaItem = r["item"]
            sources = ",".join(r.get("sources", []))
            bm25_rank = r.get("bm25_rank", None)
            vec_rank = r.get("vec_rank", None)
            vec_raw = r.get("vec_raw", None)

            snippets.append(
                f"- sources: {sources} bm25_rank: {bm25_rank} vec_rank: {vec_rank} vec_raw: {vec_raw}\n"
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

        sources_text = []
        citations: List[Dict[str, str]] = []
        for r in retrieved:
            it: MetaItem = r["item"]
            score = float(r.get("score", 0.0))
            sources = ",".join(r.get("sources", []))
            bm25_rank = r.get("bm25_rank", None)
            vec_rank = r.get("vec_rank", None)
            bm25_raw = r.get("bm25_raw", None)
            vec_raw = r.get("vec_raw", None)

            sources_text.append(
                f"[rrf_score={score:.4f} sources={sources} bm25_rank={bm25_rank} vec_rank={vec_rank} "
                f"bm25_raw={bm25_raw} vec_raw={vec_raw}]\n"
                f"Q: {it['question']}\nA: {it['answer']}\nURL: {it['url']}\n"
            )
            citations.append({"url": it["url"], "question": it["question"]})

        prompt_pair = prompts.render_pair(
            "generate_answer",
            user_query=user_query,
            sources_text=chr(10).join(sources_text),
        )
        messages = [
            SystemMessage(content=prompt_pair["system"]),
            HumanMessage(content=prompt_pair["user"]),
        ]

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

        return {
            "answer": "".join(full).strip(),
            "citations": citations,
        }

    return _run


def node_fallback(state: GraphState) -> GraphState:
    reason = str(state.get("next_node_reason", "")).strip()
    note = ""
    if reason == "max_turns_reached":
        note = "\n（補足）追加確認の上限回数に達したため、ここで終了しました。"

    msg = (
        "申し訳ありません。該当しそうなFAQが見つかりませんでした。\n"
        f"{note}\n\n"
    )
    print(msg + "\n")
    return {"answer": msg}
