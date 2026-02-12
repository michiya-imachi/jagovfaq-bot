import sys
from typing import Any, Dict, List

from app.core.types import GraphState, MetaItem
from app.prompts.loader import PromptLoader


def node_ask_clarification(llm: Any, prompts: PromptLoader):
    def _run(state: GraphState) -> GraphState:
        retrieved = state.get("retrieved", [])
        user_query = state["user_query"]

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

        prompt = prompts.render(
            "ask_clarification",
            user_query=user_query,
            context=context,
        )

        msg = llm.invoke(prompt)
        q = msg.content.strip()
        turn_count = int(state.get("turn_count", 0)) + 1
        return {
            **state,
            "clarifying_question": q,
            "need_clarification": True,
            "turn_count": turn_count,
        }

    return _run


def node_generate_answer_stream(llm: Any, prompts: PromptLoader):
    def _run(state: GraphState) -> GraphState:
        user_query = state["user_query"]
        retrieved = state.get("retrieved", [])[:5]

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

        prompt = prompts.render(
            "generate_answer",
            user_query=user_query,
            sources_text=chr(10).join(sources_text),
        )

        sys.stdout.flush()
        full: List[str] = []
        for chunk in llm.stream(prompt):
            text = getattr(chunk, "content", "")
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                full.append(text)
        sys.stdout.write("\n\n")
        sys.stdout.flush()

        return {
            **state,
            "answer": "".join(full).strip(),
            "citations": citations,
            "need_clarification": False,
        }

    return _run


def node_fallback(state: GraphState) -> GraphState:
    msg = (
        "回答: 申し訳ありません。該当しそうなFAQが見つかりませんでした。\n"
        "根拠:\n- （該当なし）\n\n"
        "よろしければ、手続き名・画面名・エラーメッセージなど具体的な情報を教えてください。"
    )
    print(msg + "\n")
    return {**state, "answer": msg, "need_clarification": False}
