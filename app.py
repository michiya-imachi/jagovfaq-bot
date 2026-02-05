import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

from prompt_loader import PromptLoader
from text_utils import simple_tokenize


class MetaItem(TypedDict):
    id: int
    question: str
    answer: str
    url: str


class GraphState(TypedDict, total=False):
    user_query: str
    retrieved: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, str]]
    need_clarification: bool
    clarifying_question: str
    turn_count: int
    max_turns: int


def resolve_index_dir() -> Path:
    # Prefer explicit path
    env_dir = os.getenv("INDEX_DIR")
    if env_dir:
        return Path(env_dir)

    # Prefer local data/index next to this app.py
    base = Path(__file__).resolve().parent
    p1 = base / "data" / "index"
    if p1.exists():
        return p1

    # Fallback to current working directory
    return Path("data/index")


def load_meta(meta_path: Path) -> Dict[int, MetaItem]:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.jsonl not found: {meta_path}")
    items: Dict[int, MetaItem] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = int(obj["id"])
            items[rid] = {
                "id": rid,
                "question": str(obj["question"]),
                "answer": str(obj["answer"]),
                "url": str(obj.get("url", "")),
            }
    return items


@dataclass
class IndexedStore:
    meta_by_id: Dict[int, MetaItem]
    bm25: Any
    faiss_index: Any
    id_map: List[int]
    embeddings: OpenAIEmbeddings

    @classmethod
    def load(cls, index_dir: Path, embeddings: OpenAIEmbeddings) -> "IndexedStore":
        meta_by_id = load_meta(index_dir / "meta.jsonl")

        with (index_dir / "bm25.pkl").open("rb") as f:
            bm25_obj = pickle.load(f)
        bm25 = bm25_obj["bm25"]

        faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

        with (index_dir / "faiss_map.pkl").open("rb") as f:
            m = pickle.load(f)
        id_map = m["id_map"]

        return cls(
            meta_by_id=meta_by_id,
            bm25=bm25,
            faiss_index=faiss_index,
            id_map=id_map,
            embeddings=embeddings,
        )

    def bm25_search(self, query: str, top_n: int) -> Dict[int, float]:
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = np.argsort(scores)[::-1][:top_n]
        return {int(i): float(scores[int(i)]) for i in ranked}

    def vec_search(self, query: str, top_n: int) -> Dict[int, float]:
        vec = self.embeddings.embed_query(query)
        v = np.array([vec], dtype=np.float32)
        faiss.normalize_L2(v)
        scores, idxs = self.faiss_index.search(v, top_n)
        res: Dict[int, float] = {}
        for score, local_idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if local_idx < 0:
                continue
            rec_id = self.id_map[local_idx]
            res[int(rec_id)] = float(score)
        return res


def make_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return ChatOpenAI(model=model, temperature=0)


def make_embeddings() -> OpenAIEmbeddings:
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def rank_candidates_fusion(
    bm25_scores: Dict[int, float],
    vec_scores: Dict[int, float],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> List[Dict[str, Any]]:
    # Fusion strategy (replaceable later with other rerankers).
    all_ids = set(bm25_scores.keys()) | set(vec_scores.keys())

    def norm_map(m: Dict[int, float]) -> Dict[int, float]:
        if not m:
            return {}
        vals = np.array(list(m.values()), dtype=np.float32)
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax - vmin < 1e-8:
            return {k: 0.0 for k in m.keys()}
        return {k: (v - vmin) / (vmax - vmin) for k, v in m.items()}

    bm25_n = norm_map(bm25_scores)
    vec_n = norm_map(vec_scores)

    candidates: List[Dict[str, Any]] = []
    for rid in all_ids:
        b = float(bm25_n.get(rid, 0.0))
        v = float(vec_n.get(rid, 0.0))
        fused = alpha * b + beta * v
        candidates.append(
            {
                "id": int(rid),
                "fused_score": float(fused),
                "bm25_raw": float(bm25_scores.get(rid, 0.0)),
                "vec_raw": float(vec_scores.get(rid, 0.0)),
            }
        )

    candidates.sort(key=lambda x: x["fused_score"], reverse=True)
    return candidates


def retrieve_hybrid(
    store: IndexedStore,
    query: str,
    bm25_top_n: int = 50,
    vec_top_n: int = 50,
    final_top_k: int = 8,
) -> List[Dict[str, Any]]:
    bm25_scores = store.bm25_search(query, top_n=bm25_top_n)
    vec_scores = store.vec_search(query, top_n=vec_top_n)

    ranked = rank_candidates_fusion(bm25_scores, vec_scores, alpha=1.0, beta=1.0)[
        :final_top_k
    ]

    results: List[Dict[str, Any]] = []
    for c in ranked:
        it = store.meta_by_id.get(c["id"])
        if not it:
            continue
        results.append(
            {
                "item": it,
                "score": c["fused_score"],
                "bm25_raw": c["bm25_raw"],
                "vec_raw": c["vec_raw"],
            }
        )
    return results


def node_retrieve(store: IndexedStore):
    def _run(state: GraphState) -> GraphState:
        query = state["user_query"].strip()
        retrieved = retrieve_hybrid(store, query)
        return {**state, "retrieved": retrieved}

    return _run


def node_route(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved", [])
    turn_count = int(state.get("turn_count", 0))
    max_turns = int(state.get("max_turns", 2))

    if not retrieved:
        need = True
    else:
        top_score = float(retrieved[0]["score"])
        need = top_score < 0.35

    if turn_count >= max_turns:
        need = False

    return {**state, "need_clarification": need}


def node_ask_clarification(llm: ChatOpenAI, prompts: PromptLoader):
    def _run(state: GraphState) -> GraphState:
        retrieved = state.get("retrieved", [])
        user_query = state["user_query"]

        snippets = []
        for r in retrieved[:3]:
            it: MetaItem = r["item"]
            snippets.append(
                f"- Q: {it['question']}\n  A: {it['answer'][:120]}...\n  URL: {it['url']}"
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


def node_generate_answer_stream(llm: ChatOpenAI, prompts: PromptLoader):
    def _run(state: GraphState) -> GraphState:
        user_query = state["user_query"]
        retrieved = state.get("retrieved", [])[:5]

        sources_text = []
        citations: List[Dict[str, str]] = []
        for r in retrieved:
            it: MetaItem = r["item"]
            score = float(r["score"])
            sources_text.append(
                f"[score={score:.3f}]\nQ: {it['question']}\nA: {it['answer']}\nURL: {it['url']}\n"
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


def wrap_node(name: str, fn):
    # Log node execution to stderr to avoid mixing with streamed stdout output.
    def _wrapped(state: GraphState) -> GraphState:
        print(f"[node] {name}", file=sys.stderr, flush=True)
        return fn(state)

    return _wrapped


def build_graph(store: IndexedStore, llm: ChatOpenAI, prompts: PromptLoader):
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", wrap_node("retrieve", node_retrieve(store)))
    graph.add_node("route", wrap_node("route", node_route))
    graph.add_node("ask", wrap_node("ask", node_ask_clarification(llm, prompts)))
    graph.add_node(
        "answer", wrap_node("answer", node_generate_answer_stream(llm, prompts))
    )
    graph.add_node("fallback", wrap_node("fallback", node_fallback))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "route")

    def _router(state: GraphState) -> str:
        retrieved = state.get("retrieved", [])
        need = bool(state.get("need_clarification", False))
        turn_count = int(state.get("turn_count", 0))
        max_turns = int(state.get("max_turns", 2))

        if (not retrieved or need) and turn_count >= max_turns:
            nxt = "fallback"
        elif need:
            nxt = "ask"
        else:
            nxt = "answer"

        print(f"[router] route -> {nxt}", file=sys.stderr, flush=True)
        return nxt

    graph.add_conditional_edges(
        "route", _router, {"ask": "ask", "answer": "answer", "fallback": "fallback"}
    )
    graph.add_edge("ask", END)
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def main() -> None:
    # Load .env from local dir first, then from current working directory
    local_env = Path(__file__).resolve().parent / ".env"
    if local_env.exists():
        load_dotenv(local_env)
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Put it in .env.", file=sys.stderr)
        sys.exit(1)

    index_dir = resolve_index_dir()
    if not index_dir.exists():
        print(f"ERROR: index dir not found: {index_dir}", file=sys.stderr)
        print(
            "If you haven't built the index here, run: uv run python .\\apps\\indexed\\indexer.py",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts = PromptLoader.load_default()

    embeddings = make_embeddings()
    store = IndexedStore.load(index_dir=index_dir, embeddings=embeddings)

    llm = make_llm()
    app = build_graph(store, llm, prompts)

    print("JaGov FAQ Bot - Indexed (BM25+FAISS) + Streaming Answer")
    print(f"Index: {index_dir}")
    print(f"Prompts: {prompts.prompt_dir}")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("> ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        state: GraphState = {"user_query": user_query, "turn_count": 0, "max_turns": 2}

        while True:
            result: GraphState = app.invoke(state)

            if result.get("need_clarification", False):
                print(f"追加質問: {result.get('clarifying_question', '')}")
                extra = input(">> ").strip()
                if not extra:
                    print("（空入力のため終了します）\n")
                    break
                state = {
                    **state,
                    "user_query": f"{state['user_query']}\n補足: {extra}",
                    "turn_count": int(
                        result.get("turn_count", state.get("turn_count", 0))
                    ),
                }
                continue

            # Answer already streamed
            break


if __name__ == "__main__":
    main()
