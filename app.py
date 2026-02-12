import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Optional, Tuple

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

    # Retrieval planning
    run_bm25: bool
    run_vec: bool
    retrieval_plan_reason: str

    # Raw retrieval outputs (optional)
    bm25_retrieved: List[Dict[str, Any]]
    vec_retrieved: List[Dict[str, Any]]

    # Final organized candidates (used downstream by ask/answer)
    retrieved: List[Dict[str, Any]]

    answer: str
    citations: List[Dict[str, str]]

    need_clarification: bool
    clarifying_question: str

    # Turn control
    turn_count: int
    max_turns: int

    # Retrieval diagnostics (optional)
    bm25_count: int
    vec_count: int
    vec_pass_count: int


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
    embeddings: Optional[OpenAIEmbeddings]

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
        if self.bm25 is None:
            raise RuntimeError("BM25 is not configured.")
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = np.argsort(scores)[::-1][:top_n]
        return {int(i): float(scores[int(i)]) for i in ranked}

    def vec_search(self, query: str, top_n: int) -> Dict[int, float]:
        if self.embeddings is None:
            raise RuntimeError("Embeddings are not configured.")
        if self.faiss_index is None:
            raise RuntimeError("FAISS index is not configured.")

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


def _get_env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
        return v
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, str(default)))
        return v
    except Exception:
        return default


def _sorted_items_desc(m: Dict[int, float]) -> List[Tuple[int, float]]:
    return sorted(m.items(), key=lambda x: x[1], reverse=True)


def node_retrieval_router():
    def _run(state: GraphState) -> GraphState:
        # This node decides which retrievers to run.
        # Default: run both. You can override via env var RETRIEVAL_MODE.
        # Supported values: "both", "bm25", "vec"
        mode = str(os.getenv("RETRIEVAL_MODE", "")).strip().lower()

        if mode in {"bm25", "keyword"}:
            run_bm25, run_vec, reason = True, False, f"env:{mode}"
        elif mode in {"vec", "vector"}:
            run_bm25, run_vec, reason = False, True, f"env:{mode}"
        else:
            run_bm25, run_vec, reason = (
                True,
                True,
                ("env:both" if mode in {"both", "hybrid"} else "default_both"),
            )

        # Return only updated keys to avoid conflicts.
        return {
            "run_bm25": bool(run_bm25),
            "run_vec": bool(run_vec),
            "retrieval_plan_reason": str(reason),
        }

    return _run


# --- Legacy fusion retriever (kept for reference) --------------------------------
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
    # Legacy single-node hybrid retriever (unused in the new graph).
    def _run(state: GraphState) -> GraphState:
        query = state["user_query"].strip()
        retrieved = retrieve_hybrid(store, query)
        return {**state, "retrieved": retrieved}

    return _run


# --- New retrieval pipeline (BM25 node + Vector node + Organizer node) -----------
def node_retrieve_bm25(store: IndexedStore):
    def _run(state: GraphState) -> GraphState:
        if not bool(state.get("run_bm25", True)):
            # Return empty results to avoid stale data in iterative runs.
            return {
                "bm25_retrieved": [],
                "bm25_count": 0,
            }

        query = state["user_query"].strip()
        topk = max(1, _get_env_int("BM25_TOPK", 10))

        bm25_scores = store.bm25_search(query, top_n=topk)
        ranked = _sorted_items_desc(bm25_scores)

        out: List[Dict[str, Any]] = []
        for rank, (rid, score) in enumerate(ranked, start=1):
            it = store.meta_by_id.get(int(rid))
            if not it:
                continue
            out.append(
                {
                    "id": int(rid),
                    "item": it,
                    "bm25_raw": float(score),
                    "bm25_rank": int(rank),
                }
            )

        # Return only updated keys to avoid conflicts in parallel execution.
        return {
            "bm25_retrieved": out,
            "bm25_count": len(out),
        }

    return _run


def node_retrieve_vec_threshold(store: IndexedStore):
    def _run(state: GraphState) -> GraphState:
        if not bool(state.get("run_vec", True)):
            # Return empty results to avoid stale data in iterative runs.
            return {
                "vec_retrieved": [],
                "vec_count": 0,
                "vec_pass_count": 0,
            }

        query = state["user_query"].strip()

        search_topn = max(1, _get_env_int("VEC_SEARCH_TOPN", 200))
        threshold = _get_env_float("VEC_THRESHOLD", 0.35)
        max_keep = max(1, _get_env_int("VEC_MAX_KEEP", 30))
        fallback_topk = max(1, _get_env_int("VEC_FALLBACK_TOPK", 30))

        vec_scores = store.vec_search(query, top_n=search_topn)
        ranked = _sorted_items_desc(vec_scores)

        passed: List[Tuple[int, float, int]] = []
        for rank, (rid, score) in enumerate(ranked, start=1):
            if float(score) >= threshold:
                passed.append((int(rid), float(score), int(rank)))
                if len(passed) >= max_keep:
                    break

        use_fallback = len(passed) == 0
        picked: List[Tuple[int, float, int]] = []
        if not use_fallback:
            picked = passed
        else:
            # Fallback: keep top results to avoid empty candidate set.
            for rank, (rid, score) in enumerate(ranked[:fallback_topk], start=1):
                picked.append((int(rid), float(score), int(rank)))
                if len(picked) >= max_keep:
                    break

        out: List[Dict[str, Any]] = []
        for rid, score, rank in picked:
            it = store.meta_by_id.get(int(rid))
            if not it:
                continue
            out.append(
                {
                    "id": int(rid),
                    "item": it,
                    "vec_raw": float(score),
                    "vec_rank": int(rank),
                    "vec_pass_threshold": (not use_fallback)
                    and (float(score) >= threshold),
                }
            )

        pass_count = sum(1 for r in out if r.get("vec_pass_threshold", False))

        # Return only updated keys to avoid conflicts in parallel execution.
        return {
            "vec_retrieved": out,
            "vec_count": len(out),
            "vec_pass_count": int(pass_count),
        }

    return _run


def node_organize_candidates():
    def _run(state: GraphState) -> GraphState:
        bm25 = state.get("bm25_retrieved", []) or []
        vec = state.get("vec_retrieved", []) or []

        rrf_k = max(1, _get_env_int("RRF_K", 60))
        final_topk = max(1, _get_env_int("FINAL_TOPK", 20))

        by_id: Dict[int, Dict[str, Any]] = {}

        for r in bm25:
            rid = int(r["id"])
            it: MetaItem = r["item"]
            by_id[rid] = {
                "id": rid,
                "item": it,
                "bm25_raw": r.get("bm25_raw"),
                "bm25_rank": r.get("bm25_rank"),
                "vec_raw": None,
                "vec_rank": None,
                "vec_pass_threshold": False,
                "sources": ["bm25"],
            }

        for r in vec:
            rid = int(r["id"])
            it: MetaItem = r["item"]
            if rid not in by_id:
                by_id[rid] = {
                    "id": rid,
                    "item": it,
                    "bm25_raw": None,
                    "bm25_rank": None,
                    "vec_raw": r.get("vec_raw"),
                    "vec_rank": r.get("vec_rank"),
                    "vec_pass_threshold": bool(r.get("vec_pass_threshold", False)),
                    "sources": ["vec"],
                }
            else:
                by_id[rid]["vec_raw"] = r.get("vec_raw")
                by_id[rid]["vec_rank"] = r.get("vec_rank")
                by_id[rid]["vec_pass_threshold"] = bool(
                    r.get("vec_pass_threshold", False)
                )
                if "vec" not in by_id[rid]["sources"]:
                    by_id[rid]["sources"].append("vec")

        organized: List[Dict[str, Any]] = []
        for rid, r in by_id.items():
            bm25_rank = r.get("bm25_rank")
            vec_rank = r.get("vec_rank")

            score = 0.0
            if isinstance(bm25_rank, int) and bm25_rank > 0:
                score += 1.0 / float(rrf_k + bm25_rank)
            if isinstance(vec_rank, int) and vec_rank > 0:
                score += 1.0 / float(rrf_k + vec_rank)

            has_both = ("bm25" in r.get("sources", [])) and (
                "vec" in r.get("sources", [])
            )

            organized.append(
                {
                    **r,
                    "has_both": bool(has_both),
                    "score": float(score),
                }
            )

        organized.sort(key=lambda x: x["score"], reverse=True)
        organized = organized[:final_topk]

        return {
            **state,
            "retrieved": organized,
        }

    return _run


def _shorten_text(text: str, max_len: int) -> str:
    # Truncate for log readability.
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _log_retrieve_route_debug(
    state: GraphState,
    retrieved: List[Dict[str, Any]],
    need_clarification: bool,
    turn_count: int,
    max_turns: int,
) -> None:
    # Debug log for routing and retrieved references. Print to stderr to avoid
    # mixing with streamed stdout output in the answer node.
    try:
        topk = int(os.getenv("ROUTE_LOG_TOPK", "5"))
    except Exception:
        topk = 5
    if topk <= 0:
        return

    user_query = _shorten_text(state.get("user_query", ""), 120)
    cand_count = len(retrieved)

    top = retrieved[0] if retrieved else None
    top_score = top.get("score") if isinstance(top, dict) else None
    top_sources = ",".join(top.get("sources", [])) if isinstance(top, dict) else ""
    top_vec_raw = top.get("vec_raw") if isinstance(top, dict) else None

    top_score_str = (
        f"{float(top_score):.4f}" if isinstance(top_score, (int, float)) else "None"
    )
    top_vec_str = (
        f"{float(top_vec_raw):.3f}" if isinstance(top_vec_raw, (int, float)) else "None"
    )

    plan = f"bm25={bool(state.get('run_bm25', True))} vec={bool(state.get('run_vec', True))}"
    plan_reason = _shorten_text(state.get("retrieval_plan_reason", ""), 40)

    print(
        f"[retrieve-route-debug] turn={turn_count}/{max_turns} "
        f"need_clarification={need_clarification} "
        f"plan={plan} plan_reason={plan_reason} "
        f"candidates={cand_count} top_score={top_score_str} top_sources={top_sources} top_vec_raw={top_vec_str} "
        f'user_query="{user_query}"',
        file=sys.stderr,
        flush=True,
    )

    for i, r in enumerate(retrieved[:topk], start=1):
        it = r.get("item", {}) or {}
        rid = r.get("id", "")
        score = r.get("score", None)
        sources = ",".join(r.get("sources", []))
        bm25_rank = r.get("bm25_rank", None)
        vec_rank = r.get("vec_rank", None)
        vec_raw = r.get("vec_raw", None)

        q = _shorten_text(it.get("question", ""), 80)
        url = _shorten_text(it.get("url", ""), 160)

        score_str = f"{float(score):.4f}" if isinstance(score, (int, float)) else "None"
        bm25_rank_str = str(bm25_rank) if isinstance(bm25_rank, int) else "-"
        vec_rank_str = str(vec_rank) if isinstance(vec_rank, int) else "-"
        vec_raw_str = (
            f"{float(vec_raw):.3f}" if isinstance(vec_raw, (int, float)) else "-"
        )

        print(
            f"[retrieve-route-debug] #{i} id={rid} score={score_str} sources={sources} bm25_rank={bm25_rank_str} vec_rank={vec_rank_str} vec_raw={vec_raw_str} "
            f'Q="{q}" url="{url}"',
            file=sys.stderr,
            flush=True,
        )


def node_retrieve_route(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved", []) or []
    turn_count = int(state.get("turn_count", 0))
    max_turns = int(state.get("max_turns", 2))

    run_bm25 = bool(state.get("run_bm25", True))
    run_vec = bool(state.get("run_vec", True))

    vec_threshold = _get_env_float("VEC_THRESHOLD", 0.35)
    vec_strong_threshold = _get_env_float("VEC_STRONG_THRESHOLD", 0.45)

    if not retrieved:
        need = True
    else:
        top = retrieved[0]

        top_vec_raw = top.get("vec_raw")
        top_vec_ok = (
            isinstance(top_vec_raw, (int, float))
            and float(top_vec_raw) >= vec_threshold
        )
        top_vec_strong = (
            isinstance(top_vec_raw, (int, float))
            and float(top_vec_raw) >= vec_strong_threshold
        )

        # If the plan disables one retriever, relax the confidence heuristic accordingly.
        if run_bm25 and (not run_vec):
            # BM25-only mode: answer as long as we have candidates.
            need = False
        elif run_vec and (not run_bm25):
            # Vector-only mode: require at least a basic similarity threshold.
            need = not bool(top_vec_ok)
        else:
            top_has_both = bool(top.get("has_both", False))

            any_both_top3 = any(bool(r.get("has_both", False)) for r in retrieved[:3])
            any_vec_strong_top3 = any(
                isinstance(r.get("vec_raw"), (int, float))
                and float(r["vec_raw"]) >= vec_strong_threshold
                for r in retrieved[:3]
            )

            # Conservative confidence heuristic (hybrid mode):
            # - If the top candidate is supported by both retrievers and vector similarity is at least OK -> answer.
            # - If vector similarity is strong -> answer.
            # - If both sources agree among top-3 and vector is strong in top-3 -> answer.
            # Otherwise -> ask clarification.
            if top_has_both and top_vec_ok:
                need = False
            elif top_vec_strong:
                need = False
            elif any_both_top3 and any_vec_strong_top3:
                need = False
            else:
                need = True

    if turn_count >= max_turns:
        need = False

    _log_retrieve_route_debug(
        state=state,
        retrieved=retrieved,
        need_clarification=need,
        turn_count=turn_count,
        max_turns=max_turns,
    )

    return {**state, "need_clarification": need}


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


def wrap_node(name: str, fn):
    # Log node execution to stderr to avoid mixing with streamed stdout output.
    def _wrapped(state: GraphState) -> GraphState:
        print(f"[node] {name}", file=sys.stderr, flush=True)
        return fn(state)

    return _wrapped


def build_graph(store: IndexedStore, llm: Any, prompts: PromptLoader):
    graph = StateGraph(GraphState)

    graph.add_node(
        "retrieval_router", wrap_node("retrieval_router", node_retrieval_router())
    )
    graph.add_node(
        "retrieve_bm25", wrap_node("retrieve_bm25", node_retrieve_bm25(store))
    )
    graph.add_node(
        "retrieve_vec", wrap_node("retrieve_vec", node_retrieve_vec_threshold(store))
    )
    graph.add_node("organize", wrap_node("organize", node_organize_candidates()))

    graph.add_node("retrieve_route", wrap_node("retrieve_route", node_retrieve_route))
    graph.add_node("ask", wrap_node("ask", node_ask_clarification(llm, prompts)))
    graph.add_node(
        "answer", wrap_node("answer", node_generate_answer_stream(llm, prompts))
    )
    graph.add_node("fallback", wrap_node("fallback", node_fallback))

    # Entry -> router -> parallel retrievals -> join -> organizer -> retrieve_route.
    graph.set_entry_point("retrieval_router")
    graph.add_edge("retrieval_router", "retrieve_bm25")
    graph.add_edge("retrieval_router", "retrieve_vec")
    graph.add_edge(["retrieve_bm25", "retrieve_vec"], "organize")
    graph.add_edge("organize", "retrieve_route")

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

        print(f"[router] retrieve_route -> {nxt}", file=sys.stderr, flush=True)
        return nxt

    graph.add_conditional_edges(
        "retrieve_route",
        _router,
        {"ask": "ask", "answer": "answer", "fallback": "fallback"},
    )
    graph.add_edge("ask", END)
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()


class _NullLLM:
    def invoke(self, _prompt: str) -> Any:
        raise RuntimeError(
            "NullLLM: invoke() was called. This should not happen in --export-graph mode."
        )

    def stream(self, _prompt: str):
        raise RuntimeError(
            "NullLLM: stream() was called. This should not happen in --export-graph mode."
        )


def _build_graph_for_export(prompts: PromptLoader):
    # Build the graph without requiring OpenAI API key or local index files.
    dummy_store = IndexedStore(
        meta_by_id={},
        bm25=None,
        faiss_index=None,
        id_map=[],
        embeddings=None,
    )
    null_llm = _NullLLM()
    return build_graph(store=dummy_store, llm=null_llm, prompts=prompts)


def export_graph_artifacts(app: Any, out_dir: Path) -> Tuple[Path, Optional[Path]]:
    # Export Mermaid (.mmd) and PNG (.png) for the compiled graph.
    out_dir.mkdir(parents=True, exist_ok=True)

    g = app.get_graph()

    mermaid_text = g.draw_mermaid()
    mmd_path = out_dir / "graph.mmd"
    mmd_path.write_text(str(mermaid_text).rstrip() + "\n", encoding="utf-8")

    png_path = out_dir / "graph.png"
    png_ok = True

    try:
        # Some versions support saving directly via output_file_path.
        g.draw_mermaid_png(output_file_path=str(png_path))
    except TypeError:
        # Other versions return PNG bytes.
        png_bytes = g.draw_mermaid_png()
        if isinstance(png_bytes, (bytes, bytearray)):
            png_path.write_bytes(png_bytes)
        else:
            png_path.write_bytes(bytes(png_bytes))
    except Exception as e:
        png_ok = False
        print(
            f"WARNING: graph PNG export failed: {type(e).__name__}: {e}",
            file=sys.stderr,
            flush=True,
        )

    return mmd_path, (png_path if png_ok else None)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JaGov FAQ Bot - Indexed (BM25+FAISS) + Streaming Answer"
    )
    parser.add_argument(
        "--export-graph",
        action="store_true",
        help="Export the LangGraph structure as Mermaid (.mmd) and PNG (.png), then exit.",
    )
    parser.add_argument(
        "--graph-out-dir",
        default="out",
        help="Output directory for graph artifacts when --export-graph is set (default: out).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    # Load .env from local dir first, then from current working directory
    local_env = Path(__file__).resolve().parent / ".env"
    if local_env.exists():
        load_dotenv(local_env)
    load_dotenv()

    if args.export_graph:
        prompts = PromptLoader.load_default()
        app = _build_graph_for_export(prompts)

        out_dir = Path(args.graph_out_dir)
        mmd_path, png_path = export_graph_artifacts(app, out_dir=out_dir)

        print("Graph exported:")
        print(f"- Mermaid: {mmd_path}")
        if png_path is not None:
            print(f"- PNG: {png_path}")
        else:
            print("- PNG: (failed)")
        return

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
