import sys
from typing import Any

from langgraph.graph import END, StateGraph

from app.core.store import IndexedStore
from app.core.types import GraphState
from app.nodes.qa import (
    node_ask_clarification,
    node_fallback,
    node_generate_answer_stream,
)
from app.nodes.retrieval import (
    node_organize_candidates,
    node_retrieval_router,
    node_retrieve_bm25,
    node_retrieve_vec_threshold,
)
from app.nodes.routing import node_retrieve_route
from app.prompts.loader import PromptLoader


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


def build_graph_for_export(prompts: PromptLoader):
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
