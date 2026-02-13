import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.retriever import BM25Retriever, RetrieverRegistry, VectorRetriever
from app.core.store import IndexedStore
from app.core.types import GraphState
from app.nodes.hitl import node_hitl_wait_user_input
from app.nodes.qa import (
    node_fallback,
    node_followup_question,
    node_generate_answer_stream,
)
from app.nodes.retrieval import (
    node_retrieval_router,
    node_retrieve_all,
)
from app.nodes.rrf import node_rrf_rank
from app.nodes.routing import make_decide_next_action
from app.nodes.standalone_question import node_standalone_question
from app.nodes.topk_filter import node_topk_filter
from app.prompts.loader import PromptLoader


logger = logging.getLogger(__name__)


def wrap_node(name: str, fn):
    # Keep debug logs off by default and enable them when --log-level debug is used.
    def _wrapped(state: GraphState) -> GraphState:
        logger.debug("[node] %s", name)
        return fn(state)

    return _wrapped


def build_graph(
    llm: Any,
    prompts: PromptLoader,
    retriever_registry: RetrieverRegistry,
    default_retrievers: list[str],
    checkpointer: Optional[Any] = None,
):
    graph = StateGraph(GraphState)

    graph.add_node(
        "standalone_question",
        wrap_node("standalone_question", node_standalone_question(llm, prompts)),
    )

    graph.add_node(
        "retrieval_router",
        wrap_node(
            "retrieval_router",
            node_retrieval_router(
                retriever_registry=retriever_registry,
                default_retrievers=default_retrievers,
            ),
        ),
    )
    graph.add_node(
        "retrieve_all",
        wrap_node("retrieve_all", node_retrieve_all(retriever_registry)),
    )

    graph.add_node("rrf_rank", wrap_node("rrf_rank", node_rrf_rank()))
    graph.add_node(
        "topk_filter",
        wrap_node(
            "topk_filter",
            node_topk_filter(input_key="merged_candidates_all", output_key="retrieved"),
        ),
    )

    graph.add_node(
        "decide_next_action",
        wrap_node(
            "decide_next_action",
            make_decide_next_action(top_n=10, candidate_source="merged_candidates_all"),
        ),
    )

    graph.add_node(
        "followup_question",
        wrap_node("followup_question", node_followup_question(llm, prompts)),
    )

    graph.add_node("HITL", wrap_node("HITL", node_hitl_wait_user_input()))

    graph.add_node(
        "answer", wrap_node("answer", node_generate_answer_stream(llm, prompts))
    )
    graph.add_node("fallback", wrap_node("fallback", node_fallback))

    # Entry: always build/refresh the standalone search query first.
    graph.set_entry_point("standalone_question")
    graph.add_edge("standalone_question", "retrieval_router")

    # Retrieval flow.
    graph.add_edge("retrieval_router", "retrieve_all")
    graph.add_edge("retrieve_all", "rrf_rank")
    graph.add_edge("rrf_rank", "topk_filter")
    graph.add_edge("topk_filter", "decide_next_action")

    def _router(state: GraphState) -> str:
        nxt = str(state.get("next_node", "answer") or "answer")
        if nxt not in {"followup_question", "answer", "fallback"}:
            nxt = "answer"
        logger.debug("[router] decide_next_action -> %s", nxt)
        return nxt

    # IMPORTANT: the router returns the destination node name directly.
    graph.add_conditional_edges(
        "decide_next_action",
        _router,
        {
            "followup_question": "followup_question",
            "answer": "answer",
            "fallback": "fallback",
        },
    )

    # HITL loop:
    # followup_question -> HITL -> standalone_question -> retrieval_router ...
    graph.add_edge("followup_question", "HITL")
    graph.add_edge("HITL", "standalone_question")

    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


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
    dummy_store = IndexedStore(
        meta_by_id={},
        bm25=None,
        faiss_index=None,
        id_map=[],
        embeddings=None,
    )
    null_llm = _NullLLM()
    retriever_registry = RetrieverRegistry(
        [
            BM25Retriever(dummy_store),
            VectorRetriever(dummy_store),
        ]
    )
    return build_graph(
        llm=null_llm,
        prompts=prompts,
        retriever_registry=retriever_registry,
        default_retrievers=["bm25", "vec"],
    )
