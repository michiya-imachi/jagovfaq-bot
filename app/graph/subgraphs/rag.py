from typing import Any

from langgraph.graph import END, StateGraph

from app.core.retriever import RetrieverRegistry
from app.core.store import IndexedStore
from app.core.types import GraphState
from app.graph.node_logging import wrap_node
from app.nodes.retrieval import (
    node_retrieval_router,
    node_retrieve_bm25,
    node_retrieve_vec_threshold,
)
from app.nodes.restore_topk_meta import node_restore_topk_meta
from app.nodes.rrf import node_rrf_rank
from app.nodes.standalone_question import node_standalone_question
from app.nodes.topk_filter import node_topk_filter
from app.prompts.loader import PromptLoader


def build_rag_subgraph(
    llm: Any,
    prompts: PromptLoader,
    retriever_registry: RetrieverRegistry,
    default_retrievers: list[str],
    store: IndexedStore,
):
    graph = StateGraph(GraphState)

    graph.add_node(
        "standalone_question",
        wrap_node(
            "standalone_question",
            node_standalone_question(llm, prompts),
            group="rag",
        ),
    )
    graph.add_node(
        "retrieval_router",
        wrap_node(
            "retrieval_router",
            node_retrieval_router(
                retriever_registry=retriever_registry,
                default_retrievers=default_retrievers,
            ),
            group="rag",
        ),
    )
    graph.add_node(
        "retrieve_bm25",
        wrap_node(
            "retrieve_bm25",
            node_retrieve_bm25(retriever_registry),
            group="rag",
        ),
    )
    graph.add_node(
        "retrieve_vec",
        wrap_node(
            "retrieve_vec",
            node_retrieve_vec_threshold(retriever_registry),
            group="rag",
        ),
    )
    graph.add_node("rrf_rank", wrap_node("rrf_rank", node_rrf_rank(), group="rag"))
    graph.add_node(
        "topk_filter",
        wrap_node(
            "topk_filter",
            node_topk_filter(input_key="merged_candidates_all", output_key="retrieved"),
            group="rag",
        ),
    )
    graph.add_node(
        "restore_topk_meta",
        wrap_node(
            "restore_topk_meta",
            node_restore_topk_meta(store=store),
            group="rag",
        ),
    )

    graph.set_entry_point("standalone_question")
    graph.add_edge("standalone_question", "retrieval_router")
    graph.add_edge("retrieval_router", "retrieve_bm25")
    graph.add_edge("retrieval_router", "retrieve_vec")
    graph.add_edge(["retrieve_bm25", "retrieve_vec"], "rrf_rank")
    graph.add_edge("rrf_rank", "topk_filter")
    graph.add_edge("topk_filter", "restore_topk_meta")
    graph.add_edge("restore_topk_meta", END)

    return graph.compile()
