import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.retriever import BM25Retriever, RetrieverRegistry, VectorRetriever
from app.core.store import IndexedStore
from app.core.types import GraphState
from app.graph.node_logging import with_subgraph_logging, wrap_node
from app.graph.subgraphs.evidence import build_evidence_subgraph
from app.graph.subgraphs.rag import build_rag_subgraph
from app.nodes.hitl import node_hitl_wait_user_input
from app.nodes.qa import (
    node_fallback,
    node_followup_question,
    node_generate_answer_stream,
)
from app.nodes.routing import make_response_router
from app.nodes.web import node_web_permission, node_web_search
from app.prompts.loader import PromptLoader


logger = logging.getLogger(__name__)


def build_graph(
    llm: Any,
    openai_client: Any,
    prompts: PromptLoader,
    retriever_registry: RetrieverRegistry,
    default_retrievers: list[str],
    checkpointer: Optional[Any] = None,
    store: Optional[IndexedStore] = None,
):
    if store is None:
        # Keep backward compatibility with previous build_graph() call sites.
        bm25_retriever = retriever_registry.get("bm25")
        inferred_store = getattr(bm25_retriever, "store", None)
        if not isinstance(inferred_store, IndexedStore):
            raise ValueError("store is required for restore/evidence strict nodes")
        store = inferred_store

    rag_subgraph = build_rag_subgraph(
        llm=llm,
        prompts=prompts,
        retriever_registry=retriever_registry,
        default_retrievers=default_retrievers,
        store=store,
    )
    evidence_subgraph = build_evidence_subgraph(
        llm=llm,
        prompts=prompts,
        store=store,
    )

    graph = StateGraph(GraphState)

    graph.add_node(
        "rag",
        with_subgraph_logging("rag", rag_subgraph),
    )
    graph.add_node(
        "evidence",
        with_subgraph_logging("evidence", evidence_subgraph),
    )
    graph.add_node(
        "response_router",
        wrap_node(
            "response_router",
            make_response_router(),
        ),
    )

    graph.add_node(
        "followup_question",
        wrap_node("followup_question", node_followup_question(llm, prompts)),
    )

    graph.add_node(
        "hitl_input_answer",
        wrap_node("hitl_input_answer", node_hitl_wait_user_input()),
    )
    graph.add_node(
        "hitl_permission_web",
        wrap_node("hitl_permission_web", node_web_permission()),
    )
    graph.add_node(
        "web_search",
        wrap_node("web_search", node_web_search(openai_client)),
    )

    graph.add_node(
        "answer", wrap_node("answer", node_generate_answer_stream(llm, prompts))
    )
    graph.add_node("fallback", wrap_node("fallback", node_fallback))

    graph.set_entry_point("rag")
    graph.add_edge("rag", "evidence")
    graph.add_edge("evidence", "response_router")

    def _router(state: GraphState) -> str:
        nxt = str(state.get("next_node", "fallback") or "fallback")
        if nxt not in {
            "followup_question",
            "hitl_permission_web",
            "answer",
            "fallback",
        }:
            nxt = "fallback"
        logger.debug("[router] response_router -> %s", nxt)
        return nxt

    # IMPORTANT: the router returns the destination node name directly.
    graph.add_conditional_edges(
        "response_router",
        _router,
        {
            "followup_question": "followup_question",
            "hitl_permission_web": "hitl_permission_web",
            "answer": "answer",
            "fallback": "fallback",
        },
    )

    graph.add_edge("followup_question", "hitl_input_answer")
    graph.add_edge("hitl_input_answer", "rag")
    graph.add_conditional_edges(
        "hitl_permission_web",
        lambda state: "web_search"
        if bool(state.get("web_search_allowed", False))
        else "answer",
        {
            "web_search": "web_search",
            "answer": "answer",
        },
    )
    graph.add_edge("web_search", "answer")

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


class _NullOpenAIClient:
    class _Responses:
        @staticmethod
        def create(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(
                "NullOpenAIClient: responses.create() was called in --export-graph mode."
            )

    responses = _Responses()


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
        openai_client=_NullOpenAIClient(),
        prompts=prompts,
        store=dummy_store,
        retriever_registry=retriever_registry,
        default_retrievers=["bm25", "vec"],
    )
