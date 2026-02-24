import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.retriever import BM25Retriever, RetrieverRegistry, VectorRetriever
from app.core.store import IndexedStore
from app.core.types import GraphState
from app.nodes.evidence_finalize import node_evidence_finalize
from app.nodes.evidence_llm import node_evidence_llm_judge
from app.nodes.evidence_router import node_evidence_router
from app.nodes.evidence_rules_strict import node_evidence_rules_strict
from app.nodes.hitl import node_hitl_wait_user_input
from app.nodes.qa import (
    node_fallback,
    node_followup_question,
    node_generate_answer_stream,
)
from app.nodes.retrieval import (
    node_retrieval_router,
    node_retrieve_bm25,
    node_retrieve_vec_threshold,
)
from app.nodes.restore_topk_meta import node_restore_topk_meta
from app.nodes.rrf import node_rrf_rank
from app.nodes.routing import make_response_router
from app.nodes.standalone_question import node_standalone_question
from app.nodes.topk_filter import node_topk_filter
from app.nodes.web import node_web_permission, node_web_search
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
        "retrieve_bm25",
        wrap_node("retrieve_bm25", node_retrieve_bm25(retriever_registry)),
    )
    graph.add_node(
        "retrieve_vec",
        wrap_node("retrieve_vec", node_retrieve_vec_threshold(retriever_registry)),
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
        "restore_topk_meta",
        wrap_node(
            "restore_topk_meta",
            node_restore_topk_meta(store=store),
        ),
    )
    graph.add_node(
        "evidence_rules_strict",
        wrap_node(
            "evidence_rules_strict",
            node_evidence_rules_strict(
                store=store,
                candidate_source="retrieved",
            ),
        ),
    )
    graph.add_node(
        "evidence_router",
        wrap_node(
            "evidence_router",
            node_evidence_router(),
        ),
    )
    graph.add_node(
        "evidence_llm_judge",
        wrap_node(
            "evidence_llm_judge",
            node_evidence_llm_judge(
                llm=llm,
                prompts=prompts,
                candidate_source="retrieved",
            ),
        ),
    )
    graph.add_node(
        "evidence_finalize",
        wrap_node(
            "evidence_finalize",
            node_evidence_finalize(),
        ),
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

    # Entry: always build/refresh the standalone search query first.
    graph.set_entry_point("standalone_question")
    graph.add_edge("standalone_question", "retrieval_router")

    # Retrieval flow.
    graph.add_edge("retrieval_router", "retrieve_bm25")
    graph.add_edge("retrieval_router", "retrieve_vec")
    graph.add_edge(["retrieve_bm25", "retrieve_vec"], "rrf_rank")
    graph.add_edge("rrf_rank", "topk_filter")
    graph.add_edge("topk_filter", "restore_topk_meta")
    graph.add_edge("restore_topk_meta", "evidence_rules_strict")
    graph.add_edge("evidence_rules_strict", "evidence_router")

    def _evidence_route(state: GraphState) -> str:
        run_llm = bool(state.get("run_evidence_llm", False))
        nxt = "evidence_llm_judge" if run_llm else "evidence_finalize"
        logger.debug("[router] evidence_router -> %s", nxt)
        return nxt

    graph.add_conditional_edges(
        "evidence_router",
        _evidence_route,
        {
            "evidence_llm_judge": "evidence_llm_judge",
            "evidence_finalize": "evidence_finalize",
        },
    )
    graph.add_edge("evidence_llm_judge", "evidence_finalize")
    graph.add_edge("evidence_finalize", "response_router")

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

    # HITL loop:
    # followup_question -> hitl_input_answer -> standalone_question -> retrieval_router ...
    graph.add_edge("followup_question", "hitl_input_answer")
    graph.add_edge("hitl_input_answer", "standalone_question")
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
