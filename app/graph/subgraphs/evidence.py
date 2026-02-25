import logging
from typing import Any

from langgraph.graph import END, StateGraph

from app.core.store import IndexedStore
from app.core.types import GraphState
from app.graph.node_logging import wrap_node
from app.nodes.evidence_finalize import node_evidence_finalize
from app.nodes.evidence_llm import node_evidence_llm_judge
from app.nodes.evidence_router import node_evidence_router
from app.nodes.evidence_rules_strict import node_evidence_rules_strict
from app.prompts.loader import PromptLoader


logger = logging.getLogger(__name__)


def build_evidence_subgraph(
    llm: Any,
    prompts: PromptLoader,
    store: IndexedStore,
):
    graph = StateGraph(GraphState)

    graph.add_node(
        "evidence_rules_strict",
        wrap_node(
            "evidence_rules_strict",
            node_evidence_rules_strict(
                store=store,
                candidate_source="retrieved",
            ),
            group="evidence",
        ),
    )
    graph.add_node(
        "evidence_router",
        wrap_node(
            "evidence_router",
            node_evidence_router(),
            group="evidence",
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
            group="evidence",
        ),
    )
    graph.add_node(
        "evidence_finalize",
        wrap_node(
            "evidence_finalize",
            node_evidence_finalize(),
            group="evidence",
        ),
    )

    graph.set_entry_point("evidence_rules_strict")
    graph.add_edge("evidence_rules_strict", "evidence_router")

    def _evidence_route(state: GraphState) -> str:
        run_llm = bool(state.get("run_evidence_llm", False))
        nxt = "evidence_llm_judge" if run_llm else "evidence_finalize"
        logger.debug("[router][evidence] evidence_router -> %s", nxt)
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
    graph.add_edge("evidence_finalize", END)

    return graph.compile()
