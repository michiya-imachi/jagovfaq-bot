import logging
from typing import Any, Dict, List, Sequence

from app.core.retriever import RetrieverRegistry, normalize_retriever_names, parse_retriever_names
from app.core.types import GraphState


logger = logging.getLogger(__name__)


def shorten_text(text: Any, max_len: int) -> str:
    # Truncate for log readability.
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "..."


def _names_from_value(value: Any) -> List[str]:
    if isinstance(value, str):
        return parse_retriever_names(value)
    if isinstance(value, (list, tuple)):
        return normalize_retriever_names([str(v) for v in value])
    return []


def node_retrieval_router(
    retriever_registry: RetrieverRegistry, default_retrievers: Sequence[str]
):
    defaults = normalize_retriever_names(default_retrievers)
    if not defaults:
        raise ValueError("default_retrievers must not be empty.")

    def _run(state: GraphState) -> GraphState:
        requested = _names_from_value(state.get("requested_retrievers"))
        if requested:
            selected = retriever_registry.select(requested)
            reason = "state:requested_retrievers"
        else:
            selected = retriever_registry.select(defaults)
            reason = "default_retrievers"

        active = [r.name for r in selected]

        q = shorten_text(state.get("search_query", ""), 120)
        logger.info(
            '[retrieval-router] active=%s reason=%s search_query="%s"',
            ",".join(active),
            reason,
            q,
        )

        return {
            "active_retrievers": active,
            "retrieval_plan_reason": str(reason),
        }

    return _run


def node_retrieve_all(retriever_registry: RetrieverRegistry):
    def _run(state: GraphState) -> GraphState:
        query = str(state.get("search_query", "")).strip()
        active = state.get("active_retrievers", []) or []
        selected = retriever_registry.select([str(name) for name in active])

        results_by_source: Dict[str, List[Dict[str, Any]]] = {}
        counts: Dict[str, int] = {}

        for retriever in selected:
            rows = retriever.retrieve(query, state)
            results_by_source[retriever.name] = list(rows)
            counts[retriever.name] = len(rows)
            logger.info(
                '[retrieve] source=%s count=%d search_query="%s"',
                retriever.name,
                len(rows),
                shorten_text(query, 120),
            )

        return {
            "retrieval_results_by_source": results_by_source,
            "retrieval_counts": counts,
        }

    return _run
