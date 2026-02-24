import logging
from typing import Any, Dict, List, Sequence

from app.core.retriever import (
    RetrieverRegistry,
    normalize_retriever_names,
    parse_retriever_names,
)
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
        run_bm25 = "bm25" in active
        run_vec = "vec" in active

        q = shorten_text(state.get("search_query", ""), 120)
        logger.info(
            '[retrieval-router] active=%s run_bm25=%s run_vec=%s reason=%s search_query="%s"',
            ",".join(active),
            str(run_bm25),
            str(run_vec),
            reason,
            q,
        )

        return {
            "active_retrievers": active,
            "run_bm25": bool(run_bm25),
            "run_vec": bool(run_vec),
            "retrieval_plan_reason": str(reason),
        }

    return _run


def _get_named_retriever(
    retriever_registry: RetrieverRegistry, name: str
) -> Any:
    try:
        return retriever_registry.get(name)
    except KeyError as e:
        raise ValueError(str(e))


def node_retrieve_bm25(retriever_registry: RetrieverRegistry):
    retriever = _get_named_retriever(retriever_registry, "bm25")

    def _run(state: GraphState) -> GraphState:
        if not bool(state.get("run_bm25", True)):
            return {"bm25_retrieved": [], "bm25_count": 0}

        query = str(state.get("search_query", "")).strip()
        rows = retriever.retrieve(query, state)

        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                rid = int(row["id"])
            except (TypeError, ValueError, KeyError):
                continue
            out.append(
                {
                    "id": rid,
                    "bm25_raw": row.get("raw_score"),
                    "bm25_rank": row.get("rank"),
                }
            )

        logger.info(
            '[retrieve] source=bm25 count=%d search_query="%s"',
            len(out),
            shorten_text(query, 120),
        )

        return {
            "bm25_retrieved": out,
            "bm25_count": len(out),
        }

    return _run


def node_retrieve_vec_threshold(retriever_registry: RetrieverRegistry):
    retriever = _get_named_retriever(retriever_registry, "vec")

    def _run(state: GraphState) -> GraphState:
        if not bool(state.get("run_vec", True)):
            return {"vec_retrieved": [], "vec_count": 0, "vec_pass_count": 0}

        query = str(state.get("search_query", "")).strip()
        rows = retriever.retrieve(query, state)

        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                rid = int(row["id"])
            except (TypeError, ValueError, KeyError):
                continue
            passed = bool(row.get("passed", False))
            out.append(
                {
                    "id": rid,
                    "vec_raw": row.get("raw_score"),
                    "vec_rank": row.get("rank"),
                    "vec_pass_threshold": passed,
                }
            )

        pass_count = sum(1 for r in out if bool(r.get("vec_pass_threshold", False)))
        logger.info(
            '[retrieve] source=vec count=%d pass=%d search_query="%s"',
            len(out),
            pass_count,
            shorten_text(query, 120),
        )

        return {
            "vec_retrieved": out,
            "vec_count": len(out),
            "vec_pass_count": int(pass_count),
        }

    return _run
