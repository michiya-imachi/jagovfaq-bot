import logging
from typing import Any

from langchain_core.runnables import RunnableLambda

from app.core.types import GraphState


logger = logging.getLogger(__name__)


def wrap_node(name: str, fn: Any, group: str | None = None):
    def _wrapped(state: GraphState) -> GraphState:
        if group:
            logger.debug("[node][%s] %s", group, name)
        else:
            logger.debug("[node] %s", name)
        return fn(state)

    return _wrapped


def with_subgraph_logging(name: str, runnable: Any) -> Any:
    def _log_start(state: GraphState) -> GraphState:
        logger.debug("[subgraph] %s start", name)
        return state

    def _log_end(state: GraphState) -> GraphState:
        logger.debug("[subgraph] %s end", name)
        return state

    return RunnableLambda(_log_start) | runnable | RunnableLambda(_log_end)
