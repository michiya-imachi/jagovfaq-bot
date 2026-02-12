import logging
from typing import Any, List

from app.core.config import get_env_int
from app.core.types import GraphState


logger = logging.getLogger(__name__)


def node_topk_filter(
    input_key: str,
    output_key: str,
    k_env: str = "FINAL_TOPK",
    default_k: int = 20,
):
    def _run(state: GraphState) -> GraphState:
        items: List[Any] = state.get(input_key, []) or []
        k = max(1, get_env_int(k_env, default_k))
        out = list(items)[:k]

        logger.info(
            "[topk-filter] input_key=%s input=%d k=%d output_key=%s output=%d",
            input_key,
            len(items),
            k,
            output_key,
            len(out),
        )

        return {output_key: out}

    return _run
