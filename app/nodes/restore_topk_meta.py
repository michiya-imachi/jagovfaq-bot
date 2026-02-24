import logging
from typing import Any, Dict, List

from app.core.store import IndexedStore
from app.core.types import GraphState, MetaItem


logger = logging.getLogger(__name__)


def node_restore_topk_meta(
    store: IndexedStore,
    input_key: str = "retrieved",
    output_key: str = "retrieved",
):
    def _run(state: GraphState) -> GraphState:
        raw_rows: List[Dict[str, Any]] = state.get(input_key, []) or []
        restored: List[Dict[str, Any]] = []

        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            try:
                rid = int(row["id"])
            except (TypeError, ValueError, KeyError) as e:
                raise ValueError(f"retrieved_candidate_id_invalid: {e}") from e

            item = store.meta_by_id.get(rid)
            if not isinstance(item, dict):
                raise ValueError(f"meta_not_found_for_id:{rid}")

            meta: MetaItem = {
                "id": int(item.get("id", rid)),
                "question": str(item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "url": str(item.get("url", "")),
            }
            restored.append({**row, "id": rid, "item": meta})

        logger.info(
            "[restore-topk-meta] input_key=%s input=%d output_key=%s output=%d clear_intermediate=true",
            input_key,
            len(raw_rows),
            output_key,
            len(restored),
        )

        return {
            output_key: restored,
            "bm25_retrieved": [],
            "vec_retrieved": [],
            "merged_candidates_all": [],
        }

    return _run
