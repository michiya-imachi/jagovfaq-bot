from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from app.core.candidate import RetrievedCandidate
from app.core.config import get_env_float, get_env_int
from app.core.store import IndexedStore
from app.core.types import GraphState


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, state: GraphState) -> List[RetrievedCandidate]:
        ...


def parse_retriever_names(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [part.strip().lower() for part in str(raw).split(",") if part.strip()]


def normalize_retriever_names(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for name in names:
        key = str(name).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


@dataclass(frozen=True)
class BM25Retriever:
    store: IndexedStore
    name: str = "bm25"

    def retrieve(self, query: str, state: GraphState) -> List[RetrievedCandidate]:
        topk = max(1, get_env_int("BM25_TOPK", 10))
        scores = self.store.bm25_search(query, top_n=topk)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        out: List[RetrievedCandidate] = []
        for rank, (rid, score) in enumerate(ranked, start=1):
            item = self.store.meta_by_id.get(int(rid))
            if not item:
                continue
            out.append(
                {
                    "id": int(rid),
                    "item": item,
                    "retriever": self.name,
                    "raw_score": float(score),
                    "rank": int(rank),
                    "passed": None,
                    "features": {"bm25_raw": float(score)},
                }
            )
        return out


@dataclass(frozen=True)
class VectorRetriever:
    store: IndexedStore
    name: str = "vec"

    def retrieve(self, query: str, state: GraphState) -> List[RetrievedCandidate]:
        search_topn = max(1, get_env_int("VEC_SEARCH_TOPN", 200))
        threshold = get_env_float("VEC_THRESHOLD", 0.35)
        max_keep = max(1, get_env_int("VEC_MAX_KEEP", 30))
        fallback_topk = max(1, get_env_int("VEC_FALLBACK_TOPK", 30))

        scores = self.store.vec_search(query, top_n=search_topn)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        passed_rows: List[Tuple[int, float, int]] = []
        for rank, (rid, score) in enumerate(ranked, start=1):
            if float(score) >= threshold:
                passed_rows.append((int(rid), float(score), int(rank)))
                if len(passed_rows) >= max_keep:
                    break

        use_fallback = len(passed_rows) == 0
        picked: List[Tuple[int, float, int]] = []
        if use_fallback:
            for rank, (rid, score) in enumerate(ranked[:fallback_topk], start=1):
                picked.append((int(rid), float(score), int(rank)))
                if len(picked) >= max_keep:
                    break
        else:
            picked = passed_rows

        out: List[RetrievedCandidate] = []
        for rid, score, rank in picked:
            item = self.store.meta_by_id.get(int(rid))
            if not item:
                continue
            passed = (not use_fallback) and (float(score) >= threshold)
            out.append(
                {
                    "id": int(rid),
                    "item": item,
                    "retriever": self.name,
                    "raw_score": float(score),
                    "rank": int(rank),
                    "passed": bool(passed),
                    "features": {
                        "vec_raw": float(score),
                        "vec_pass_threshold": bool(passed),
                    },
                }
            )
        return out


class RetrieverRegistry:
    def __init__(self, retrievers: Iterable[Retriever]) -> None:
        by_name: Dict[str, Retriever] = {}
        for retriever in retrievers:
            name = str(retriever.name).strip().lower()
            if not name:
                raise ValueError("Retriever name must not be empty.")
            if name in by_name:
                raise ValueError(f"Duplicate retriever name: {name}")
            by_name[name] = retriever
        self._by_name = by_name

    def names(self) -> List[str]:
        return list(self._by_name.keys())

    def get(self, name: str) -> Retriever:
        key = str(name).strip().lower()
        if key not in self._by_name:
            raise KeyError(f"Unknown retriever: {name}")
        return self._by_name[key]

    def select(self, names: Sequence[str]) -> List[Retriever]:
        normalized = normalize_retriever_names(names)
        if not normalized:
            raise ValueError("No retrievers selected.")

        unknown = [name for name in normalized if name not in self._by_name]
        if unknown:
            raise ValueError(
                "Unknown retriever(s): "
                + ", ".join(sorted(unknown))
                + f". Available: {', '.join(self.names())}"
            )

        return [self._by_name[name] for name in normalized]
