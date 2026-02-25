from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict


class RetrieverHit(TypedDict):
    id: int
    retriever: str
    raw_score: Optional[float]
    rank: Optional[int]
    passed: Optional[bool]
    features: Dict[str, Any]


# Backward-compatible alias for a staged rename.
RetrievedCandidate = RetrieverHit
