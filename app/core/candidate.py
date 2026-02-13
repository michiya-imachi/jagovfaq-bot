from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

from app.core.types import MetaItem


class RetrievedCandidate(TypedDict):
    id: int
    item: MetaItem
    retriever: str
    raw_score: Optional[float]
    rank: Optional[int]
    passed: Optional[bool]
    features: Dict[str, Any]
