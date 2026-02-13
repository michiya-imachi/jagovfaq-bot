from typing import Any, Dict, List, TypedDict


class MetaItem(TypedDict):
    id: int
    question: str
    answer: str
    url: str


class GraphState(TypedDict, total=False):
    # User query
    original_user_query: str
    search_query: str

    # HITL follow-up (single-turn buffer)
    followup_question: str
    followup_answer: str

    # Retrieval planning
    run_bm25: bool
    run_vec: bool
    retrieval_plan_reason: str

    # Raw retrieval outputs (optional)
    bm25_retrieved: List[Dict[str, Any]]
    vec_retrieved: List[Dict[str, Any]]

    # All ranked candidates after RRF (pre TopK)
    rrf_ranked_all: List[Dict[str, Any]]

    # Final candidates after TopK filter (used downstream by ask/answer)
    retrieved: List[Dict[str, Any]]

    answer: str
    citations: List[Dict[str, str]]

    # Routing
    need_followup: bool
    next_node: str  # "followup_question" | "answer" | "fallback"
    next_node_reason: str  # e.g. "max_turns_reached" | "need_followup" | "enough_evidence"

    # Turn control
    turn_count: int
    max_turns: int

    # Retrieval diagnostics (optional)
    bm25_count: int
    vec_count: int
    vec_pass_count: int
