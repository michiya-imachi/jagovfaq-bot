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
    requested_retrievers: List[str]
    active_retrievers: List[str]
    run_bm25: bool
    run_vec: bool
    retrieval_plan_reason: str

    # Raw retrieval outputs (optional)
    bm25_retrieved: List[Dict[str, Any]]
    vec_retrieved: List[Dict[str, Any]]

    # Retrieval outputs by source name (legacy optional)
    retrieval_results_by_source: Dict[str, List[Dict[str, Any]]]

    # All ranked candidates after RRF merge (pre TopK)
    merged_candidates_all: List[Dict[str, Any]]

    # Final candidates after TopK filter (used downstream by ask/answer)
    retrieved: List[Dict[str, Any]]

    answer: str
    citations: List[Dict[str, str]]

    # Evidence assessment
    local_evidence_level: str  # "high" | "low" | "none"
    local_evidence_reason: str
    web_needed: bool

    # Web permission
    web_permission_question: str
    web_permission_answer: str
    web_permission_asked: bool
    web_search_allowed: bool
    web_search_declined: bool

    # Web search
    web_query: str
    web_results: List[Dict[str, Any]]
    web_search_error: str
    web_search_attempted: bool

    # Routing
    need_followup: bool
    next_node: str  # "followup_question" | "web_permission" | "answer" | "fallback"
    next_node_reason: str  # e.g. "high_evidence_local_answer" | "force_web_check_multi_turn" | "need_followup"

    # Turn control
    turn_count: int
    max_turns: int

    # Retrieval diagnostics (optional)
    bm25_count: int
    vec_count: int
    vec_pass_count: int
    retrieval_counts: Dict[str, int]
