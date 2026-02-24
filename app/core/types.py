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

    # Follow-up question + HITL answer (single-turn buffer)
    followup_question: str
    hitl_input_answer: str

    # Retrieval planning
    requested_retrievers: List[str]
    active_retrievers: List[str]
    run_bm25: bool
    run_vec: bool
    retrieval_plan_reason: str

    # Raw retrieval outputs (ID-centric only; no item payload)
    # bm25_retrieved row: id, bm25_rank, bm25_raw
    # vec_retrieved row: id, vec_rank, vec_raw, vec_pass_threshold
    bm25_retrieved: List[Dict[str, Any]]
    vec_retrieved: List[Dict[str, Any]]

    # Retrieval outputs by source name (legacy optional)
    retrieval_results_by_source: Dict[str, List[Dict[str, Any]]]

    # All ranked candidates after RRF merge (pre TopK, still ID-centric)
    merged_candidates_all: List[Dict[str, Any]]

    # Final candidates after TopK filter.
    # - right after topk_filter: ID-centric
    # - after restore_topk_meta: item attached for downstream usage
    retrieved: List[Dict[str, Any]]

    answer: str
    citations: List[Dict[str, str]]

    # Evidence assessment
    evidence_rules_level: str  # "high" | "low" | "none"
    evidence_rules_reason: str
    run_evidence_llm: bool
    evidence_route_reason: str
    evidence_llm_level: str  # "high" | "low" | "none"
    evidence_llm_action: str  # "answer" | "followup" | "web"
    evidence_llm_reason: str
    evidence_llm_error: str
    evidence_action_hint: str  # "answer" | "followup" | "web"
    local_evidence_level: str  # "high" | "low" | "none"
    local_evidence_reason: str
    web_needed: bool

    # HITL web permission
    hitl_permission_web_question: str
    hitl_permission_web_answer: str
    hitl_permission_web_asked: bool
    web_search_allowed: bool
    web_search_declined: bool

    # Web search
    web_query: str
    web_results: List[Dict[str, Any]]
    web_search_error: str
    web_search_attempted: bool

    # Routing
    need_followup: bool
    next_node: str  # "followup_question" | "hitl_permission_web" | "answer" | "fallback"
    next_node_reason: str  # e.g. "hint_answer" | "hint_web_need_permission" | "hint_followup"

    # Turn control
    turn_count: int
    max_turns: int

    # Retrieval diagnostics (optional)
    bm25_count: int
    vec_count: int
    vec_pass_count: int
    retrieval_counts: Dict[str, int]
