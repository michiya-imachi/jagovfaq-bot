from typing import Any, Dict, List, TypedDict


class MetaItem(TypedDict):
    id: int
    question: str
    answer: str
    url: str


class GraphState(TypedDict, total=False):
    original_user_query: str
    user_query: str
    search_query: str

    run_bm25: bool
    run_vec: bool
    retrieval_plan_reason: str

    bm25_retrieved: List[Dict[str, Any]]
    vec_retrieved: List[Dict[str, Any]]

    rrf_ranked_all: List[Dict[str, Any]]
    retrieved: List[Dict[str, Any]]

    answer: str
    citations: List[Dict[str, str]]

    need_clarification: bool
    clarifying_question: str
    clarifying_answer: str
    clarifications: List[Dict[str, str]]

    # Routing: destination node name ("followup_question" | "answer" | "fallback")
    next_node: str

    turn_count: int
    max_turns: int

    bm25_count: int
    vec_count: int
    vec_pass_count: int
