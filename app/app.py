import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from langgraph.types import Command

from app.core.config import load_env_files, resolve_index_dir
from app.core.llm import make_embeddings, make_llm
from app.core.logging import setup_logging
from app.core.retriever import (
    BM25Retriever,
    RetrieverRegistry,
    VectorRetriever,
    normalize_retriever_names,
    parse_retriever_names,
)
from app.core.store import IndexedStore
from app.core.types import GraphState
from app.graph.builder import build_graph, build_graph_for_export
from app.graph.export import export_graph_artifacts
from app.prompts.loader import PromptLoader


LOG_LEVEL_CHOICES = ["debug", "info", "warning", "error", "critical"]
DEFAULT_RETRIEVERS = ["bm25", "vec"]
logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JaGov FAQ Bot - Indexed (BM25+FAISS) + Streaming Answer"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level for debug/error logs (default: info).",
    )
    parser.add_argument(
        "--export-graph",
        action="store_true",
        help="Export the LangGraph structure as Mermaid (.mmd) and PNG (.png), then exit.",
    )
    parser.add_argument(
        "--graph-out-dir",
        default="out",
        help="Output directory for graph artifacts when --export-graph is set (default: out).",
    )
    parser.add_argument(
        "--retrievers",
        default=None,
        help="Comma-separated retriever names. Example: bm25,vec",
    )
    return parser.parse_args(argv)


def resolve_retriever_selection(
    cli_value: Optional[str],
    env_value: Optional[str],
    default_names: Sequence[str] = DEFAULT_RETRIEVERS,
) -> Tuple[List[str], str]:
    defaults = normalize_retriever_names(default_names)
    if not defaults:
        raise ValueError("default retriever list must not be empty.")

    if cli_value is not None:
        names = normalize_retriever_names(parse_retriever_names(cli_value))
        if not names:
            raise ValueError("No retrievers selected from --retrievers.")
        return names, "cli"

    if env_value is not None and str(env_value).strip():
        names = normalize_retriever_names(parse_retriever_names(env_value))
        if not names:
            raise ValueError("No retrievers selected from RETRIEVERS.")
        return names, "env"

    return list(defaults), "default"


def _extract_interrupt_payload(result: Any) -> Optional[Any]:
    if not isinstance(result, dict):
        return None
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None

    first = interrupts
    if isinstance(interrupts, (list, tuple)) and len(interrupts) > 0:
        first = interrupts[0]

    if hasattr(first, "value"):
        return getattr(first, "value")

    if isinstance(first, dict) and "value" in first:
        return first.get("value")

    return first


def _payload_to_question(payload: Any) -> str:
    if isinstance(payload, dict):
        q = payload.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip()
        return str(payload)
    if isinstance(payload, str):
        return payload.strip()
    return str(payload)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    load_env_files()

    if args.export_graph:
        prompts = PromptLoader.load_default()
        graph_app = build_graph_for_export(prompts)

        out_dir = Path(args.graph_out_dir)
        mmd_path, png_path = export_graph_artifacts(graph_app, out_dir=out_dir)

        print("Graph exported:")
        print(f"- Mermaid: {mmd_path}")
        if png_path is not None:
            print(f"- PNG: {png_path}")
        else:
            print("- PNG: (failed)")
        return

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. Put it in .env.")
        sys.exit(1)

    index_dir = resolve_index_dir()
    if not index_dir.exists():
        logger.error("index dir not found: %s", index_dir)
        logger.error(
            "If you haven't built the index here, run: uv run python -m app.indexer.indexer"
        )
        sys.exit(1)

    prompts = PromptLoader.load_default()
    embeddings = make_embeddings()
    store = IndexedStore.load(index_dir=index_dir, embeddings=embeddings)

    llm = make_llm()
    retriever_registry = RetrieverRegistry(
        [
            BM25Retriever(store),
            VectorRetriever(store),
        ]
    )
    try:
        selected_retrievers, selected_source = resolve_retriever_selection(
            cli_value=args.retrievers,
            env_value=os.getenv("RETRIEVERS"),
            default_names=DEFAULT_RETRIEVERS,
        )
        # Validate selected names once at startup.
        retriever_registry.select(selected_retrievers)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    graph_app = build_graph(
        llm=llm,
        prompts=prompts,
        retriever_registry=retriever_registry,
        default_retrievers=selected_retrievers,
    )

    print("JaGov FAQ Bot - Indexed (BM25+FAISS) + Streaming Answer")
    print(f"Index: {index_dir}")
    print(f"Prompts: {prompts.prompt_dir}")
    print(f"Retrievers ({selected_source}): {','.join(selected_retrievers)}")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("> ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        state: GraphState = {
            "original_user_query": user_query,
            "search_query": user_query,
            "followup_question": "",
            "followup_answer": "",
            "requested_retrievers": list(selected_retrievers),
            "turn_count": 0,
            "max_turns": 2,
        }

        result: Any = graph_app.invoke(state, config=thread_config)

        while True:
            payload = _extract_interrupt_payload(result)
            if payload is None:
                break

            question = _payload_to_question(payload)
            print(f"追加質問: {question}")
            extra = input(">> ").strip()

            if not extra:
                print("（空入力のため終了します）\n")
                break
            if extra.lower() in {"exit", "quit"}:
                print("（終了します）\n")
                return

            result = graph_app.invoke(Command(resume=extra), config=thread_config)


if __name__ == "__main__":
    main()
