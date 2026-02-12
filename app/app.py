import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from app.core.config import load_env_files, resolve_index_dir
from app.core.llm import make_embeddings, make_llm
from app.core.store import IndexedStore
from app.core.types import GraphState
from app.graph.builder import build_graph, build_graph_for_export
from app.graph.export import export_graph_artifacts
from app.prompts.loader import PromptLoader


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JaGov FAQ Bot - Indexed (BM25+FAISS) + Streaming Answer"
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
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
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
        print("ERROR: OPENAI_API_KEY is not set. Put it in .env.", file=sys.stderr)
        sys.exit(1)

    index_dir = resolve_index_dir()
    if not index_dir.exists():
        print(f"ERROR: index dir not found: {index_dir}", file=sys.stderr)
        print(
            "If you haven't built the index here, run: uv run python -m app.indexer.indexer",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts = PromptLoader.load_default()
    embeddings = make_embeddings()
    store = IndexedStore.load(index_dir=index_dir, embeddings=embeddings)

    llm = make_llm()
    graph_app = build_graph(store, llm, prompts)

    print("JaGov FAQ Bot - Indexed (BM25+FAISS) + Streaming Answer")
    print(f"Index: {index_dir}")
    print(f"Prompts: {prompts.prompt_dir}")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("> ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        state: GraphState = {"user_query": user_query, "turn_count": 0, "max_turns": 2}

        while True:
            result: GraphState = graph_app.invoke(state)

            if result.get("need_clarification", False):
                print(f"追加質問: {result.get('clarifying_question', '')}")
                extra = input(">> ").strip()
                if not extra:
                    print("（空入力のため終了します）\n")
                    break
                state = {
                    **state,
                    "user_query": f"{state['user_query']}\n補足: {extra}",
                    "turn_count": int(
                        result.get("turn_count", state.get("turn_count", 0))
                    ),
                }
                continue

            # Answer already streamed.
            break


if __name__ == "__main__":
    main()
