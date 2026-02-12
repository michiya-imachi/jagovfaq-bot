import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from app.core.logging import setup_logging
from app.core.text_utils import simple_tokenize


LOG_LEVEL_CHOICES = ["debug", "info", "warning", "error", "critical"]
logger = logging.getLogger(__name__)


@dataclass
class FaqRecord:
    rec_id: int
    question: str
    answer: str
    url: str
    text: str


def normalize_item(obj: Dict[str, Any]) -> Tuple[str, str, str]:
    q = obj.get("Question") or obj.get("question") or obj.get("Q") or ""
    a = obj.get("Answer") or obj.get("answer") or obj.get("A") or ""
    u = obj.get("url") or obj.get("URL") or obj.get("link") or ""
    return str(q).strip(), str(a).strip(), str(u).strip()


def load_faq_file(path: Path) -> List[FaqRecord]:
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {path}")

    records: List[FaqRecord] = []
    rec_id = 0

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q, a, u = normalize_item(obj)
                if not q or not a:
                    continue
                text = f"Q: {q}\nQ: {q}\nA: {a}"
                records.append(
                    FaqRecord(rec_id=rec_id, question=q, answer=a, url=u, text=text)
                )
                rec_id += 1

    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            arr = json.loads(f.read())
        if not isinstance(arr, list):
            raise ValueError("JSON must be a list of records.")
        for obj in arr:
            q, a, u = normalize_item(obj)
            if not q or not a:
                continue
            text = f"Q: {q}\nQ: {q}\nA: {a}"
            records.append(
                FaqRecord(rec_id=rec_id, question=q, answer=a, url=u, text=text)
            )
            rec_id += 1
    else:
        raise ValueError("Supported formats: .jsonl or .json")

    if not records:
        raise ValueError(
            "No valid FAQ items found. Check keys like Question/Answer/url."
        )
    return records


def build_bm25(records: List[FaqRecord]) -> Dict[str, Any]:
    corpus_tokens = [simple_tokenize(r.text) for r in records]
    bm25 = BM25Okapi(corpus_tokens)
    return {"bm25": bm25}


def embed_documents_with_tqdm(
    client: OpenAI,
    model: str,
    texts: List[str],
    batch_size: int,
    timeout_s: float,
    max_retries: int,
) -> np.ndarray:
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    vectors: List[List[float]] = []

    pbar = tqdm(total=total, desc="Embedding", unit="docs", dynamic_ncols=True)
    start = time.perf_counter()

    try:
        for b in range(total_batches):
            i0 = b * batch_size
            i1 = min(total, i0 + batch_size)
            chunk = texts[i0:i1]

            # Show which batch we're on.
            elapsed = time.perf_counter() - start
            pbar.set_postfix_str(f"batch {b+1}/{total_batches} elapsed {elapsed:.0f}s")

            last_err: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    resp = client.embeddings.create(
                        model=model,
                        input=chunk,
                        timeout=timeout_s,
                    )
                    resp_data = sorted(resp.data, key=lambda x: x.index)
                    vectors.extend([d.embedding for d in resp_data])
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    # Print retry info without breaking the progress bar.
                    tqdm.write(
                        f"[Embedding] retry {attempt+1}/{max_retries} failed: {type(e).__name__}: {e}"
                    )
                    if attempt < max_retries:
                        time.sleep(min(2**attempt, 10))
                    else:
                        raise

            if last_err is not None:
                raise last_err

            pbar.update(i1 - i0)

    finally:
        pbar.close()

    mat = np.array(vectors, dtype=np.float32)
    if mat.shape[0] != total:
        raise ValueError(
            f"Embedding count mismatch: expected {total}, got {mat.shape[0]}"
        )
    return mat


def build_faiss_from_embeddings(mat: np.ndarray) -> Tuple[faiss.Index, int]:
    if mat.ndim != 2:
        raise ValueError("Embedding output must be 2D array.")
    dim = int(mat.shape[1])
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    return index, dim


def write_meta(records: List[FaqRecord], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "id": r.rec_id,
                "question": r.question,
                "answer": r.answer,
                "url": r.url,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_manifest(out_path: Path, meta: Dict[str, Any]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JaGov FAQ search index")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level for debug/error logs (default: info).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. Put it in .env.")
        sys.exit(1)

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    faq_path = Path(os.getenv("FAQ_PATH", "data/jagovfaqs.jsonl"))
    if not faq_path.exists():
        faq_path = Path("data/jagovfaqs.json")

    out_dir = Path("data/index")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tunables.
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    timeout_s = float(os.getenv("EMBED_TIMEOUT_S", "60"))
    max_retries = int(os.getenv("EMBED_MAX_RETRIES", "3"))

    print(f"Loading: {faq_path}")
    records = load_faq_file(faq_path)
    print(f"Records: {len(records)}")

    print("Building BM25...")
    bm25_obj = build_bm25(records)
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25_obj, f)

    print(f"Building FAISS (embedding model: {embedding_model})...")
    client = OpenAI()

    texts = [r.text for r in records]
    mat = embed_documents_with_tqdm(
        client=client,
        model=embedding_model,
        texts=texts,
        batch_size=batch_size,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )

    index, dim = build_faiss_from_embeddings(mat)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    id_map = [r.rec_id for r in records]
    with (out_dir / "faiss_map.pkl").open("wb") as f:
        pickle.dump({"id_map": id_map, "dim": dim}, f)

    print("Writing meta.jsonl...")
    write_meta(records, out_dir / "meta.jsonl")

    print("Writing manifest.json...")
    write_manifest(
        out_dir / "manifest.json",
        {
            "faq_path": str(faq_path).replace("\\", "/"),
            "embedding_model": embedding_model,
            "bm25": {"question_boost": 2},
            "faiss": {"index_type": "IndexFlatIP", "normalized": True},
            "record_count": len(records),
            "embedding": {
                "batch_size": batch_size,
                "timeout_s": timeout_s,
                "max_retries": max_retries,
            },
        },
    )

    print("\nDone.")
    print("Outputs:")
    print(f"- {out_dir / 'bm25.pkl'}")
    print(f"- {out_dir / 'faiss.index'}")
    print(f"- {out_dir / 'faiss_map.pkl'}")
    print(f"- {out_dir / 'meta.jsonl'}")
    print(f"- {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
