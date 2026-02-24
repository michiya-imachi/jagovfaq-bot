import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.core.types import MetaItem
from app.core.text_utils import simple_tokenize


def load_meta(meta_path: Path) -> Dict[int, MetaItem]:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.jsonl not found: {meta_path}")
    items: Dict[int, MetaItem] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = int(obj["id"])
            items[rid] = {
                "id": rid,
                "question": str(obj["question"]),
                "answer": str(obj["answer"]),
                "url": str(obj.get("url", "")),
            }
    return items


@dataclass
class IndexedStore:
    meta_by_id: Dict[int, MetaItem]
    bm25: Any
    faiss_index: Any
    id_map: List[int]
    embeddings: Optional[OpenAIEmbeddings]
    id_to_faiss_idx: Dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        id_to_idx: Dict[int, int] = {}
        for local_idx, rec_id in enumerate(self.id_map):
            rid = int(rec_id)
            if rid in id_to_idx:
                raise ValueError(f"duplicate_faiss_id_map:{rid}")
            id_to_idx[rid] = int(local_idx)
        self.id_to_faiss_idx = id_to_idx

    @classmethod
    def load(cls, index_dir: Path, embeddings: OpenAIEmbeddings) -> "IndexedStore":
        meta_by_id = load_meta(index_dir / "meta.jsonl")

        with (index_dir / "bm25.pkl").open("rb") as f:
            bm25_obj = pickle.load(f)
        bm25 = bm25_obj["bm25"]

        faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

        with (index_dir / "faiss_map.pkl").open("rb") as f:
            m = pickle.load(f)
        id_map = m["id_map"]

        return cls(
            meta_by_id=meta_by_id,
            bm25=bm25,
            faiss_index=faiss_index,
            id_map=id_map,
            embeddings=embeddings,
        )

    def bm25_search(self, query: str, top_n: int) -> Dict[int, float]:
        if self.bm25 is None:
            raise RuntimeError("BM25 is not configured.")
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = np.argsort(scores)[::-1][:top_n]
        return {int(i): float(scores[int(i)]) for i in ranked}

    def vec_search(self, query: str, top_n: int) -> Dict[int, float]:
        if self.embeddings is None:
            raise RuntimeError("Embeddings are not configured.")
        if self.faiss_index is None:
            raise RuntimeError("FAISS index is not configured.")

        vec = self.embeddings.embed_query(query)
        v = np.array([vec], dtype=np.float32)
        faiss.normalize_L2(v)
        scores, idxs = self.faiss_index.search(v, top_n)

        res: Dict[int, float] = {}
        for score, local_idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if local_idx < 0:
                continue
            rec_id = self.id_map[local_idx]
            res[int(rec_id)] = float(score)
        return res

    def _embed_query_normalized(self, query: str) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Embeddings are not configured.")

        vec = self.embeddings.embed_query(query)
        q = np.array([vec], dtype=np.float32)
        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError("query_embedding_shape_invalid")
        faiss.normalize_L2(q)
        return q[0]

    def _faiss_local_idx(self, rec_id: int) -> int:
        rid = int(rec_id)
        if rid not in self.id_to_faiss_idx:
            raise ValueError(f"faiss_id_reverse_lookup_failed:{rid}")
        return int(self.id_to_faiss_idx[rid])

    def query_doc_similarities(
        self,
        query: str,
        rec_ids: List[int],
    ) -> Dict[int, float]:
        if self.faiss_index is None:
            raise RuntimeError("FAISS index is not configured.")

        q_vec = self._embed_query_normalized(query)
        sims: Dict[int, float] = {}
        for rec_id in rec_ids:
            rid = int(rec_id)
            local_idx = self._faiss_local_idx(rid)
            try:
                reconstructed = self.faiss_index.reconstruct(local_idx)
            except Exception as e:
                raise ValueError(f"faiss_reconstruct_failed:{rid}:{e}") from e

            doc_vec = np.array(reconstructed, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(doc_vec))
            if norm <= 0.0:
                raise ValueError(f"faiss_doc_vector_invalid_norm:{rid}")
            doc_vec = doc_vec / norm

            sim = float(np.dot(q_vec, doc_vec))
            sims[rid] = sim
        return sims
