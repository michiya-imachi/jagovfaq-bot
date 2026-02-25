import unittest
from unittest.mock import patch

from app.core.retriever import BM25Retriever, VectorRetriever


class _DummyBm25Store:
    def __init__(self):
        self.calls = []

    def bm25_search(self, query: str, top_n: int):
        self.calls.append((query, top_n))
        return {10: 0.3, 3: 0.9}


class _DummyVecStore:
    def __init__(self):
        self.calls = []

    def vec_search(self, query: str, top_n: int):
        self.calls.append((query, top_n))
        return {20: 0.3, 30: 0.2}


class RetrieverIdCentricTests(unittest.TestCase):
    def test_bm25_retrieve_returns_id_only_rows_without_meta_lookup(self):
        store = _DummyBm25Store()
        retriever = BM25Retriever(store=store)

        with patch.dict("os.environ", {"BM25_TOPK": "5"}, clear=False):
            rows = retriever.retrieve("test-query", {})

        self.assertEqual(store.calls, [("test-query", 5)])
        self.assertEqual([row["id"] for row in rows], [3, 10])
        self.assertEqual([row["rank"] for row in rows], [1, 2])
        self.assertNotIn("item", rows[0])

    def test_vec_retrieve_returns_id_only_rows_without_meta_lookup(self):
        store = _DummyVecStore()
        retriever = VectorRetriever(store=store)

        with patch.dict(
            "os.environ",
            {
                "VEC_SEARCH_TOPN": "5",
                "VEC_THRESHOLD": "0.5",
                "VEC_MAX_KEEP": "2",
                "VEC_FALLBACK_TOPK": "2",
            },
            clear=False,
        ):
            rows = retriever.retrieve("test-query", {})

        self.assertEqual(store.calls, [("test-query", 5)])
        self.assertEqual([row["id"] for row in rows], [20, 30])
        self.assertEqual([row["passed"] for row in rows], [False, False])
        self.assertNotIn("item", rows[0])


if __name__ == "__main__":
    unittest.main()
