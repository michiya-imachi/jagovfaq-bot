import unittest

from app.core.retriever import RetrieverRegistry
from app.nodes.retrieval import node_retrieve_all


class DummyRetriever:
    def __init__(self, name: str, rows):
        self.name = name
        self._rows = rows

    def retrieve(self, query: str, state):
        return list(self._rows)


class RetrieveAllNodeTests(unittest.TestCase):
    def test_collects_results_from_multiple_retrievers(self):
        bm25 = DummyRetriever(
            "bm25",
            [
                {
                    "id": 1,
                    "item": {"id": 1, "question": "Q1", "answer": "A1", "url": "u1"},
                    "retriever": "bm25",
                    "raw_score": 1.2,
                    "rank": 1,
                    "passed": None,
                    "features": {},
                }
            ],
        )
        vec = DummyRetriever(
            "vec",
            [
                {
                    "id": 2,
                    "item": {"id": 2, "question": "Q2", "answer": "A2", "url": "u2"},
                    "retriever": "vec",
                    "raw_score": 0.8,
                    "rank": 1,
                    "passed": True,
                    "features": {},
                }
            ],
        )
        registry = RetrieverRegistry([bm25, vec])
        run = node_retrieve_all(registry)

        out = run(
            {
                "search_query": "test",
                "active_retrievers": ["bm25", "vec"],
            }
        )

        self.assertIn("retrieval_results_by_source", out)
        self.assertEqual(len(out["retrieval_results_by_source"]["bm25"]), 1)
        self.assertEqual(len(out["retrieval_results_by_source"]["vec"]), 1)
        self.assertEqual(out["retrieval_counts"], {"bm25": 1, "vec": 1})

    def test_unknown_active_retriever_raises(self):
        registry = RetrieverRegistry([DummyRetriever("bm25", [])])
        run = node_retrieve_all(registry)

        with self.assertRaises(ValueError):
            run(
                {
                    "search_query": "test",
                    "active_retrievers": ["foo"],
                }
            )


if __name__ == "__main__":
    unittest.main()
