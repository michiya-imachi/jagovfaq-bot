import unittest

from app.core.retriever import RetrieverRegistry
from app.nodes.retrieval import node_retrieve_bm25, node_retrieve_vec_threshold


class DummyRetriever:
    def __init__(self, name: str, rows):
        self.name = name
        self._rows = rows

    def retrieve(self, query: str, state):
        return list(self._rows)


class RetrieveSplitNodeTests(unittest.TestCase):
    def test_bm25_collects_rows_and_count(self):
        registry = RetrieverRegistry(
            [
                DummyRetriever(
                    "bm25",
                    [
                        {
                            "id": 1,
                            "item": {
                                "id": 1,
                                "question": "Q1",
                                "answer": "A1",
                                "url": "u1",
                            },
                            "retriever": "bm25",
                            "raw_score": 1.2,
                            "rank": 1,
                            "passed": None,
                            "features": {},
                        }
                    ],
                ),
                DummyRetriever("vec", []),
            ]
        )
        run = node_retrieve_bm25(registry)

        out = run({"search_query": "test", "run_bm25": True})

        self.assertEqual(out["bm25_count"], 1)
        self.assertEqual(len(out["bm25_retrieved"]), 1)
        self.assertEqual(out["bm25_retrieved"][0]["id"], 1)
        self.assertEqual(out["bm25_retrieved"][0]["bm25_rank"], 1)

    def test_bm25_disabled_returns_empty(self):
        registry = RetrieverRegistry([DummyRetriever("bm25", []), DummyRetriever("vec", [])])
        run = node_retrieve_bm25(registry)

        out = run({"search_query": "test", "run_bm25": False})

        self.assertEqual(out, {"bm25_retrieved": [], "bm25_count": 0})

    def test_vec_collects_rows_and_pass_count(self):
        registry = RetrieverRegistry(
            [
                DummyRetriever("bm25", []),
                DummyRetriever(
                    "vec",
                    [
                        {
                            "id": 2,
                            "item": {
                                "id": 2,
                                "question": "Q2",
                                "answer": "A2",
                                "url": "u2",
                            },
                            "retriever": "vec",
                            "raw_score": 0.8,
                            "rank": 1,
                            "passed": True,
                            "features": {},
                        },
                        {
                            "id": 3,
                            "item": {
                                "id": 3,
                                "question": "Q3",
                                "answer": "A3",
                                "url": "u3",
                            },
                            "retriever": "vec",
                            "raw_score": 0.1,
                            "rank": 2,
                            "passed": False,
                            "features": {},
                        },
                    ],
                ),
            ]
        )
        run = node_retrieve_vec_threshold(registry)

        out = run({"search_query": "test", "run_vec": True})

        self.assertEqual(out["vec_count"], 2)
        self.assertEqual(out["vec_pass_count"], 1)
        self.assertEqual(out["vec_retrieved"][0]["vec_pass_threshold"], True)
        self.assertEqual(out["vec_retrieved"][1]["vec_pass_threshold"], False)

    def test_vec_disabled_returns_empty(self):
        registry = RetrieverRegistry([DummyRetriever("bm25", []), DummyRetriever("vec", [])])
        run = node_retrieve_vec_threshold(registry)

        out = run({"search_query": "test", "run_vec": False})

        self.assertEqual(
            out,
            {
                "vec_retrieved": [],
                "vec_count": 0,
                "vec_pass_count": 0,
            },
        )

    def test_missing_retriever_raises_on_factory(self):
        with self.assertRaises(ValueError):
            node_retrieve_bm25(RetrieverRegistry([DummyRetriever("vec", [])]))

        with self.assertRaises(ValueError):
            node_retrieve_vec_threshold(RetrieverRegistry([DummyRetriever("bm25", [])]))


if __name__ == "__main__":
    unittest.main()
