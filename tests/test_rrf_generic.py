import unittest

from app.nodes.rrf import node_rrf_rank


class RrfGenericTests(unittest.TestCase):
    def test_rrf_merges_variable_sources(self):
        run = node_rrf_rank()
        state = {
            "search_query": "foo",
            "active_retrievers": ["bm25", "vec", "rule"],
            "retrieval_plan_reason": "test",
            "retrieval_counts": {"bm25": 2, "vec": 2, "rule": 1},
            "retrieval_results_by_source": {
                "bm25": [
                    {
                        "id": 1,
                        "item": {"id": 1, "question": "Q1", "answer": "A1", "url": "u1"},
                        "retriever": "bm25",
                        "raw_score": 10.0,
                        "rank": 1,
                        "passed": None,
                        "features": {},
                    },
                    {
                        "id": 2,
                        "item": {"id": 2, "question": "Q2", "answer": "A2", "url": "u2"},
                        "retriever": "bm25",
                        "raw_score": 9.0,
                        "rank": 2,
                        "passed": None,
                        "features": {},
                    },
                ],
                "vec": [
                    {
                        "id": 1,
                        "item": {"id": 1, "question": "Q1", "answer": "A1", "url": "u1"},
                        "retriever": "vec",
                        "raw_score": 0.9,
                        "rank": 1,
                        "passed": True,
                        "features": {},
                    },
                    {
                        "id": 3,
                        "item": {"id": 3, "question": "Q3", "answer": "A3", "url": "u3"},
                        "retriever": "vec",
                        "raw_score": 0.7,
                        "rank": 2,
                        "passed": True,
                        "features": {},
                    },
                ],
                "rule": [
                    {
                        "id": 3,
                        "item": {"id": 3, "question": "Q3", "answer": "A3", "url": "u3"},
                        "retriever": "rule",
                        "raw_score": 1.0,
                        "rank": 1,
                        "passed": True,
                        "features": {},
                    }
                ],
            },
        }

        out = run(state)
        merged = out["merged_candidates_all"]
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["id"], 1)
        self.assertTrue(merged[0]["has_multiple_sources"])
        self.assertTrue(merged[0]["passed_any"])
        self.assertIn("bm25", merged[0]["source_contrib"])
        self.assertIn("vec", merged[0]["source_contrib"])


if __name__ == "__main__":
    unittest.main()
