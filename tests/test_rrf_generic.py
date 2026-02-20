import unittest

from app.nodes.rrf import node_rrf_rank


class RrfGenericTests(unittest.TestCase):
    def test_rrf_merges_split_bm25_and_vec_inputs(self):
        run = node_rrf_rank()
        state = {
            "search_query": "foo",
            "active_retrievers": ["bm25", "vec"],
            "retrieval_plan_reason": "test",
            "bm25_count": 2,
            "vec_count": 2,
            "vec_pass_count": 2,
            "bm25_retrieved": [
                {
                    "id": 1,
                    "item": {"id": 1, "question": "Q1", "answer": "A1", "url": "u1"},
                    "bm25_raw": 10.0,
                    "bm25_rank": 1,
                },
                {
                    "id": 2,
                    "item": {"id": 2, "question": "Q2", "answer": "A2", "url": "u2"},
                    "bm25_raw": 9.0,
                    "bm25_rank": 2,
                },
            ],
            "vec_retrieved": [
                {
                    "id": 1,
                    "item": {"id": 1, "question": "Q1", "answer": "A1", "url": "u1"},
                    "vec_raw": 0.9,
                    "vec_rank": 1,
                    "vec_pass_threshold": True,
                },
                {
                    "id": 3,
                    "item": {"id": 3, "question": "Q3", "answer": "A3", "url": "u3"},
                    "vec_raw": 0.7,
                    "vec_rank": 2,
                    "vec_pass_threshold": True,
                },
            ],
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
