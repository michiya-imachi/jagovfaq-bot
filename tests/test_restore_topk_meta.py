import unittest

from app.nodes.restore_topk_meta import node_restore_topk_meta


class _DummyStore:
    def __init__(self):
        self.meta_by_id = {
            1: {"id": 1, "question": "Q1", "answer": "A1", "url": "u1"},
            2: {"id": 2, "question": "Q2", "answer": "A2", "url": "u2"},
        }


class RestoreTopkMetaTests(unittest.TestCase):
    def test_restore_item_and_clear_intermediate_keys(self):
        store = _DummyStore()
        run = node_restore_topk_meta(store=store)

        out = run(
            {
                "retrieved": [
                    {"id": 1, "rrf_score": 0.9},
                    {"id": 2, "rrf_score": 0.8},
                ],
                "bm25_retrieved": [{"id": 1}],
                "vec_retrieved": [{"id": 2}],
                "merged_candidates_all": [{"id": 3}],
            }
        )

        self.assertEqual(len(out["retrieved"]), 2)
        self.assertEqual(out["retrieved"][0]["item"]["question"], "Q1")
        self.assertEqual(out["retrieved"][1]["item"]["url"], "u2")
        self.assertEqual(out["bm25_retrieved"], [])
        self.assertEqual(out["vec_retrieved"], [])
        self.assertEqual(out["merged_candidates_all"], [])

    def test_raise_when_meta_missing(self):
        store = _DummyStore()
        run = node_restore_topk_meta(store=store)
        with self.assertRaises(ValueError):
            run({"retrieved": [{"id": 999}]})


if __name__ == "__main__":
    unittest.main()
