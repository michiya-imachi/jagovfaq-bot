import unittest

from app.nodes import evidence_rules_strict
from app.nodes.evidence_rules_strict import node_evidence_rules_strict


class _DummyStore:
    def __init__(self, sim_by_id=None, error=None):
        self.sim_by_id = sim_by_id or {}
        self.error = error
        self.calls = []

    def query_doc_similarities(self, query, rec_ids):
        self.calls.append((query, list(rec_ids)))
        if self.error is not None:
            raise self.error

        out = {}
        for rid in rec_ids:
            if rid not in self.sim_by_id:
                raise ValueError(f"faiss_id_reverse_lookup_failed:{rid}")
            out[rid] = self.sim_by_id[rid]
        return out


class EvidenceRulesStrictTests(unittest.TestCase):
    def test_none_when_no_candidates(self):
        store = _DummyStore()
        run = node_evidence_rules_strict(store=store, candidate_source="retrieved")
        out = run({"retrieved": []})
        self.assertEqual(out["evidence_rules_level"], "none")
        self.assertEqual(out["evidence_rules_reason"], "no_candidates")
        self.assertEqual(store.calls, [])

    def test_high_when_max_sim_is_gte_threshold(self):
        th = float(evidence_rules_strict._VEC_SIM_THRESHOLD)
        store = _DummyStore(sim_by_id={1: th, 2: max(0.0, th - 0.05)})
        run = node_evidence_rules_strict(store=store, candidate_source="retrieved")
        out = run(
            {
                "search_query": "test query",
                "retrieved": [
                    {"id": 1, "sources": ["bm25"], "rrf_score": 0.10},
                    {"id": 2, "sources": ["vec"], "rrf_score": 0.09},
                ]
            }
        )
        self.assertEqual(out["evidence_rules_level"], "high")
        self.assertTrue(out["evidence_rules_reason"].startswith("vecsim_max_ge_"))
        self.assertAlmostEqual(out["retrieved"][0]["sim_score"], store.sim_by_id[1])
        self.assertEqual(out["retrieved"][0]["source"], "bm25")
        self.assertIn("rrf_score", out["retrieved"][0])

    def test_low_when_max_sim_is_lt_threshold(self):
        th = float(evidence_rules_strict._VEC_SIM_THRESHOLD)
        store = _DummyStore(
            sim_by_id={
                1: max(0.0, th - 0.01),
                2: max(0.0, th - 0.20),
            }
        )
        run = node_evidence_rules_strict(store=store, candidate_source="retrieved")
        out = run(
            {
                "search_query": "test query",
                "retrieved": [
                    {"id": 1, "sources": ["bm25"], "rrf_score": 0.10},
                    {"id": 2, "sources": ["vec"], "rrf_score": 0.09},
                ]
            }
        )
        self.assertEqual(out["evidence_rules_level"], "low")
        self.assertTrue(out["evidence_rules_reason"].startswith("vecsim_max_lt_"))
        self.assertAlmostEqual(out["retrieved"][1]["sim_score"], store.sim_by_id[2])
        self.assertEqual(out["retrieved"][1]["source"], "vec")

    def test_raises_when_faiss_reverse_lookup_missing(self):
        store = _DummyStore(sim_by_id={1: 0.59})
        run = node_evidence_rules_strict(store=store, candidate_source="retrieved")
        with self.assertRaises(ValueError):
            run({"search_query": "test query", "retrieved": [{"id": 999}]})


if __name__ == "__main__":
    unittest.main()
