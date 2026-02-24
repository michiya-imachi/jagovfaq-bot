import unittest

from app.nodes.evidence_router import node_evidence_router


class EvidenceRouterTests(unittest.TestCase):
    def test_high_skips_llm(self):
        run = node_evidence_router()
        out = run({"evidence_rules_level": "high"})
        self.assertFalse(out["run_evidence_llm"])
        self.assertEqual(out["evidence_route_reason"], "rules_high_skip_llm")

    def test_low_or_none_runs_llm(self):
        run = node_evidence_router()
        out_low = run({"evidence_rules_level": "low"})
        out_none = run({"evidence_rules_level": "none"})

        self.assertTrue(out_low["run_evidence_llm"])
        self.assertEqual(out_low["evidence_route_reason"], "rules_not_high_run_llm")
        self.assertTrue(out_none["run_evidence_llm"])
        self.assertEqual(out_none["evidence_route_reason"], "rules_not_high_run_llm")


if __name__ == "__main__":
    unittest.main()
