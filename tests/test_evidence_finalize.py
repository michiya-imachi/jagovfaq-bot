import unittest

from app.nodes.evidence_finalize import node_evidence_finalize


class EvidenceFinalizeTests(unittest.TestCase):
    def test_rules_high_branch(self):
        run = node_evidence_finalize()
        out = run(
            {
                "run_evidence_llm": False,
                "evidence_rules_level": "high",
                "evidence_rules_reason": "top1_passed_and_multi",
            }
        )

        self.assertEqual(out["local_evidence_level"], "high")
        self.assertEqual(
            out["local_evidence_reason"], "rules_strict_high:top1_passed_and_multi"
        )
        self.assertEqual(out["evidence_action_hint"], "answer")
        self.assertFalse(out["web_needed"])

    def test_llm_success_branch(self):
        run = node_evidence_finalize()
        out = run(
            {
                "run_evidence_llm": True,
                "evidence_rules_level": "low",
                "evidence_llm_level": "low",
                "evidence_llm_action": "web",
                "evidence_llm_reason": "need_web_confirmation",
                "evidence_llm_error": "",
            }
        )

        self.assertEqual(out["local_evidence_level"], "low")
        self.assertEqual(out["local_evidence_reason"], "llm:need_web_confirmation")
        self.assertEqual(out["evidence_action_hint"], "web")
        self.assertTrue(out["web_needed"])

    def test_llm_failure_falls_back_to_rules(self):
        run = node_evidence_finalize()
        out = run(
            {
                "run_evidence_llm": True,
                "evidence_rules_level": "none",
                "evidence_rules_reason": "no_candidates",
                "evidence_llm_error": "JSONDecodeError: bad payload",
            }
        )

        self.assertEqual(out["local_evidence_level"], "none")
        self.assertIn("llm_failed_fallback_rules:", out["local_evidence_reason"])
        self.assertEqual(out["evidence_action_hint"], "followup")
        self.assertTrue(out["web_needed"])


if __name__ == "__main__":
    unittest.main()
