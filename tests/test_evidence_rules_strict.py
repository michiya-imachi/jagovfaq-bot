import unittest
from unittest.mock import patch

from app.nodes.evidence_rules_strict import node_evidence_rules_strict


class EvidenceRulesStrictTests(unittest.TestCase):
    def test_none_when_no_candidates(self):
        run = node_evidence_rules_strict(candidate_source="merged_candidates_all")
        out = run({"merged_candidates_all": []})
        self.assertEqual(out["evidence_rules_level"], "none")
        self.assertEqual(out["evidence_rules_reason"], "no_candidates")

    def test_high_only_when_top1_passed_and_multi(self):
        run = node_evidence_rules_strict(candidate_source="merged_candidates_all")
        out = run(
            {
                "merged_candidates_all": [
                    {"id": 1, "passed_any": True, "has_multiple_sources": True}
                ]
            }
        )
        self.assertEqual(out["evidence_rules_level"], "high")
        self.assertEqual(out["evidence_rules_reason"], "top1_passed_and_multi")

    def test_low_when_top1_is_not_strong_even_if_lower_rank_is_strong(self):
        run = node_evidence_rules_strict(candidate_source="merged_candidates_all")
        out = run(
            {
                "merged_candidates_all": [
                    {"id": 1, "passed_any": False, "has_multiple_sources": False},
                    {"id": 2, "passed_any": True, "has_multiple_sources": True},
                ]
            }
        )
        self.assertEqual(out["evidence_rules_level"], "low")
        self.assertEqual(out["evidence_rules_reason"], "not_strong_enough")

    def test_topn_is_applied(self):
        run = node_evidence_rules_strict(candidate_source="merged_candidates_all")
        with patch.dict("os.environ", {"EVIDENCE_TOPN": "1"}, clear=False):
            out = run(
                {
                    "merged_candidates_all": [
                        {"id": 1, "passed_any": False, "has_multiple_sources": False},
                        {"id": 2, "passed_any": True, "has_multiple_sources": True},
                    ]
                }
            )
        self.assertEqual(out["evidence_rules_level"], "low")
        self.assertEqual(out["evidence_rules_reason"], "not_strong_enough")


if __name__ == "__main__":
    unittest.main()
