import unittest
from unittest.mock import patch

from app.nodes.evidence import node_evidence_assess


class EvidenceAssessTests(unittest.TestCase):
    def test_none_when_no_candidates(self):
        run = node_evidence_assess(candidate_source="merged_candidates_all")
        out = run({"merged_candidates_all": [], "turn_count": 1})
        self.assertEqual(out["local_evidence_level"], "none")
        self.assertEqual(out["local_evidence_reason"], "no_candidates")
        self.assertTrue(out["web_needed"])

    def test_high_when_passed_any_found(self):
        run = node_evidence_assess(candidate_source="merged_candidates_all")
        out = run(
            {
                "merged_candidates_all": [
                    {"id": 1, "passed_any": True, "has_multiple_sources": False}
                ],
                "turn_count": 1,
            }
        )
        self.assertEqual(out["local_evidence_level"], "high")
        self.assertFalse(out["web_needed"])

    def test_low_when_no_passed_or_multi(self):
        run = node_evidence_assess(candidate_source="merged_candidates_all")
        out = run(
            {
                "merged_candidates_all": [
                    {"id": 1, "passed_any": False, "has_multiple_sources": False}
                ],
                "turn_count": 1,
            }
        )
        self.assertEqual(out["local_evidence_level"], "low")
        self.assertTrue(out["web_needed"])

    def test_turn_zero_does_not_require_web(self):
        run = node_evidence_assess(candidate_source="merged_candidates_all")
        out = run(
            {
                "merged_candidates_all": [
                    {"id": 1, "passed_any": False, "has_multiple_sources": False}
                ],
                "turn_count": 0,
            }
        )
        self.assertEqual(out["local_evidence_level"], "low")
        self.assertFalse(out["web_needed"])

    def test_topn_is_respected(self):
        run = node_evidence_assess(candidate_source="merged_candidates_all")
        with patch.dict("os.environ", {"EVIDENCE_TOPN": "1"}):
            out = run(
                {
                    "merged_candidates_all": [
                        {"id": 1, "passed_any": False, "has_multiple_sources": False},
                        {"id": 2, "passed_any": True, "has_multiple_sources": False},
                    ],
                    "turn_count": 1,
                }
            )
        self.assertEqual(out["local_evidence_level"], "low")
        self.assertTrue(out["web_needed"])


if __name__ == "__main__":
    unittest.main()
