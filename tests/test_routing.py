import unittest

from app.nodes.routing import make_decide_next_action


class RoutingTests(unittest.TestCase):
    def test_no_candidates_goes_followup_before_limit(self):
        run = make_decide_next_action(top_n=1, candidate_source="retrieved")
        out = run(
            {
                "retrieved": [],
                "turn_count": 0,
                "max_turns": 2,
                "active_retrievers": ["bm25", "vec"],
            }
        )
        self.assertEqual(out["next_node"], "followup_question")

    def test_no_candidates_goes_fallback_at_limit(self):
        run = make_decide_next_action(top_n=1, candidate_source="retrieved")
        out = run(
            {
                "retrieved": [],
                "turn_count": 2,
                "max_turns": 2,
                "active_retrievers": ["bm25", "vec"],
            }
        )
        self.assertEqual(out["next_node"], "fallback")

    def test_passed_candidate_goes_answer(self):
        run = make_decide_next_action(top_n=1, candidate_source="retrieved")
        out = run(
            {
                "retrieved": [
                    {
                        "id": 1,
                        "score": 0.5,
                        "passed_any": True,
                        "has_multiple_sources": False,
                    }
                ],
                "turn_count": 0,
                "max_turns": 2,
                "active_retrievers": ["vec"],
            }
        )
        self.assertEqual(out["next_node"], "answer")


if __name__ == "__main__":
    unittest.main()
