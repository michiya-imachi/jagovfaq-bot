import unittest

from app.nodes.routing import make_decide_next_action


class RoutingTests(unittest.TestCase):
    def test_turn_gte_2_forces_web_permission_once_when_not_high(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "low",
                "web_needed": False,
                "turn_count": 2,
                "max_turns": 2,
                "hitl_permission_web_asked": False,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "hitl_permission_web")
        self.assertEqual(out["next_node_reason"], "force_web_check_multi_turn")

    def test_turn_gte_2_does_not_reask_after_permission_asked(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "high",
                "web_needed": False,
                "turn_count": 2,
                "max_turns": 2,
                "hitl_permission_web_asked": True,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "answer")
        self.assertEqual(out["next_node_reason"], "high_evidence_local_answer")

    def test_turn_gte_2_does_not_reask_after_web_attempted(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "high",
                "web_needed": False,
                "turn_count": 2,
                "max_turns": 2,
                "hitl_permission_web_asked": False,
                "web_search_attempted": True,
            }
        )
        self.assertEqual(out["next_node"], "answer")
        self.assertEqual(out["next_node_reason"], "high_evidence_local_answer")

    def test_web_needed_goes_web_permission(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "low",
                "web_needed": True,
                "turn_count": 1,
                "max_turns": 2,
                "hitl_permission_web_asked": False,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "hitl_permission_web")
        self.assertEqual(out["next_node_reason"], "need_web_permission")

    def test_web_needed_does_not_reask_when_already_asked(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "low",
                "web_needed": True,
                "turn_count": 1,
                "max_turns": 2,
                "hitl_permission_web_asked": True,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "followup_question")
        self.assertEqual(out["next_node_reason"], "need_followup")

    def test_high_evidence_goes_answer_without_web_permission(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "high",
                "web_needed": False,
                "turn_count": 0,
                "max_turns": 2,
                "hitl_permission_web_asked": False,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "answer")
        self.assertEqual(out["next_node_reason"], "high_evidence_local_answer")

    def test_high_evidence_skips_web_permission_even_when_turn_gte_2(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "high",
                "web_needed": False,
                "turn_count": 2,
                "max_turns": 2,
                "hitl_permission_web_asked": False,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "answer")
        self.assertEqual(out["next_node_reason"], "high_evidence_local_answer")

    def test_low_evidence_before_limit_goes_followup(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "low",
                "web_needed": False,
                "turn_count": 0,
                "max_turns": 2,
                "hitl_permission_web_asked": False,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "followup_question")
        self.assertEqual(out["next_node_reason"], "need_followup")

    def test_low_evidence_without_web_needed_flag_goes_web_permission_on_first_turn(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "low",
                # web_needed intentionally omitted to test fallback inference.
                "turn_count": 0,
                "max_turns": 2,
                "hitl_permission_web_asked": False,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "hitl_permission_web")
        self.assertEqual(out["next_node_reason"], "need_web_permission")

    def test_low_evidence_at_limit_goes_fallback(self):
        run = make_decide_next_action(top_n=10, candidate_source="merged_candidates_all")
        out = run(
            {
                "local_evidence_level": "low",
                "web_needed": False,
                "turn_count": 0,
                "max_turns": 0,
                "hitl_permission_web_asked": False,
                "web_search_attempted": False,
            }
        )
        self.assertEqual(out["next_node"], "fallback")
        self.assertEqual(out["next_node_reason"], "max_turns_reached")


if __name__ == "__main__":
    unittest.main()
