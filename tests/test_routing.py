import unittest

from app.nodes.routing import make_response_router


class RoutingTests(unittest.TestCase):
    def test_hint_answer_goes_answer(self):
        run = make_response_router()
        out = run(
            {
                "evidence_action_hint": "answer",
                "turn_count": 0,
                "max_turns": 2,
            }
        )
        self.assertEqual(out["next_node"], "answer")
        self.assertEqual(out["next_node_reason"], "hint_answer")
        self.assertFalse(out["need_followup"])

    def test_hint_web_goes_permission_when_can_ask(self):
        run = make_response_router()
        out = run(
            {
                "evidence_action_hint": "web",
                "turn_count": 1,
                "max_turns": 2,
                "hitl_permission_web_asked": False,
                "web_search_attempted": False,
                "web_search_declined": False,
            }
        )
        self.assertEqual(out["next_node"], "hitl_permission_web")
        self.assertEqual(out["next_node_reason"], "hint_web_need_permission")
        self.assertFalse(out["need_followup"])

    def test_hint_web_goes_answer_when_declined(self):
        run = make_response_router()
        out = run(
            {
                "evidence_action_hint": "web",
                "turn_count": 1,
                "max_turns": 2,
                "hitl_permission_web_asked": True,
                "web_search_attempted": False,
                "web_search_declined": True,
            }
        )
        self.assertEqual(out["next_node"], "answer")
        self.assertEqual(out["next_node_reason"], "hint_web_cannot_ask_answer")
        self.assertFalse(out["need_followup"])

    def test_hint_followup_goes_followup_before_limit(self):
        run = make_response_router()
        out = run(
            {
                "evidence_action_hint": "followup",
                "turn_count": 1,
                "max_turns": 2,
            }
        )
        self.assertEqual(out["next_node"], "followup_question")
        self.assertEqual(out["next_node_reason"], "hint_followup")
        self.assertTrue(out["need_followup"])

    def test_hint_followup_goes_fallback_on_limit(self):
        run = make_response_router()
        out = run(
            {
                "evidence_action_hint": "followup",
                "turn_count": 2,
                "max_turns": 2,
            }
        )
        self.assertEqual(out["next_node"], "fallback")
        self.assertEqual(out["next_node_reason"], "hint_followup_max_turns")
        self.assertTrue(out["need_followup"])

    def test_invalid_or_missing_hint_falls_back_to_followup_policy(self):
        run = make_response_router()
        out_invalid = run(
            {
                "evidence_action_hint": "unknown",
                "turn_count": 0,
                "max_turns": 2,
            }
        )
        out_missing = run(
            {
                "turn_count": 0,
                "max_turns": 2,
            }
        )

        self.assertEqual(out_invalid["next_node"], "followup_question")
        self.assertEqual(out_invalid["next_node_reason"], "hint_invalid_followup")
        self.assertEqual(out_missing["next_node"], "followup_question")
        self.assertEqual(out_missing["next_node_reason"], "hint_invalid_followup")


if __name__ == "__main__":
    unittest.main()
