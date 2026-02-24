import unittest

from app.nodes.evidence_llm import node_evidence_llm_judge


class _DummyPrompts:
    def __init__(self):
        self.calls = []

    def render_pair(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        return {"system": "sys", "user": "user"}


class _DummyMessage:
    def __init__(self, content):
        self.content = content


class _DummyLLM:
    def __init__(self, content: str):
        self._content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _DummyMessage(self._content)


class EvidenceLlmJudgeTests(unittest.TestCase):
    def test_success_parses_json_and_truncates_answer(self):
        long_answer = "A" * 900
        llm = _DummyLLM(
            '{"level":"high","action":"answer","reason":"direct_match"}'
        )
        prompts = _DummyPrompts()
        run = node_evidence_llm_judge(llm=llm, prompts=prompts)

        out = run(
            {
                "run_evidence_llm": True,
                "search_query": "test query",
                "retrieved": [
                    {
                        "final_rank": 1,
                        "rrf_score": 0.5,
                        "sources": ["bm25", "vec"],
                        "passed_any": True,
                        "has_multiple_sources": True,
                        "item": {
                            "question": "Q",
                            "answer": long_answer,
                            "url": "https://example.com",
                        },
                    }
                ],
            }
        )

        self.assertEqual(out["evidence_llm_level"], "high")
        self.assertEqual(out["evidence_llm_action"], "answer")
        self.assertEqual(out["evidence_llm_reason"], "direct_match")
        self.assertEqual(out["evidence_llm_error"], "")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(prompts.calls[-1][0], "evidence_judge")

        candidates_text = prompts.calls[-1][1]["candidates_text"]
        self.assertIn("question: Q", candidates_text)
        self.assertNotIn("A" * 850, candidates_text)

    def test_invalid_json_sets_error(self):
        llm = _DummyLLM("not-json")
        prompts = _DummyPrompts()
        run = node_evidence_llm_judge(llm=llm, prompts=prompts)

        out = run({"run_evidence_llm": True, "retrieved": []})

        self.assertEqual(out["evidence_llm_level"], "")
        self.assertEqual(out["evidence_llm_action"], "")
        self.assertEqual(out["evidence_llm_reason"], "")
        self.assertIn("JSONDecodeError", out["evidence_llm_error"])

    def test_invalid_enum_sets_error(self):
        llm = _DummyLLM(
            '{"level":"great","action":"answer","reason":"invalid_level"}'
        )
        prompts = _DummyPrompts()
        run = node_evidence_llm_judge(llm=llm, prompts=prompts)

        out = run({"run_evidence_llm": True, "retrieved": []})

        self.assertEqual(out["evidence_llm_level"], "")
        self.assertIn("ValueError", out["evidence_llm_error"])
        self.assertIn("invalid_level", out["evidence_llm_error"])

    def test_skip_when_run_flag_is_false(self):
        llm = _DummyLLM(
            '{"level":"high","action":"answer","reason":"should_not_run"}'
        )
        prompts = _DummyPrompts()
        run = node_evidence_llm_judge(llm=llm, prompts=prompts)

        out = run({"run_evidence_llm": False, "retrieved": []})

        self.assertEqual(out, {})
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(len(prompts.calls), 0)


if __name__ == "__main__":
    unittest.main()
