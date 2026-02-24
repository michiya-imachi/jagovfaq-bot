import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.nodes.qa import node_generate_answer_stream


class _DummyPrompts:
    def __init__(self):
        self.calls = []

    def render_pair(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        return {"system": "sys", "user": "user"}


class _DummyLLM:
    def __init__(self, text: str = "LLM回答"):
        self.text = text

    def stream(self, _messages):
        yield SimpleNamespace(content=self.text)


def _candidate(url: str, question: str = "Q", answer: str = "A"):
    return {
        "item": {"id": 1, "question": question, "answer": answer, "url": url},
        "score": 0.5,
        "sources": ["bm25"],
        "source_details": {},
    }


class AnswerWebIntegrationTests(unittest.TestCase):
    def test_uses_web_prompt_when_web_results_exist_and_dedupes_citations(self):
        prompts = _DummyPrompts()
        llm = _DummyLLM(text="本文")
        run = node_generate_answer_stream(llm, prompts)

        state = {
            "search_query": "テスト",
            "retrieved": [_candidate("https://local.example/1")],
            "web_results": [
                {"title": "T1", "url": "https://web.example/1", "snippet": ""},
                {"title": "T2", "url": "https://local.example/1", "snippet": ""},
            ],
            "local_evidence_level": "high",
        }

        with patch("sys.stdout", new=io.StringIO()):
            out = run(state)

        self.assertEqual(prompts.calls[-1][0], "generate_answer_with_web")
        self.assertEqual(len(out["citations"]), 2)
        self.assertEqual(out["citations"][0]["url"], "https://local.example/1")
        self.assertEqual(out["citations"][1]["url"], "https://web.example/1")

    def test_declined_adds_warning(self):
        prompts = _DummyPrompts()
        llm = _DummyLLM(text="本文")
        run = node_generate_answer_stream(llm, prompts)

        state = {
            "search_query": "テスト",
            "retrieved": [_candidate("https://local.example/1")],
            "web_results": [],
            "web_search_declined": True,
            "local_evidence_level": "low",
        }

        with patch("sys.stdout", new=io.StringIO()):
            out = run(state)

        self.assertEqual(prompts.calls[-1][0], "generate_answer")
        self.assertIn("Web検索は実行していません", out["answer"])
        self.assertIn("ローカルFAQ候補が弱い", out["answer"])

    def test_web_error_and_zero_results_warnings(self):
        prompts = _DummyPrompts()
        llm = _DummyLLM(text="本文")
        run = node_generate_answer_stream(llm, prompts)

        state_error = {
            "search_query": "テスト",
            "retrieved": [_candidate("https://local.example/1")],
            "web_results": [],
            "web_search_error": "RuntimeError: boom",
            "local_evidence_level": "low",
        }
        with patch("sys.stdout", new=io.StringIO()):
            out_error = run(state_error)
        self.assertIn("Web検索が失敗したため", out_error["answer"])

        state_zero = {
            "search_query": "テスト",
            "retrieved": [_candidate("https://local.example/1")],
            "web_results": [],
            "web_search_attempted": True,
            "local_evidence_level": "low",
        }
        with patch("sys.stdout", new=io.StringIO()):
            out_zero = run(state_zero)
        self.assertIn("Web検索でも確証が得られませんでした", out_zero["answer"])

if __name__ == "__main__":
    unittest.main()
