import io
import unittest
from unittest.mock import patch

from app.nodes.web import node_web_search


class _DummySource:
    def __init__(self, url: str):
        self.url = url


class _DummyAction:
    def __init__(self, action_type: str, sources):
        self.type = action_type
        self.sources = sources


class _DummyItem:
    def __init__(self, item_type: str, action):
        self.type = item_type
        self.action = action


class _DummyResponse:
    def __init__(self, output):
        self.output = output


class _DummyResponsesAPI:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls = []

    def create(self, **_kwargs):
        self.calls.append(dict(_kwargs))
        if self._error is not None:
            raise self._error
        return self._response


class _DummyClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.responses = _DummyResponsesAPI(response=response, error=error)


class _DummyClientWithOptions:
    def __init__(self, response=None, error: Exception | None = None):
        self.responses = _DummyResponsesAPI(response=response, error=error)
        self.with_options_calls = []

    def with_options(self, **kwargs):
        self.with_options_calls.append(dict(kwargs))
        return self


class WebSearchTests(unittest.TestCase):
    def test_success_extracts_urls_and_limits_topk(self):
        response = _DummyResponse(
            [
                _DummyItem(
                    "web_search_call",
                    _DummyAction(
                        "search",
                        [
                            _DummySource("https://example.com/a"),
                            _DummySource("https://example.com/b"),
                            _DummySource("https://example.com/a"),
                        ],
                    ),
                )
            ]
        )
        client = _DummyClient(response=response)
        run = node_web_search(client)

        with patch.dict("os.environ", {"WEB_TOPK": "2"}, clear=False):
            out = run({"search_query": "foo", "web_search_allowed": True})

        self.assertEqual(out["web_query"], "foo")
        self.assertEqual(len(out["web_results"]), 2)
        self.assertEqual(out["web_results"][0]["url"], "https://example.com/a")
        self.assertEqual(out["web_results"][1]["url"], "https://example.com/b")
        self.assertTrue(out["web_search_attempted"])
        self.assertEqual(out["web_search_error"], "")

    def test_failure_sets_error(self):
        client = _DummyClient(error=RuntimeError("boom"))
        run = node_web_search(client)
        out = run({"search_query": "foo", "web_search_allowed": True})

        self.assertTrue(out["web_search_attempted"])
        self.assertEqual(out["web_results"], [])
        self.assertIn("RuntimeError", out["web_search_error"])

    def test_zero_results_is_allowed(self):
        response = _DummyResponse([])
        client = _DummyClient(response=response)
        run = node_web_search(client)
        out = run({"search_query": "foo", "web_search_allowed": True})

        self.assertTrue(out["web_search_attempted"])
        self.assertEqual(out["web_results"], [])
        self.assertEqual(out["web_search_error"], "")

    def test_default_timeout_is_45_seconds(self):
        response = _DummyResponse([])
        client = _DummyClient(response=response)
        run = node_web_search(client)

        with patch.dict("os.environ", {}, clear=True):
            out = run({"search_query": "foo", "web_search_allowed": True})

        self.assertTrue(out["web_search_attempted"])
        self.assertEqual(client.responses.calls[0]["timeout"], 45.0)

    def test_env_timeout_override(self):
        response = _DummyResponse([])
        client = _DummyClient(response=response)
        run = node_web_search(client)

        with patch.dict("os.environ", {"WEB_SEARCH_TIMEOUT_S": "60"}, clear=True):
            out = run({"search_query": "foo", "web_search_allowed": True})

        self.assertTrue(out["web_search_attempted"])
        self.assertEqual(client.responses.calls[0]["timeout"], 60.0)

    def test_retries_uses_with_options_when_available(self):
        response = _DummyResponse([])
        client = _DummyClientWithOptions(response=response)
        run = node_web_search(client)

        with patch.dict("os.environ", {"WEB_SEARCH_MAX_RETRIES": "1"}, clear=True):
            out = run({"search_query": "foo", "web_search_allowed": True})

        self.assertTrue(out["web_search_attempted"])
        self.assertEqual(client.with_options_calls, [{"max_retries": 1}])
        self.assertEqual(client.responses.calls[0]["timeout"], 45.0)

    def test_retries_fallback_without_with_options(self):
        response = _DummyResponse([])
        client = _DummyClient(response=response)
        run = node_web_search(client)

        with patch.dict("os.environ", {"WEB_SEARCH_MAX_RETRIES": "1"}, clear=True):
            out = run({"search_query": "foo", "web_search_allowed": True})

        self.assertTrue(out["web_search_attempted"])
        self.assertEqual(out["web_search_error"], "")
        self.assertEqual(client.responses.calls[0]["timeout"], 45.0)

    def test_retries_are_clamped_to_allowed_range(self):
        response = _DummyResponse([])
        client = _DummyClientWithOptions(response=response)
        run = node_web_search(client)

        with patch.dict("os.environ", {"WEB_SEARCH_MAX_RETRIES": "99"}, clear=True):
            out_high = run({"search_query": "foo", "web_search_allowed": True})
        with patch.dict("os.environ", {"WEB_SEARCH_MAX_RETRIES": "-3"}, clear=True):
            out_low = run({"search_query": "bar", "web_search_allowed": True})

        self.assertTrue(out_high["web_search_attempted"])
        self.assertTrue(out_low["web_search_attempted"])
        self.assertEqual(
            client.with_options_calls,
            [{"max_retries": 5}, {"max_retries": 0}],
        )

    def test_prints_searching_message_while_web_search_runs(self):
        response = _DummyResponse([])
        client = _DummyClient(response=response)
        run = node_web_search(client)

        with patch("sys.stdout", new=io.StringIO()) as out:
            run({"search_query": "foo", "web_search_allowed": True})
            printed = out.getvalue()

        self.assertIn("Web検索中...", printed)


if __name__ == "__main__":
    unittest.main()
