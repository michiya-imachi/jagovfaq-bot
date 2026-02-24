import unittest
from unittest.mock import patch

from app.nodes.web import node_web_permission


class WebPermissionTests(unittest.TestCase):
    def test_yes_words_allow_search(self):
        run = node_web_permission()

        with patch("app.nodes.web.interrupt", return_value="yes"):
            out = run({})
        self.assertTrue(out["web_search_allowed"])
        self.assertFalse(out["web_search_declined"])
        self.assertTrue(out["hitl_permission_web_asked"])

        with patch("app.nodes.web.interrupt", return_value="ok"):
            out = run({})
        self.assertTrue(out["web_search_allowed"])
        self.assertFalse(out["web_search_declined"])
        self.assertTrue(out["hitl_permission_web_asked"])

    def test_no_or_unknown_words_decline_search(self):
        run = node_web_permission()

        with patch("app.nodes.web.interrupt", return_value="no"):
            out = run({})
        self.assertFalse(out["web_search_allowed"])
        self.assertTrue(out["web_search_declined"])
        self.assertTrue(out["hitl_permission_web_asked"])

        with patch("app.nodes.web.interrupt", return_value="maybe"):
            out = run({})
        self.assertFalse(out["web_search_allowed"])
        self.assertTrue(out["web_search_declined"])
        self.assertTrue(out["hitl_permission_web_asked"])

        with patch("app.nodes.web.interrupt", return_value=""):
            out = run({})
        self.assertFalse(out["web_search_allowed"])
        self.assertTrue(out["web_search_declined"])
        self.assertTrue(out["hitl_permission_web_asked"])


if __name__ == "__main__":
    unittest.main()
