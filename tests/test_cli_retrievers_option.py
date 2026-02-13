import unittest

from app.app import resolve_retriever_selection


class CliRetrieverOptionTests(unittest.TestCase):
    def test_cli_overrides_env(self):
        names, source = resolve_retriever_selection(
            cli_value="bm25",
            env_value="vec",
            default_names=["bm25", "vec"],
        )
        self.assertEqual(names, ["bm25"])
        self.assertEqual(source, "cli")

    def test_env_used_when_cli_missing(self):
        names, source = resolve_retriever_selection(
            cli_value=None,
            env_value="vec,bm25",
            default_names=["bm25", "vec"],
        )
        self.assertEqual(names, ["vec", "bm25"])
        self.assertEqual(source, "env")

    def test_default_used_when_both_missing(self):
        names, source = resolve_retriever_selection(
            cli_value=None,
            env_value=None,
            default_names=["bm25", "vec"],
        )
        self.assertEqual(names, ["bm25", "vec"])
        self.assertEqual(source, "default")

    def test_empty_cli_value_is_invalid(self):
        with self.assertRaises(ValueError):
            resolve_retriever_selection(
                cli_value="",
                env_value="bm25",
                default_names=["bm25", "vec"],
            )


if __name__ == "__main__":
    unittest.main()
