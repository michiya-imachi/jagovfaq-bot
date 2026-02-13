import unittest

from app.core.retriever import (
    RetrieverRegistry,
    normalize_retriever_names,
    parse_retriever_names,
)


class DummyRetriever:
    def __init__(self, name: str):
        self.name = name

    def retrieve(self, query: str, state):
        return []


class RetrieverRegistryTests(unittest.TestCase):
    def test_parse_retriever_names(self):
        self.assertEqual(parse_retriever_names("bm25, vec"), ["bm25", "vec"])
        self.assertEqual(parse_retriever_names(None), [])

    def test_normalize_retriever_names(self):
        self.assertEqual(
            normalize_retriever_names([" BM25 ", "vec", "bm25"]),
            ["bm25", "vec"],
        )

    def test_select_with_dedup(self):
        reg = RetrieverRegistry([DummyRetriever("bm25"), DummyRetriever("vec")])
        selected = reg.select(["bm25", "vec", "bm25"])
        self.assertEqual([r.name for r in selected], ["bm25", "vec"])

    def test_select_unknown_raises(self):
        reg = RetrieverRegistry([DummyRetriever("bm25"), DummyRetriever("vec")])
        with self.assertRaises(ValueError):
            reg.select(["bm25", "foo"])


if __name__ == "__main__":
    unittest.main()
