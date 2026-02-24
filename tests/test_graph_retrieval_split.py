import unittest

from app.graph.builder import build_graph_for_export
from app.prompts.loader import PromptLoader


class GraphRetrievalSplitTests(unittest.TestCase):
    def test_export_graph_uses_split_retrieval_nodes(self):
        app = build_graph_for_export(PromptLoader.load_default())
        mermaid = str(app.get_graph().draw_mermaid())

        self.assertIn("retrieve_bm25", mermaid)
        self.assertIn("retrieve_vec", mermaid)
        self.assertNotIn("retrieve_all", mermaid)
        self.assertIn("hitl_input_answer", mermaid)
        self.assertIn("hitl_permission_web", mermaid)
        self.assertNotIn("HITL", mermaid)


if __name__ == "__main__":
    unittest.main()
