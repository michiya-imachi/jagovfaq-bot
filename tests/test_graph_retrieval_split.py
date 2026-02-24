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
        self.assertIn("restore_topk_meta", mermaid)
        self.assertIn("evidence_rules_strict", mermaid)
        self.assertIn("evidence_router", mermaid)
        self.assertIn("evidence_llm_judge", mermaid)
        self.assertIn("evidence_finalize", mermaid)
        self.assertIn("response_router", mermaid)
        self.assertIn("topk_filter --> restore_topk_meta", mermaid)
        self.assertIn("restore_topk_meta --> evidence_rules_strict", mermaid)
        self.assertNotIn("topk_filter --> evidence_rules_strict", mermaid)
        self.assertNotIn("evidence_assess", mermaid)
        self.assertNotIn("decide_next_action", mermaid)
        self.assertNotIn("HITL", mermaid)


if __name__ == "__main__":
    unittest.main()
