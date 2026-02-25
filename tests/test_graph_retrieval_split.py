import unittest

from app.graph.builder import build_graph_for_export
from app.prompts.loader import PromptLoader


class GraphRetrievalSplitTests(unittest.TestCase):
    def test_export_graph_collapses_rag_and_evidence_subgraphs_by_default(self):
        app = build_graph_for_export(PromptLoader.load_default())
        mermaid = str(app.get_graph().draw_mermaid())

        self.assertIn("rag", mermaid)
        self.assertIn("evidence", mermaid)
        self.assertIn("response_router", mermaid)

        self.assertIn("__start__ --> rag", mermaid)
        self.assertIn("rag --> evidence", mermaid)
        self.assertIn("evidence --> response_router", mermaid)
        self.assertIn("hitl_input_answer --> rag", mermaid)

        self.assertNotIn("standalone_question", mermaid)
        self.assertNotIn("retrieval_router", mermaid)
        self.assertNotIn("retrieve_bm25", mermaid)
        self.assertNotIn("retrieve_vec", mermaid)
        self.assertNotIn("rrf_rank", mermaid)
        self.assertNotIn("topk_filter", mermaid)
        self.assertNotIn("restore_topk_meta", mermaid)
        self.assertNotIn("evidence_rules_strict", mermaid)
        self.assertNotIn("evidence_router", mermaid)
        self.assertNotIn("evidence_llm_judge", mermaid)
        self.assertNotIn("evidence_finalize", mermaid)

    def test_export_graph_xray_expands_rag_and_evidence_subgraphs(self):
        app = build_graph_for_export(PromptLoader.load_default())
        mermaid = str(app.get_graph(xray=True).draw_mermaid())

        self.assertIn("hitl_input_answer", mermaid)
        self.assertIn("hitl_permission_web", mermaid)
        self.assertIn("standalone_question", mermaid)
        self.assertIn("retrieval_router", mermaid)
        self.assertIn("retrieve_bm25", mermaid)
        self.assertIn("retrieve_vec", mermaid)
        self.assertIn("rrf_rank", mermaid)
        self.assertIn("topk_filter", mermaid)
        self.assertIn("restore_topk_meta", mermaid)
        self.assertIn("evidence_rules_strict", mermaid)
        self.assertIn("evidence_router", mermaid)
        self.assertIn("evidence_llm_judge", mermaid)
        self.assertIn("evidence_finalize", mermaid)
        self.assertIn("response_router", mermaid)
        self.assertIn("restore_topk_meta --> evidence", mermaid)
        self.assertNotIn("evidence_assess", mermaid)
        self.assertNotIn("decide_next_action", mermaid)


if __name__ == "__main__":
    unittest.main()
