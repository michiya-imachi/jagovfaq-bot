import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

from app.graph.export import ExportGraphArtifactsResult, export_graph_artifacts


class _GraphWithOutputPath:
    def __init__(self, marker: str):
        self._marker = marker

    def draw_mermaid(self):
        return f"graph TD;\nA-->{self._marker}"

    def draw_mermaid_png(self, output_file_path=None):
        if output_file_path is None:
            return b"PNG"
        Path(output_file_path).write_bytes(f"PNG:{self._marker}".encode("utf-8"))


class _GraphWithBytesFallback:
    def __init__(self, marker: str):
        self._marker = marker

    def draw_mermaid(self):
        return f"graph TD;\nA-->{self._marker}"

    def draw_mermaid_png(self, output_file_path=None):
        if output_file_path is not None:
            raise TypeError("output_file_path is not supported")
        return f"PNG:{self._marker}".encode("utf-8")


class _GraphWithPngFailure:
    def __init__(self, marker: str):
        self._marker = marker

    def draw_mermaid(self):
        return f"graph TD;\nA-->{self._marker}"

    def draw_mermaid_png(self, output_file_path=None):
        raise RuntimeError("boom")


class GraphExportXrayTests(unittest.TestCase):
    def test_export_always_outputs_collapsed_and_expanded(self):
        app = Mock()
        app.get_graph.side_effect = [
            _GraphWithOutputPath("collapsed"),
            _GraphWithOutputPath("expanded"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = export_graph_artifacts(app=app, out_dir=Path(tmp))

            self.assertIsInstance(result, ExportGraphArtifactsResult)
            app.get_graph.assert_has_calls([call(xray=False), call(xray=True)])
            self.assertEqual(app.get_graph.call_count, 2)

            self.assertEqual(result.collapsed_mmd.name, "graph_collapsed.mmd")
            self.assertEqual(result.collapsed_png.name, "graph_collapsed.png")
            self.assertEqual(result.expanded_mmd.name, "graph_expanded.mmd")
            self.assertEqual(result.expanded_png.name, "graph_expanded.png")

            self.assertIn(
                "collapsed",
                result.collapsed_mmd.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "expanded",
                result.expanded_mmd.read_text(encoding="utf-8"),
            )

    def test_export_png_bytes_fallback_works_for_both_graphs(self):
        app = Mock()
        app.get_graph.side_effect = [
            _GraphWithBytesFallback("collapsed"),
            _GraphWithBytesFallback("expanded"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = export_graph_artifacts(app=app, out_dir=Path(tmp))

            self.assertIsNotNone(result.collapsed_png)
            self.assertIsNotNone(result.expanded_png)
            self.assertTrue(result.collapsed_png.exists())
            self.assertTrue(result.expanded_png.exists())

    def test_export_returns_partial_result_when_one_png_fails(self):
        app = Mock()
        app.get_graph.side_effect = [
            _GraphWithPngFailure("collapsed"),
            _GraphWithOutputPath("expanded"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = export_graph_artifacts(app=app, out_dir=Path(tmp))

            self.assertIsNone(result.collapsed_png)
            self.assertIsNotNone(result.expanded_png)
            self.assertTrue(result.collapsed_mmd.exists())
            self.assertTrue(result.expanded_mmd.exists())


if __name__ == "__main__":
    unittest.main()
