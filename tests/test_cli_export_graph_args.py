import unittest

from app.app import parse_args


class CliExportGraphArgsTests(unittest.TestCase):
    def test_export_graph_option_is_valid(self):
        args = parse_args(["--export-graph"])
        self.assertTrue(args.export_graph)

    def test_legacy_graph_xray_option_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--export-graph", "--graph-xray"])

    def test_legacy_graph_xray_depth_option_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--export-graph", "--graph-xray-depth", "1"])


if __name__ == "__main__":
    unittest.main()
