import unittest

from app.app import _interrupt_ui, _payload_to_type


class CliInterruptModeTests(unittest.TestCase):
    def test_hitl_mode_label_and_prompt(self):
        payload = {"type": "HITL_INPUT_ANSWER", "question": "追加質問ですか？"}
        label, prompt = _interrupt_ui(payload)
        self.assertEqual(label, "追加質問: 追加質問ですか？")
        self.assertEqual(prompt, ">> ")

    def test_web_permission_label_and_prompt(self):
        payload = {"type": "HITL_PERMISSION_WEB", "question": "Web検索しますか？"}
        label, prompt = _interrupt_ui(payload)
        self.assertEqual(label, "Web検索確認: Web検索しますか？")
        self.assertEqual(prompt, "(はい/いいえ) >> ")

    def test_missing_type_defaults_to_hitl(self):
        payload = {"question": "typeなし"}
        self.assertEqual(_payload_to_type(payload), "HITL_INPUT_ANSWER")
        label, prompt = _interrupt_ui(payload)
        self.assertEqual(label, "追加質問: typeなし")
        self.assertEqual(prompt, ">> ")


if __name__ == "__main__":
    unittest.main()
