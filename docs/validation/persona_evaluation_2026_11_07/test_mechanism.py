import json
import unittest
from benchmark import HERE, load, generation_messages, label_messages, trial
from mechanism import choose, validate_annotations
from prepare_reviews import visible_profile

class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.case = load(HERE / "cases.json")[0]
        self.profile = load(HERE / "profiles.json")[0]

    def test_both_generators_receive_identical_persona_and_context(self):
        a = json.loads(generation_messages(self.case, self.profile, "direct")[1]["content"])
        b = json.loads(generation_messages(self.case, self.profile, "coverage")[1]["content"])
        self.assertEqual(a["persona"], b["persona"]); self.assertEqual(a["context"], b["context"])

    def test_label_request_does_not_receive_target_or_self_labels(self):
        pool = [{"id": f"C{i:02}", "text": f"观察{i}", "attention": 0, "initiative": 0} for i in range(1,7)]
        messages, hidden, mapping = label_messages(self.case, "test", pool)
        obj = json.loads(messages[1]["content"])
        self.assertEqual(set(obj), {"context", "candidates"})
        self.assertTrue(all(set(c) == {"id", "text"} for c in obj["candidates"]))
        self.assertEqual(set(mapping.values()), {r["id"] for r in pool})

    def test_unknown_cannot_fall_back_to_self_tags(self):
        pool = [{"id": "C01", "text": "确认。", "attention": 1, "initiative": 0}]
        self.assertIsNone(choose(pool, [{"id": "C01", "attention": None, "initiative": 0}], self.profile))

    def test_selector_delivers_exact_text_without_style_gate(self):
        text = "已记录。这里仍有一枚新的叶芽。"
        pool = [{"id": "C01", "text": text}]
        label = {"id": "C01", **self.profile["target"]}
        self.assertEqual(choose(pool, [label], self.profile)["text"], text)

    def test_review_persona_is_visible_target_hidden(self):
        obj = visible_profile(self.profile)
        self.assertEqual(set(obj), {"name", "spec"}); self.assertEqual(obj["spec"], self.profile["spec"])

    def test_literal_evidence_is_not_semantic_truth(self):
        hidden = [{"id": "X01", "text": "你好。"}]
        raw = json.dumps({"labels": [{"id": "X01", "attention": 2,
            "focus_evidence": [{"kind": "experience", "quote": "你好"}],
            "initiative": 0, "initiative_quote": "你好。"}]})
        # Literal validation accepts this semantically questionable proposal.
        self.assertEqual(validate_annotations(raw, hidden, {"X01": "C01"})[0]["attention"], 2)

if __name__ == "__main__": unittest.main()
