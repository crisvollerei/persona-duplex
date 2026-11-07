"""Offline checks of the actual selection and delivery boundaries."""
import json
import unittest
from unittest.mock import Mock
from selection import parse_pool, parse_choice, select, random_select
from benchmark import run_output, messages


class SelectionChecks(unittest.TestCase):
    def setUp(self):
        self.pool = [{"id": f"C{i+1:02}", "text": f"自然回复{i}", "attention": i % 3,
                      "initiative": i // 3} for i in range(6)]
        self.profile = {"id": "P", "target": {"attention": 2, "initiative": 1}, "description": "偏好"}
        self.case = {"id": "S", "history": [], "user": "聊天", "facts": [], "boundaries": []}

    def test_reject_corrupt_labels_and_duplicates(self):
        valid = {"candidates": self.pool}
        self.assertEqual(parse_pool(json.dumps(valid)), self.pool)
        for value in (True, 1.0, 3, None):
            changed = json.loads(json.dumps(valid)); changed["candidates"][0]["attention"] = value
            with self.assertRaises(ValueError): parse_pool(json.dumps(changed))
        with self.assertRaises(ValueError): parse_pool('{"candidates":[],"candidates":[]}')

    def test_order_independent_program_and_random(self):
        self.assertEqual(select(self.pool, self.profile), "C06")
        self.assertEqual(select(list(reversed(self.pool)), self.profile), "C06")
        self.assertEqual(random_select(self.pool, "S", "P"), random_select(list(reversed(self.pool)), "S", "P"))

    def test_only_existing_choice_or_explicit_abstention(self):
        self.assertIsNone(parse_choice('{"candidate_id":null}', self.pool))
        for text in ('{"candidate_id":"missing"}', '{"candidate_id":"C01","reason":"x"}', '{"candidate_id":1}'):
            with self.assertRaises(ValueError): parse_choice(text, self.pool)

    def test_program_delivers_original_candidate_without_model(self):
        client = Mock(); client.call.side_effect = AssertionError("must not call")
        row = run_output(("S", "P", "program"), {"S": self.case}, {"P": self.profile},
                         {"S": {"status": "ok", "candidates": self.pool}}, client)
        self.assertEqual(row["say"], self.pool[5]["text"])
        self.assertIsNone(row["call_id"])

    def test_failed_pool_does_not_disable_direct(self):
        client = Mock(); client.call.return_value = {"id": "S-P-direct", "status": "ok", "output": "直接回复"}
        args = ({"S": self.case}, {"P": self.profile}, {"S": {"status": "invalid_pool", "candidates": []}}, client)
        self.assertEqual(run_output(("S", "P", "program"), *args)["status"], "pool_unavailable")
        self.assertEqual(run_output(("S", "P", "direct"), *args)["say"], "直接回复")

    def test_pool_cannot_see_target_and_context_is_shared(self):
        pool = json.loads(messages("pools", self.case)[1]["content"])
        direct = json.loads(messages("direct", self.case, self.profile)[1]["content"])
        prompt = json.loads(messages("prompt", self.case, self.profile, self.pool)[1]["content"])
        self.assertEqual(pool, {"context": direct["context"]})
        self.assertEqual(prompt["context"], direct["context"])
        self.assertEqual(prompt["profile"], direct["profile"])


if __name__ == "__main__": unittest.main()
