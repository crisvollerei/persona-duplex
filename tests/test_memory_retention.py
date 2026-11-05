"""Preserve user statements as evidence without upgrading model interpretations."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from persona.consolidate import Consolidator
from persona.store import MAX_JUDGMENT_QUOTES, Store, rebuild_views


class MemoryRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name))
        self.events = [
            {"turn_id": "t_1", "user": "我明确采用17分钟专注、3分钟休息，适合我。",
             "need": "需要学习建议", "direct": "模型猜测用户习惯25分钟", "say": "试试25分钟。"},
            {"turn_id": "t_2", "user": "我继续用17分钟专注、3分钟休息。",
             "need": "需要学习建议", "direct": "模型推断", "say": "可以继续。"},
        ]

    def item(self, **kwargs):
        return {"judgment": "用户倾向通过固定学习节奏启动。", "confidence": .7,
                "evidence_ids": ["t_1", "t_2"], **kwargs}

    def consolidate(self, item=None, events=None):
        model = lambda prompt: json.dumps([item or self.item()], ensure_ascii=False)
        return Consolidator(self.store, llm=model)._consolidate(self.events if events is None else events)

    def test_abstract_judgment_keeps_exact_numeric_preference(self):
        record, = self.consolidate()
        self.assertEqual([q["quote"] for q in record["evidence_quotes"]],
                         [event["user"] for event in self.events])
        self.assertTrue(all(q["evidence_scope"] == "specified" for q in record["evidence_quotes"]))
        self.assertNotIn("25分钟", json.dumps(record["evidence_quotes"], ensure_ascii=False))

    def test_substring_quote_retains_negation_units_and_correction(self):
        self.events[0]["user"] = "更正：我不喜欢咖啡，偏好热的无糖豆浆，不是冰豆浆。"
        record, = self.consolidate(self.item(evidence_ids=["t_1"], evidence_quotes=[
            {"turn_id": "t_1", "quote": "喜欢咖啡"}]))
        self.assertEqual(record["evidence_quotes"][0]["quote"], self.events[0]["user"])

    def test_batch_reference_does_not_claim_specified_support(self):
        item = self.item()
        item.pop("evidence_ids")
        record, = self.consolidate(item)
        self.assertEqual(record["evidence_scope"], "batch")
        self.assertEqual({q["evidence_scope"] for q in record["evidence_quotes"]}, {"batch"})

    def test_only_selected_user_events_are_attached(self):
        record, = self.consolidate(self.item(evidence_ids=["t_2"]))
        self.assertEqual(record["evidence_quotes"], [{
            "turn_id": "t_2", "quote": self.events[1]["user"], "evidence_scope": "specified"}])

    def test_quote_recency_uses_event_order_not_model_id_order(self):
        record, = self.consolidate(self.item(evidence_ids=["t_2", "t_1"]))
        self.assertEqual([q["turn_id"] for q in record["evidence_quotes"]], ["t_1", "t_2"])

    def test_explicit_empty_evidence_does_not_attach_the_whole_batch(self):
        record, = self.consolidate(self.item(evidence_ids=[]))
        self.assertNotIn("evidence_quotes", record)

    def test_unknown_or_unselected_quote_id_is_rejected(self):
        for turn_id in ("not-in-batch", "t_2"):
            with self.subTest(turn_id=turn_id), self.assertRaises(ValueError):
                self.consolidate(self.item(evidence_ids=["t_1"], evidence_quotes=[{
                    "turn_id": turn_id, "quote": self.events[0]["user"]}]))
        self.assertEqual(self.store.judgments(), [])

    def test_model_generated_content_cannot_be_quoted_as_user_evidence(self):
        with self.assertRaises(ValueError):
            self.consolidate(self.item(evidence_quotes=[{"turn_id": "t_1", "quote": "25分钟"}]))
        self.assertEqual(self.store.judgments(), [])

    def test_fabricated_quote_preserves_backlog(self):
        for event in self.events:
            self.store.push_buffer(event)
        item = self.item(evidence_quotes=[{"turn_id": "t_1", "quote": "用户从未说过的话"}])
        with self.assertRaises(ValueError):
            Consolidator(self.store, llm=lambda prompt: json.dumps([item])).run_once()
        self.assertEqual(self.store.judgments(), [])
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))), 2)

    def test_malformed_quotes_are_rejected(self):
        for value in ({}, [None], [{"turn_id": "t_1", "quote": ""}],
                      [{"turn_id": "t_1", "quote": 17}]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.consolidate(self.item(evidence_quotes=value))

    def test_rebuild_and_restart_preserve_exact_source_statements(self):
        record, = self.consolidate()
        expected = record["evidence_quotes"]
        reopened = Store(self.store.root)
        self.assertEqual(reopened.user_judgments()[0]["evidence_quotes"], expected)
        rebuild_views(reopened)
        self.assertEqual(reopened.load_view("judgments")["items"][0]["evidence_quotes"], expected)

    def test_dedup_refreshes_evidence_without_replacing_or_strengthening_judgment(self):
        old = self.store.add_judgment(self.item()["judgment"], .9, affects={"warmth": 1})
        self.assertEqual(self.consolidate(self.item(confidence=.4)), [])
        refreshed, = self.store.judgments()
        self.assertEqual(refreshed["id"], old["id"])
        self.assertEqual(refreshed["confidence"], .9)
        self.assertEqual(refreshed["affects"], {"warmth": 1})
        self.assertEqual(len(refreshed["evidence_quotes"]), 2)
        self.assertEqual(sum(r["kind"] == "judgment_add" for r in self.store.read_ledger()), 1)
        self.assertEqual(sum(r["kind"] == "judgment_evidence" for r in self.store.read_ledger()), 1)

    def test_same_evidence_does_not_create_another_refresh(self):
        self.consolidate()
        self.assertEqual(self.consolidate(), [])
        self.assertEqual(sum(r["kind"] == "judgment_evidence" for r in self.store.read_ledger()), 0)

    def test_metadata_refresh_cannot_revive_a_concurrently_superseded_judgment(self):
        old = self.store.add_judgment(self.item()["judgment"], .7)
        con = Consolidator(self.store, llm=lambda prompt: json.dumps([self.item()]))
        candidates = con._draft_judgments(self.events)
        current = self.store.add_judgment("用户不再采用之前的固定节奏。", .8, supersedes=old["id"])
        self.assertEqual(con._save_judgments(candidates), [])
        self.assertEqual([j["id"] for j in self.store.user_judgments()], [current["id"]])

    def test_metadata_only_refresh_does_not_call_story_model(self):
        self.store.add_judgment(self.item()["judgment"], .7)
        for event in self.events:
            self.store.push_buffer(event)
        calls = []
        def model(prompt):
            calls.append(prompt)
            return json.dumps([self.item()])
        result = Consolidator(self.store, llm=model).run_once()
        self.assertEqual(result["judgments"], 0)
        self.assertFalse(result["self_story"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.store.judgments()[0]["evidence_quotes"]), 2)
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])

    def test_evidence_refresh_replays_and_is_bounded_without_erasing_ledger(self):
        old = self.store.add_judgment("preference", .6)
        for index in range(MAX_JUDGMENT_QUOTES + 4):
            self.store.attach_judgment_evidence(old["id"], [{
                "turn_id": f"t_{index}", "quote": f"user statement {index}", "evidence_scope": "specified"}])
        quotes = self.store.judgments()[0]["evidence_quotes"]
        self.assertEqual(len(quotes), MAX_JUDGMENT_QUOTES)
        self.assertEqual(quotes[0]["turn_id"], "t_4")
        rebuild_views(self.store)
        self.assertEqual(self.store.judgments()[0]["evidence_quotes"], quotes)
        self.assertEqual(sum(r["kind"] == "judgment_evidence" for r in self.store.read_ledger()), MAX_JUDGMENT_QUOTES + 4)

    def test_model_readings_never_become_user_quotes_in_heuristic_path(self):
        for event in self.events:
            self.store.push_buffer(event)
        Consolidator(self.store).run_once()
        diagnostic, = self.store.judgments()
        self.assertNotIn("evidence_quotes", diagnostic)
        self.assertEqual(self.store.user_judgments(), [])

    def test_memory_model_rejection_does_not_create_user_memory(self):
        result = Consolidator(self.store, llm=lambda prompt: "[]")._consolidate(self.events)
        self.assertEqual(result, [])
        self.assertEqual(self.store.judgments(), [])

    def test_source_backed_correction_supersedes_previous_preference(self):
        old = self.store.add_judgment("用户喜欢冰咖啡", .7)
        self.events[0]["user"] = "更正以前的偏好，我不喜欢冰咖啡了，今后选热无糖豆浆。"
        record, = self.consolidate(self.item(judgment="用户明确更正为热无糖豆浆。",
                                            supersedes=old["id"], evidence_ids=["t_1"]))
        self.assertEqual([j["id"] for j in self.store.user_judgments()], [record["id"]])
        self.assertEqual(record["evidence_quotes"][0]["quote"], self.events[0]["user"])

    def test_prompt_preserves_concrete_preferences_without_upgrading_model_readings(self):
        prompt = Consolidator(self.store)._consolidation_prompt(self.events)
        self.assertIn("具体名称、数字、单位、否认和条件", prompt)
        self.assertIn("evidence_quotes", prompt)
        self.assertIn("当前明确更正优先", prompt)
        self.assertIn("反复出现同一个模型读法，不算重复的用户证据", prompt)
        self.assertNotIn("写倾向，不写事实", prompt)

    def test_story_failure_cannot_commit_quotes_without_the_judgment(self):
        for event in self.events:
            self.store.push_buffer(event)
        con = Consolidator(self.store, llm=lambda prompt: json.dumps([self.item()])
                           if "只输出 JSON 数组" in prompt else "A remembered preference.")
        with patch.object(con, "_rewrite_story", side_effect=OSError("disk")), self.assertRaises(OSError):
            con.run_once()
        self.assertEqual(self.store.judgments(), [])
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
