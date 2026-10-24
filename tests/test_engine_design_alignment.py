"""Local design boundaries; scripted outputs do not prove model behavior."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.prompt import SECTION_HEAD, SECTION_SAY, SECTION_TAIL
from persona.store import Store
from persona.topicjudge import AsyncJudge, LLMJudge


ROOT = Path(__file__).resolve().parent.parent
HEAD = "need: listen\nconfidence: .7\ndirect: neutral\nbaseline: ordinary\ndisplace: none"
TAIL = ("settle: -\ndelta:\n  closeness: .05\npending:\n"
        "  claim: new reading\n  confidence: .6\n  predicts: new evidence")


def malformed(tail=TAIL):
    return f"{SECTION_SAY}\nHello.\n{SECTION_HEAD}\n{HEAD}\n{SECTION_TAIL}\n{tail}"


class DesignAlignmentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(tmp.name)
        self.cfg = PersonaConfig.load(ROOT / "personas" / "default.toml")

    def run_output(self, output, streaming=False, **kwargs):
        llm = ScriptedLLM([output])
        stream_calls = []
        emitted = []

        def stream(messages):
            stream_calls.append(messages)
            yield from output

        engine = TurnEngine(self.cfg, self.store, llm,
                            stream_llm=stream if streaming else None, **kwargs)
        result = engine.turn("hello", on_delta=emitted.append if streaming else None)
        self.assertEqual(len(llm.calls) + len(stream_calls), 1)
        if streaming:
            self.assertEqual("".join(emitted), result.say)
        return engine, result

    def test_wrong_order_never_applies_tail_or_recycles_metadata(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                before = self.store.relation()
                engine, result = self.run_output(malformed(), streaming)
                self.assertEqual(result.say, "Hello.")
                self.assertFalse(result.validity["valid"])
                self.assertIsNone(result.pending_added)
                self.assertEqual(result.relation_changed, {})
                self.assertEqual(self.store.relation(), before)
                self.assertEqual(engine.history[-1]["content"], result.say)
                self.assertEqual(engine._last_need, "")
                self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])

    def test_wrong_order_cannot_settle_existing_pending(self):
        pending = self.store.add_pending("old reading", .6, "old evidence", "t0", self.cfg.limits)
        _, result = self.run_output(malformed(f"settle: {pending['id']} | confirmed | invented"))
        self.assertEqual(result.settled, [])
        self.assertEqual([p["id"] for p in self.store.pendings()], [pending["id"]])

    def test_invalid_model_watch_does_not_block_independent_user_evidence(self):
        pending = self.store.add_pending("old reading", .6, "old evidence", "t0", self.cfg.limits)
        verdict = [{"id": pending["id"], "verdict": "returned", "why": "explicit user evidence"}]
        judge = AsyncJudge(LLMJudge(ScriptedLLM([json.dumps(verdict)])), grace=3)
        _, result = self.run_output(malformed(f"settle: {pending['id']} | watch"), judge=judge)
        self.assertEqual([p["outcome"] for p in result.settled], ["CONFIRMED"])
        self.assertFalse(result.validity["valid"])

    def test_duplicate_or_missing_markers_never_apply_tail(self):
        good = make_reply(HEAD, "Hello.", TAIL)
        variants = [good.replace(SECTION_SAY, ""),
                    good + f"\n{SECTION_TAIL}\nsettle: -",
                    "A preface.\n" + good]
        for raw in variants:
            with self.subTest(raw=raw):
                _, result = self.run_output(raw)
                self.assertFalse(result.validity["valid"])
                self.assertIsNone(result.pending_added)
                self.assertEqual(result.relation_changed, {})

    def test_valid_output_still_applies_tail_and_keeps_format_history(self):
        engine, result = self.run_output(make_reply(HEAD, "Hello.", TAIL))
        self.assertTrue(result.validity["valid"])
        self.assertIsNotNone(result.pending_added)
        self.assertTrue(result.relation_changed)
        self.assertIn(SECTION_HEAD, engine.history[-1]["content"])
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))), 1)

    def test_tentative_reading_and_late_exit_remain_unchanged(self):
        say = "我猜你是想把这件事慢慢讲清楚。" + "不用急着回答。" * 8 + "猜错你就说。"
        _, result = self.run_output(make_reply(HEAD, say, "settle: -"))
        self.assertEqual(result.say, say)
        self.assertTrue(result.validity["valid"])
        self.assertTrue(any("完整回复中有退路" in w for w in result.warnings))
        self.assertFalse(any("硬失败" in w for w in result.warnings))

    def test_absent_exit_phrase_is_not_a_semantic_hard_failure(self):
        _, result = self.run_output(make_reply(HEAD, "我猜你今天比较忙？", "settle: -"))
        self.assertTrue(result.validity["valid"])
        self.assertFalse(any("硬失败" in w for w in result.warnings))

    def test_persona_bad_expression_does_not_call_observer(self):
        for streaming in (False, True):
            calls = []

            def tool(messages, *, tool, **kwargs):
                calls.append(tool["function"]["name"])
                self.assertEqual(calls[-1], "raw_reply")
                return {"need": "listen", "confidence": .7, "raw": "neutral", "constraints": []}

            with self.subTest(streaming=streaming):
                engine, result = self.run_output(malformed(), streaming,
                                                 protocol="persona", tool_llm=tool)
                self.assertEqual(calls, ["raw_reply"])
                self.assertFalse(result.validity["valid"])
                self.assertIsNone(result.parsed.tail)
                self.assertEqual(engine.history[-1]["content"], result.say)

    def test_persona_head_say_only_is_a_valid_expression(self):
        calls = []

        def tool(messages, *, tool, **kwargs):
            calls.append(tool["function"]["name"])
            if calls[-1] == "raw_reply":
                return {"need": "listen", "confidence": .7, "raw": "neutral", "constraints": []}
            return {"settle": [], "pending": None}

        output = f"{SECTION_HEAD}\nbaseline: ordinary\ndisplace: none\n{SECTION_SAY}\nHello."
        _, result = self.run_output(output, protocol="persona", tool_llm=tool)
        self.assertEqual(calls, ["raw_reply", "settle_turn"])
        self.assertTrue(result.validity["valid"])


if __name__ == "__main__":
    unittest.main()
