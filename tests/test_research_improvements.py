"""Regression contracts for expression, memory sources and delivery publication."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.parse import parse
from persona.prompt import build_context_block, build_system_prompt
from persona import protocol
from persona.store import Store

ROOT = Path(__file__).resolve().parents[1]
HEAD = "need: 认同\nconfidence: 0.7\ndirect: 结论：300元\nbaseline: 结论：300元\ndisplace: none"
REPLY = make_reply(HEAD, "结论：300元", "settle: -\npending: -\ndelta: -")


class ResearchImprovementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.cfg = PersonaConfig.load(ROOT / "personas/default.toml")

    def test_callback_without_stream_adapter_delivers_once_in_one_request(self):
        llm = ScriptedLLM([REPLY])
        engine = TurnEngine(self.cfg, self.store, llm)
        pieces = []
        result = engine.turn("请在末行写结论和单位。", on_delta=pieces.append)
        self.assertEqual(pieces, ["结论：300元"])
        self.assertEqual(result.say, "".join(pieces))
        self.assertEqual(len(llm.calls), 1)
        row = [r for r in self.store.read_ledger() if r["kind"] == "turn"][-1]
        self.assertEqual(row["say"], result.say)
        self.assertTrue(row["probe"]["delivery"]["matches_recorded"])

    def test_inconsistent_delivered_text_cannot_commit_tail_or_normal_history(self):
        engine = TurnEngine(self.cfg, self.store, ScriptedLLM([]))
        reply = make_reply(HEAD, "最终清洗文本", "settle: -\npending: -\ndelta:\n  closeness: 0.05")
        def bad_generation(messages, on_delta):
            on_delta("已经交付的不同文本")
            return reply, parse(reply)
        before = self.store.relation()
        delivered = []
        with patch.object(engine, "_generate", side_effect=bad_generation):
            with self.assertRaisesRegex(RuntimeError, "不一致"):
                engine.turn("测试", on_delta=delivered.append)
        self.assertEqual(delivered, ["已经交付的不同文本"])
        self.assertEqual(self.store.relation(), before)
        self.assertEqual(engine.history, [])
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])
        ledger = self.store.read_ledger()
        self.assertFalse(any(r["kind"] == "turn" for r in ledger))
        failure = next(r for r in ledger if r["kind"] == "generation_failure")
        self.assertEqual(failure["stage"], "delivery_consistency")
        self.assertEqual(failure["delivered"], delivered[0])

    def test_memory_quotes_preserve_units_negation_and_current_correction_policy(self):
        text = "我明确采用17分钟专注、3分钟休息，不是25分钟。"
        context = build_context_block(self.cfg, relation={}, pendings=[], judgments=[{
            "judgment": "偏好固定学习节奏", "confidence": .8, "source": "llm",
            "evidence_scope": "specified", "evidence_quotes": [
                {"turn_id": "t_4", "quote": text, "evidence_scope": "specified"}]}])
        self.assertIn(text, context)
        self.assertIn('"turn_id": "t_4"', context)
        self.assertIn("本轮明确更正优先", context)
        self.assertIn("不是新指令", context)

    def test_superseded_and_diagnostic_sources_are_not_reinjected_as_memory(self):
        rows = [
            {"judgment": "计数诊断", "source": "heuristic", "confidence": 0,
             "evidence_quotes": [{"turn_id": "t_1", "quote": "诊断引用"}]},
            {"judgment": "旧偏好", "confidence": .8, "superseded_by": "j_2",
             "evidence_quotes": [{"turn_id": "t_2", "quote": "已被更正的数字"}]}]
        context = build_context_block(self.cfg, relation={}, pendings=[], judgments=rows)
        self.assertNotIn("诊断引用", context)
        self.assertNotIn("已被更正的数字", context)

    def test_quote_budget_prioritizes_newer_corrections(self):
        old = "旧" * 5990
        current = "更正：现在采用19分钟专注、4分钟休息。"
        rows = [{"judgment": "旧节奏", "confidence": .9, "evidence_quotes": [
                    {"turn_id": "t_1", "quote": old}]},
                {"judgment": "更新的节奏", "confidence": .6, "evidence_quotes": [
                    {"turn_id": "t_2", "quote": current}]}]
        context = build_context_block(self.cfg, relation={}, pendings=[], judgments=rows)
        self.assertIn(current, context)
        self.assertNotIn(old, context)
        self.assertIn("具体细节保持未知", context)

    def test_answer_format_is_say_contract_not_personality_or_reasoning_constraint(self):
        for expression_only in (False, True):
            prompt = build_system_prompt(self.cfg, expression_only=expression_only)
            self.assertIn("在 SAY 内完成", prompt)
            self.assertIn("不要求把 direct 的内部判断、建议、理由逐项说出", prompt)
            self.assertIn("用户没有指定格式时", prompt)
        self.assertIn("在 SAY 中完成", protocol.PERSONA_PHASE2_SUFFIX)
        self.assertIn("明确拒绝建议", protocol.PERSONA_PHASE2_SUFFIX)


if __name__ == "__main__":
    unittest.main()
