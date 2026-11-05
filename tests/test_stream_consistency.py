"""Delivery remains faithful when ordinary brackets resemble model controls."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.parse import safe_say, strip_special_tokens
from persona.store import Store
from persona.stream import StreamParser


ROOT = Path(__file__).resolve().parents[1]


def reply(say):
    return make_reply(
        "need: 核对\nconfidence: 0.8\ndirect: B公司在苏州。\n"
        "baseline: B公司在苏州。\ndisplace: none",
        say, "settle: -\npending: -\ndelta: -")


def partitionings(text):
    yield [text]
    yield list(text)
    for cut in range(len(text) + 1):
        yield [text[:cut], text[cut:]]
    rng = random.Random(7705)
    for _ in range(12):
        cuts = sorted({0, len(text), *(rng.randrange(len(text)+1) for _ in range(12))})
        yield [text[a:b] for a, b in zip(cuts, cuts[1:])]


class TestStreamConsistency(unittest.TestCase):
    def assert_delivery(self, chunks, expected, blocked=False):
        parser = StreamParser()
        seen = []
        for chunk in chunks:
            seen.extend(parser.feed(chunk))
        parsed = parser.finish()
        delivered = "".join(seen)
        self.assertTrue(parsed.say.startswith(delivered), (delivered, parsed.say, chunks))
        delivered += parsed.say[len(delivered):]
        self.assertEqual(delivered, expected)
        self.assertEqual(parsed.say, expected)
        self.assertEqual(parser.output_blocked, blocked)

    def test_normal_angle_brackets_html_and_math_are_preserved(self):
        for text in ("B公司在苏州。\n<苏州>", "结论：<1152>元。", "<00:25>",
                     "x < 3 且 y > 2", "<span>正常 HTML</span>", "<s>旧价格</s>",
                     "<head>", "<HEAD>", "<tail>", "[INSTAGRAM]",
                     "用 <value> 作为占位。", "[重要] 不要改掉正常的括号。"):
            self.assertEqual(safe_say(text), text)
            for chunks in partitionings(reply(text)):
                with self.subTest(text=text, chunks=len(chunks)):
                    self.assert_delivery(chunks, text)

    def test_real_control_tokens_are_blocked_before_any_delivery(self):
        for token in ("<|endoftext|>", "<｜end▁of▁sentence｜>", "<end_of_turn>",
                      "<end_of_text>", "<start_of_turn>", "<eot_id>",
                      "<eos_token>", "[INST]", "[/INST]"):
            self.assertEqual(strip_special_tokens("可见。" + token), "可见。")
            for chunks in partitionings(reply("可见。" + token + "私有后缀")):
                with self.subTest(token=token, chunks=len(chunks)):
                    self.assert_delivery(chunks, "可见。", blocked=True)

    def test_persona_expression_preserves_angle_answer_all_splits(self):
        say = "按最后更正，B公司在苏州。\n<苏州>"
        raw = "<<<HEAD>>>\nbaseline: B公司在苏州。\ndisplace: none\n<<<SAY>>>\n" + say
        for chunks in partitionings(raw):
            parser = StreamParser(expression_only=True); seen = []
            for chunk in chunks: seen.extend(parser.feed(chunk))
            parsed = parser.finish()
            visible = "".join(seen)
            self.assertTrue(parsed.say.startswith(visible))
            self.assertEqual(parsed.say, say)

    def test_internal_protocol_and_scaffold_stay_behind_the_boundary(self):
        for suffix in ("\ndirect: 私有内容", "（内心：私有内容）", "<< ^ HEAD >>私有内容"):
            for chunks in partitionings(reply("可见。" + suffix)):
                with self.subTest(suffix=suffix, chunks=len(chunks)):
                    parser = StreamParser(); seen = []
                    for chunk in chunks: seen.extend(parser.feed(chunk))
                    parsed = parser.finish()
                    self.assertNotIn("私有", "".join(seen))
                    self.assertEqual(parsed.say, "可见。")
                    self.assertTrue(parsed.say.startswith("".join(seen)))

    def test_ordinary_inequality_cannot_hide_a_partial_internal_scaffold(self):
        for chunks in partitionings(reply("x < 3（判断：私有内容）不应交付的后缀")):
            self.assert_delivery(chunks, "x < 3", blocked=True)

    def test_recorded_city_counterexample_keeps_the_delivered_answer(self):
        text = "按你最后的更正，B公司现在在苏州。成都那条已作废。\n<苏州>"
        raw = reply(text)
        with tempfile.TemporaryDirectory() as directory:
            cfg = PersonaConfig.load(ROOT / "personas/default.toml")
            engine = TurnEngine(cfg, Store(directory),
                                ScriptedLLM([]), stream_llm=lambda messages: iter(raw))
            seen = []; result = engine.turn("B公司在哪？", on_delta=seen.append)
            self.assertEqual(result.say, text)
            self.assertEqual(result.say, "".join(seen))
            self.assertTrue(result.validity["valid"])
            saved = [e for e in engine.store.read_ledger() if e["kind"] == "turn"][-1]
            self.assertEqual(saved["say"], result.say)


if __name__ == "__main__":
    unittest.main()
