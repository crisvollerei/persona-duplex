"""解析器测试。重点验证「不阻塞」和「声明检测」。

跑法：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona.config import DISPLACE_NONE, PersonaConfig  # noqa: E402
from persona.llm import make_reply                      # noqa: E402
from persona.parse import parse                          # noqa: E402


class TestSectionSplit(unittest.TestCase):
    def test_full_protocol(self):
        raw = make_reply(
            "need: 只想被听\nconfidence: 0.72\n"
            "direct: 你根本不是在问我\ndisplace: 降级",
            "今天是不是特别长啊。",
            "settle: -\npending:\n  claim: 只想被听\n  confidence: 0.7\n  predicts: 会继续说细节",
        )
        t = parse(raw)
        self.assertIsNotNone(t.head)
        self.assertEqual(t.head.need, "只想被听")
        self.assertAlmostEqual(t.head.confidence, 0.72)
        self.assertEqual(t.head.displace, "降级")
        self.assertEqual(t.say, "今天是不是特别长啊。")
        self.assertIsNotNone(t.tail)
        self.assertIsNotNone(t.tail.pending)
        self.assertEqual(t.tail.pending.claim, "只想被听")
        self.assertTrue(t.has_displacement)

    def test_bare_say_still_works(self):
        """模型完全不守协议 —— 也必须能回话。"""
        t = parse("今天是不是特别长啊。")
        self.assertEqual(t.say, "今天是不是特别长啊。")
        self.assertIsNone(t.head)
        self.assertTrue(t.is_pure_say)
        self.assertTrue(t.ok)

    def test_missing_tail(self):
        raw = make_reply("need: 认同\nconfidence: 0.9\ndirect: 好\ndisplace: none",
                         "真好。")
        t = parse(raw)
        self.assertIsNotNone(t.head)
        self.assertIsNone(t.tail)
        self.assertEqual(t.say, "真好。")
        self.assertFalse(t.has_displacement)

    def test_garbage_is_not_fatal(self):
        for junk in ["", "   ", "<<<HEAD>>>", "<<<SAY>>>", "<<<TAIL>>>"]:
            t = parse(junk)          # 不能抛
            self.assertIsInstance(t.say, str)


class TestDisplace(unittest.TestCase):
    def test_normalizes_kind(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: x\ndisplace: 降级（判断→问题）",
                         "y")
        self.assertEqual(parse(raw).head.displace, "降级")

    def test_unknown_kind_falls_back_to_none(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: x\ndisplace: 乱写的", "y")
        self.assertEqual(parse(raw).head.displace, DISPLACE_NONE)

    def test_dash_means_none(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: x\ndisplace: -", "y")
        self.assertEqual(parse(raw).head.displace, DISPLACE_NONE)


class TestDeclaration(unittest.TestCase):
    def test_catches_declaration(self):
        raw = make_reply(
            "need: a\nconfidence: 0.5\ndirect: x\ndisplace: 截断",
            "我本来想说点别的，但是算了。",
        )
        t = parse(raw)
        self.assertTrue(t.declared)
        self.assertGreaterEqual(len(t.declared), 1)

    def test_clean_say_has_no_declaration(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: x\ndisplace: 降级",
                         "你最近挺忙的？")
        self.assertFalse(parse(raw).declared)

    def test_declaration_only_in_say(self):
        """HEAD 里出现这个词不算 —— 那是思考通道，用户看不到。"""
        raw = make_reply(
            "need: a\nconfidence: 0.5\n我本来想直接骂他\ndisplace: 截断",
            "……算了。",
        )
        self.assertFalse(parse(raw).declared)


class TestConsistency(unittest.TestCase):
    def test_claiming_displacement_but_identical_is_exposed(self):
        """声称额外位移，但人格基准和 say 一样。"""
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: 今天很累\nbaseline: 今天真长\ndisplace: 降级",
                         "今天真长")
        self.assertFalse(parse(raw).is_consistent())

    def test_real_displacement_passes(self):
        raw = make_reply(
            "need: a\nconfidence: 0.5\n"
            "direct: 你根本不是在问我，你只是想让人知道你有多累\ndisplace: 降级",
            "今天是不是特别长啊。",
        )
        self.assertTrue(parse(raw).is_consistent())

    def test_missing_baseline_is_unknown_not_inconsistent(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: \ndisplace: 截断", "……算了。")
        self.assertTrue(parse(raw).is_consistent())


class TestConfidenceParsing(unittest.TestCase):
    def test_percent(self):
        raw = make_reply("need: a\nconfidence: 72%\ndirect: x\ndisplace: none", "y")
        self.assertAlmostEqual(parse(raw).head.confidence, 0.72)

    def test_out_of_range_clamped(self):
        raw = make_reply("need: a\nconfidence: 5\ndirect: x\ndisplace: none", "y")
        self.assertAlmostEqual(parse(raw).head.confidence, 1.0)

    def test_missing_becomes_zero(self):
        raw = make_reply("need: a\ndirect: x\ndisplace: none", "y")
        self.assertAlmostEqual(parse(raw).head.confidence, 0.0)


class TestSettleParsing(unittest.TestCase):
    def test_settle_line(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: x\ndisplace: none", "y",
                         "settle: p_0003 | confirmed | 用户又补了一条细节")
        tail = parse(raw).tail
        self.assertEqual(len(tail.settle), 1)
        self.assertEqual(tail.settle[0].pending_id, "p_0003")
        self.assertEqual(tail.settle[0].outcome, "confirmed")

    def test_settle_dash_means_nothing(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: x\ndisplace: none", "y",
                         "settle: -")
        self.assertEqual(parse(raw).tail.settle, [])

    def test_bad_outcome_ignored(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: x\ndisplace: none", "y",
                         "settle: p_1 | 大概吧 | 说不清")
        self.assertEqual(parse(raw).tail.settle, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
