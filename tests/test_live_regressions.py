"""真实 API 跑出来的回归。

这一批问题的共同点：**只有在真实模型 + 连续多轮下才会暴露**。
单元测试里的假数据太规整，20 轮真跑一遍才知道漏了什么。

数据来源：DeepSeek deepseek-chat，每次 20 轮，
按 docs/how-to-start.md 的五种情境设计（需要被听 / 想被推一把 / 换话题 /
绕回来 / 语域突变）。

七次真跑的演进：

    run  缺TAIL  读违规  挂起  结算  堆积  warning均值
    run2   19      2      1     1     0      1.4     ← 修前基线
    run3   10      3     10     9     1      1.8
    run4    0      4     19    13     6      3.4
    run5    0      8     19    11     8      2.8
    run6    2      5     15    12     3      3.2
    run7    0      3     15     7     8      3.6
    run8    0      0     20    17     3      1.4     ← 全部修完

对应到这四条：
  · 历史里不带 TAIL → 模型跟着不写 TAIL（缺 95%）
  · 「猜你」检测只认行首 → 「我猜你」「猜一句」全漏
  · 极短回应被当成换题 → pending 永远关不掉，堆到 8 条
  · 受伤 note 固定字符串 → 第二次受伤被静默吞掉
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.config import PersonaConfig                        # noqa: E402
from persona.engine import TurnEngine                            # noqa: E402
from persona.llm import ScriptedLLM, make_reply                  # noqa: E402
from persona.metrics import compute                              # noqa: E402
from persona.parse import (                                      # noqa: E402
    detect_read_aloud,
    has_exit_offer,
)
from persona.prompt import build_history_entry                   # noqa: E402
from persona.settle import extract_features, judge               # noqa: E402
from persona.store import Store                                  # noqa: E402
from persona.stream import StreamParser                          # noqa: E402
from persona.topicjudge import LLMJudge, JudgeRequest            # noqa: E402


def _cfg() -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


class TestHistoryCarriesTail(unittest.TestCase):
    """历史是**示范**，不是日志。

    第一版只补了 HEAD，20 轮真跑下来缺 TAIL 率 95% —— 因为模型看到
    历史里每轮都没有 TAIL，就跟着不写。历史里的样例比 system prompt
    的规则更有说服力。
    """

    TAIL = "settle: -\npending:\n  claim: 对方会简单应一句\n  confidence: 0.50\n  predicts: 回复很短"

    def test_tail_present_in_history(self):
        from persona.parse import parse
        t = parse(make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                             "嗯。", self.TAIL))
        entry = build_history_entry(t.say, t.head, t.tail)
        self.assertIn("<<<TAIL>>>", entry)
        self.assertIn("pending:", entry)
        self.assertIn("claim:", entry)

    def test_tail_written_even_when_model_dropped_it(self):
        """这一轮模型没写 TAIL —— 历史里也要补骨架。

        不补的话，那个残缺样例会永久留在历史里，之后每轮照着它丢。
        纠正一次的成本远高于一开始就不让坏样例进历史。
        """
        from persona.parse import parse
        raw = make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                         "嗯。")
        t = parse(raw)
        self.assertIsNone(t.tail)
        entry = build_history_entry(t.say, t.head, t.tail)
        self.assertIn("<<<TAIL>>>", entry, "模型丢了 TAIL，历史里也没补")
        self.assertIn("settle: -", entry)

    def test_history_roundtrips(self):
        """存进历史的格式，下次还能解析出三段。"""
        from persona.parse import parse
        raw = make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                         "嗯。", self.TAIL)
        t = parse(raw)
        entry = build_history_entry(t.say, t.head, t.tail)
        again = parse(entry)
        self.assertIsNotNone(again.head)
        self.assertIsNotNone(again.tail)
        self.assertEqual(again.say, "嗯。")


class TestReadAloudVariants(unittest.TestCase):
    """「猜你」有无数种写法。第一版只认行首的「猜」，20 轮里漏了 7 条。

    修的办法不是把规则写全（写不完），而是**按对象判**：
    猜对方 = 违规，猜第三方 = 正常。
    """

    VIOLATIONS = [
        "猜你心里早有答案了。",
        "我猜“感觉还行”这四个字，是你在给自己留后路。",
        "我猜这声叹气后面还有半句。",
        "先猜一个：这问题不是想找答案。",
        "猜一句：这家你心里已经有位置了。",
        "我要猜，你更像是怕自己小题大做。",
        "听起来你不太想说。",
    ]

    OK_CASES = [
        "我猜他也不会来。",
        "猜她大概不会来了。",
        "今天是不是特别长啊。",
        "别猜了，他就是在拖。",
    ]

    def test_all_variants_caught(self):
        for line in self.VIOLATIONS:
            self.assertTrue(detect_read_aloud(line), f"漏判: {line}")

    def test_third_party_not_flagged(self):
        for line in self.OK_CASES:
            self.assertEqual(detect_read_aloud(line), [], f"误报: {line}")

    def test_exit_offer_detected(self):
        """设计文档 §7 的硬线：留了退路是台阶，没留是审判。"""
        self.assertTrue(has_exit_offer("我猜你难受那会儿一个人扛的。猜错你就说。"))
        self.assertTrue(has_exit_offer("猜一句：失眠跟工作有关。不对就当我没说。"))
        self.assertFalse(has_exit_offer("我猜你不是真想问我该不该，是已经想走了。"))

    def test_detection_does_not_rewrite_say(self):
        """解析层不改模型说的话 —— 只统计。

        曾经在输出侧硬删，把「不对就当我没说」连同前面的猜测一起删掉，
        恰好删掉了退路本身。恢复成只观测。
        """
        import persona.parse as P
        self.assertFalse(hasattr(P, "strip_read_aloud"),
                         "strip_read_aloud 应在修歪后移除，只保留统计")


class TestStreamingParagraphPurity(unittest.TestCase):
    """流式只保证「用户看到的 == 账本记的」，不审查措辞。

    早先试过在流式层按句删「猜你」，代价是：
      · 紧跟的退路（「不对就当我没说」）被一起删掉，位移被削平；
      · 按句缓冲让首字延迟从 0.9s 变成一句说完才吐。
    措辞治理归 prompt，不归解析层。
    """

    RAW = ("<<<HEAD>>>need: a\nconfidence: 0.6\ndirect: x\ndisplace: none\n"
           "<<<SAY>>>我猜你不是真想问该不该。那我先问一句：真辞了，你最怕什么。"
           "\n<<<TAIL>>>settle: -")

    def test_visible_matches_final_say(self):
        """用户看到的必须就是账本里记的 —— 否则账本在说谎。"""
        sp = StreamParser()
        out = []
        for ch in self.RAW:
            out.extend(sp.feed(ch))
        parsed = sp.finish()
        self.assertEqual("".join(out).strip(), parsed.say)

    def test_read_aloud_observed_but_not_rewritten(self):
        sp = StreamParser()
        for ch in self.RAW:
            sp.feed(ch)
        parsed = sp.finish()
        self.assertTrue(parsed.read_aloud, "没有观测到「猜」")
        self.assertIn("我猜你", parsed.say, "解析层不该改模型说的话")

    def test_normal_text_still_streams(self):
        raw = make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                         "今天是不是特别长啊。", "settle: -")
        sp = StreamParser()
        out = []
        for ch in raw:
            out.extend(sp.feed(ch))
        self.assertEqual("".join(out), "今天是不是特别长啊。")


class TestMinimalReplyHandling(unittest.TestCase):
    """「嗯」「算了」「唉」不是换题，是不表态。

    把它们当换题的后果：pending 永远关不掉，十几轮后堆到 8 条，
    判定器每轮判 8 句，warning 爆炸，而真正该学的一条都没学到。
    """

    def test_minimal_reply_classified_as_watch_not_left(self):
        feat = extract_features("嗯。", pending_claim="c", predicts="p",
                                prev_ai_say="你是想听钱的说法还是实在的")
        v = judge(feat, baseline_formality=0.25)
        self.assertEqual(v.kind, "WATCH")
        self.assertIn("短回复", v.reason)
        # 关键区别：走的是「不表态」分支，不是「换题」分支
        self.assertNotIn("可能只是不关注", v.reason)

    def test_two_consecutive_minimal_replies_remain_unknown(self):
        """连续短回复也不构成反驳。"""
        feat = extract_features("嗯。", pending_claim="c", predicts="p",
                                prev_ai_say="x", minimal_streak=1)
        v = judge(feat, baseline_formality=0.25)
        self.assertEqual(v.kind, "WATCH")
        self.assertFalse(v.settle)

    def test_content_reply_not_treated_as_minimal(self):
        """带内容的短句不算 —— 「算了不说了。你觉得我该不该换个工作」有真问题。"""
        feat = extract_features("算了不说了。你觉得我该不该换个工作。",
                                pending_claim="c", predicts="p")
        self.assertFalse(feat.text.strip() in ("嗯。",))

    def test_short_replies_do_not_force_pending_closed(self):
        """引擎层保留未知，队列由上限和 TTL 限制而非伪造结论。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            replies = [
                make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                           "嗯。", "settle: -\npending:\n  claim: c\n"
                                   "  confidence: 0.6\n  predicts: p"),
            ] + [make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none",
                            "嗯。", "settle: -")] * 4
            e = TurnEngine(_cfg(), store, ScriptedLLM(replies))
            e.turn("今天好累")
            self.assertEqual(len(store.pendings()), 1)
            e.turn("嗯。")
            e.turn("嗯。")
            self.assertEqual(len(store.pendings()), 1)
            self.assertFalse([r for r in store.read_ledger() if r["kind"] == "pending_settle"])


class TestRepairNoteUniqueness(unittest.TestCase):
    """受伤是「用户」的状态，不是「预言」的状态。

    两个方向都要守住：
      · 同一轮里多条预言同时受伤 → 只能开一条修复（用户只被伤了一次）
      · 不同轮次受伤 → 必须各开一条（否则第二次受伤被静默吞掉）
    """

    def test_same_turn_opens_one_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            e = TurnEngine(_cfg(), store, ScriptedLLM([]))
            e._turn_seq = 7
            for pid in ("p_0001", "p_0002", "p_0003"):
                e._apply_outcome({"id": pid, "confidence": 0.7}, "SUSPECTED_HURT")
            items = store.load_view("repair", {}).get("items", [])
            self.assertEqual(len(items), 1,
                             "一轮里多条预言受伤，开了多条修复")

    def test_different_turns_open_separate_repairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            e = TurnEngine(_cfg(), store, ScriptedLLM([]))
            e._turn_seq = 7
            e._apply_outcome({"id": "p_0001", "confidence": 0.7}, "SUSPECTED_HURT")
            e._probe = {}
            e._turn_seq = 12
            e._apply_outcome({"id": "p_0002", "confidence": 0.7}, "SUSPECTED_HURT")
            items = store.load_view("repair", {}).get("items", [])
            self.assertEqual(len(items), 2, "第二次受伤被当成重复吞掉了")
            self.assertNotEqual(items[0]["note"], items[1]["note"])


class TestJudgeTokenBudget(unittest.TestCase):
    """max_tokens=120 只够判 2-3 条；挂 8 条时 JSON 被截断，整批失败。"""

    def test_budget_scales_with_pendings(self):
        j = LLMJudge(lambda messages, **kw: "[]")
        self.assertLess(j._budget(1), j._budget(8))
        self.assertGreaterEqual(j._budget(8), 400)

    def test_llm_receives_budget(self):
        seen = {}

        def fake(messages, *, max_tokens=None):
            seen["max_tokens"] = max_tokens
            return '[{"id":"p_0001","verdict":"unclear","why":"x"}]'

        j = LLMJudge(fake)
        req = JudgeRequest(user_text="嗯", pendings=[
            {"id": "p_0001", "claim": "c", "predicts": "p"}])
        j.judge(req)
        self.assertIsNotNone(seen.get("max_tokens"))

    def test_plain_callable_still_works(self):
        """测试和自定义判定器常常只接受 messages 一个参数。"""
        j = LLMJudge(lambda messages: '[{"id":"p_0001","verdict":"unclear"}]')
        req = JudgeRequest(user_text="嗯", pendings=[
            {"id": "p_0001", "claim": "c", "predicts": "p"}])
        self.assertEqual(j.judge(req)["p_0001"].verdict, "unclear")


class TestWatchesDeduped(unittest.TestCase):
    def test_same_pending_counted_once(self):
        ledger = [
            {"kind": "turn", "probe": {"settle": [
                {"pid": "p_0001", "settle": False},
                {"pid": "p_0001", "settle": False},
                {"pid": "p_0002", "settle": False},
            ]}},
        ]
        m = compute(ledger)
        self.assertEqual(len(m.watched_ids), 2, "同一条预言被数了多次")
        self.assertEqual(m.watched_ids, {"p_0001", "p_0002"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
