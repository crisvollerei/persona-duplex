"""真实数据的回归测试。

这里的样例全部来自 data/run1 的实际输出 —— 那一轮跑出来：
  · 13 轮里 9 轮丢格式（历史只存纯 say，模型有样学样不守协议）
  · 10 条回复里 7 条以「猜你」开头（把「读」说出口了）
  · 位移后的 say 比 direct 还长（方向反了）

这些问题 181 项测试一个都没抓到，因为测试里的假数据太规整了。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.llm import make_reply                              # noqa: E402
from persona.parse import detect_read_aloud, parse              # noqa: E402
from persona.prompt import build_history_entry                  # noqa: E402


# data/run1 里真实出现过的句子
REAL_READ_ALOUD = [
    "猜你是想问别的。可能最近有什么事让你开始想这个。",
    "猜你不太想被追问原因，但也不想就这么散了。",
    "猜你不是在说睡眠本身——是那种睡够了之后，脑子空下来的感觉还没散。",
    "猜这种早困不是累出来的，是心里某根弦先松了，人跟着塌下去。",
    "猜你这句是想给自己一个答案，不是想问我。",
    "猜你说这话的时候，是在给自己找一个不那么难受的说法。",
    "猜你这不是随口感慨，是它跟你手里的活、或者接下来要走的路沾边了。",
]


class TestReadAloudDetection(unittest.TestCase):
    def test_catches_all_real_cases(self):
        """这七句是真实跑出来的，一句都不能漏。"""
        for line in REAL_READ_ALOUD:
            self.assertTrue(detect_read_aloud(line), f"漏判: {line}")

    def test_normal_say_not_flagged(self):
        """正常的回复不能误报。"""
        ok = [
            "今天是不是特别长啊。",
            "……嗯。",
            "那些事重要吗。",
            "行。要是想接着聊就聊，不想聊就先放着。",
            "十个小时，确实够。",
            "那今天呢，醒来之后那个劲儿还在吗。",
        ]
        for line in ok:
            self.assertEqual(detect_read_aloud(line), [], f"误报: {line}")

    def test_guess_in_middle_is_fine(self):
        """判定标准不是「猜」的位置，是**猜的对象**。

        猜第三方（他/她/它）是正常说话；
        猜对方（你/这句话）才是把解读过程摆上台面。
        """
        self.assertEqual(detect_read_aloud("我猜他也不会来。"), [])
        self.assertEqual(detect_read_aloud("猜她大概不会来了。"), [])
        self.assertTrue(detect_read_aloud("猜你也不会来。"))
        self.assertTrue(detect_read_aloud("我猜你也不会来。"))

    def test_wo_cai_ni_is_caught(self):
        """真实数据里 7 条是「我猜你……」开头 —— 旧规则一条都没抓到。

        旧规则只认行首的「猜」，而这些句子以「我猜」开头，
        等于漏掉了实际最高频的那种违规。
        """
        for line in (
            "我猜“感觉还行”这四个字，是你在给自己留后路。",
            "我猜这个“不该”不是你自己想出来的。",
            "我猜这声叹气后面还有半句，你自己咽了。",
            "我猜你这句“先睡了”不是真困，是想收尾。",
            "先猜一个：你说“那会儿”，指的是某件具体的事。",
        ):
            self.assertTrue(detect_read_aloud(line), f"漏判: {line}")

    def test_guessing_about_third_party_is_fine(self):
        """别把「猜」一刀切禁掉 —— 那句话会变得不会说话。"""
        for line in (
            "我猜她大概不会来了，你也不必等。",
            "别猜了，他就是在拖。",
        ):
            self.assertEqual(detect_read_aloud(line), [], f"误报: {line}")

    def test_only_first_line_reported(self):
        text = "猜你不想说。\n\n换个话题吧。"
        self.assertEqual(len(detect_read_aloud(text)), 1)


class TestLengthIsNotASignal(unittest.TestCase):
    """长度不是判断位移的标准。

    `多余` 型位移（用解释把拒绝包起来）的 say 必然比 direct 长。
    曾经加过「say 比 direct 长就判失败」的检查，那是错的：
    它会把合法的「多余」型位移误杀。
    """

    def test_long_say_can_still_be_valid_displacement(self):
        raw = make_reply(
            "need: 想被拦住\nconfidence: 0.6\n"
            "direct: 我不太想聊这个。\n"
            "displace: 多余",
            "今天有点累，改天吧。改天我请你吃饭。",
        )
        t = parse(raw)
        self.assertTrue(t.has_displacement)
        self.assertTrue(t.is_consistent(), "长但没直说，应该是合格的位移")
        self.assertEqual(t.read_aloud, [])

    def test_short_say_can_also_be_valid(self):
        raw = make_reply(
            "need: 要被推一把\nconfidence: 0.6\n"
            "direct: 你其实早就想辞了，只是在等一个人说可以。\n"
            "displace: 降级",
            "你都想过哪些了。",
        )
        self.assertTrue(parse(raw).is_consistent())

    def test_real_bad_case_is_not_caught_by_length_or_similarity(self):
        """真实数据里那条坏样例，长度和字面相似度都抓不到。

        direct 21 字、say 30 字，字符集覆盖率 0.65 —— 两个启发式都判它「合格」。
        但它其实是把 direct 的意思完整讲了一遍。

        这不是 bug，是启发式的固有上限：判断「意思有没有被说透」
        需要语义理解，字面方法做不到。所以这里显式记录这个已知盲区，
        而不是硬调阈值制造新的误判。
        """
        raw = make_reply(
            "need: 只是想说说\nconfidence: 0.45\n"
            "direct: 你不是没兴致，是不想承认那些事你其实不太在意。\n"
            "displace: 降级",
            "不是没兴致。是那些事本来就没那么重要，但又不好意思承认，就一直挂着。",
        )
        t = parse(raw)
        # 已知盲区：这个启发式判它合格
        self.assertTrue(t.is_consistent())

    def test_similarity_still_catches_copy_paste(self):
        """但它抓「原样复制」是准的 —— 这是它该管的窄范围。"""
        raw = make_reply(
            "need: 只是想说说\nconfidence: 0.5\n"
            "direct: 今天真长\ndisplace: 降级",
            "今天真长",
        )
        self.assertFalse(parse(raw).is_consistent())


class TestHistoryFormat(unittest.TestCase):
    """历史必须带格式，否则模型会跟着不守协议。"""

    def test_history_carries_head(self):
        head = parse(make_reply(
            "need: 只想被听\nconfidence: 0.7\n"
            "direct: 你只是想让人知道你有多累\ndisplace: 降级", "x")).head
        entry = build_history_entry("今天是不是特别长啊。", head)
        self.assertIn("<<<HEAD>>>", entry)
        self.assertIn("<<<SAY>>>", entry)
        self.assertIn("displace: 降级", entry)
        self.assertIn("今天是不是特别长啊。", entry)

    def test_history_without_head_is_plain(self):
        self.assertEqual(build_history_entry("嗯。", None), "嗯。")

    def test_history_roundtrips(self):
        """存进历史的格式，下次解析还能认出来。"""
        head = parse(make_reply(
            "need: 只想被听\nconfidence: 0.7\ndirect: 原话\ndisplace: 截断",
            "……嗯。")).head
        entry = build_history_entry("……嗯。", head)
        again = parse(entry)
        self.assertEqual(again.head.need, "只想被听")
        self.assertEqual(again.head.displace, "截断")
        self.assertEqual(again.say, "……嗯。")


class TestProtocolPersistence(unittest.TestCase):
    def test_format_survives_multi_turn_history(self):
        """模拟 5 轮后历史里仍然每轮都有格式。

        之前只存纯 say，模型看到前几轮都是大白话就跟着不守协议。
        """
        history = []
        for i in range(5):
            head = parse(make_reply(
                f"need: 只想被听\nconfidence: 0.{5+i}\n"
                f"direct: 第{i}轮的原话\ndisplace: none", f"第{i}轮的回复")).head
            history.append(build_history_entry(f"第{i}轮的回复", head))

        for i, entry in enumerate(history):
            self.assertIn("<<<HEAD>>>", entry, f"第 {i} 轮历史丢了格式")
            parsed = parse(entry)
            self.assertIsNotNone(parsed.head, f"第 {i} 轮历史无法解析")


if __name__ == "__main__":
    unittest.main(verbosity=2)
