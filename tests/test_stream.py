"""流式解析测试。

核心风险：把标记的前缀当正文吐出去。用户会看到「……好<<<TA」。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.llm import make_reply          # noqa: E402
from persona.stream import StreamParser     # noqa: E402


def run(chunks):
    sp = StreamParser()
    out = []
    for c in chunks:
        out.extend(sp.feed(c))
    return sp, out


class TestNoMarkerLeak(unittest.TestCase):
    def test_char_by_char(self):
        """逐字喂 —— 最狠的切法，不能漏一个标记字符。"""
        raw = make_reply(
            "need: 只想被听\nconfidence: 0.7\ndirect: 你只是想让人知道你累\ndisplace: 降级",
            "今天是不是特别长啊。",
            "settle: -",
        )
        sp, out = run(list(raw))
        joined = "".join(out)
        self.assertEqual(joined, "今天是不是特别长啊。")
        self.assertNotIn("<", joined)
        self.assertNotIn("TAIL", joined)

    def test_partial_marker_held_back(self):
        """缓冲区末尾是标记前半截时，必须扣住不吐。"""
        sp = StreamParser()
        out = []
        out.extend(sp.feed("<<<SAY>>>你好"))
        out.extend(sp.feed("<<<TA"))          # 半个标记
        self.assertEqual("".join(out), "你好", "半个标记被吐出去了")
        out.extend(sp.feed("IL>>>"))
        self.assertEqual("".join(out), "你好")

    def test_finish_returns_tail(self):
        raw = make_reply("need: a\nconfidence: 0.5\ndirect: x\ndisplace: none",
                         "嗯。", "settle: -\npending:\n  claim: c\n  confidence: 0.6\n  predicts: p")
        sp, _ = run(list(raw))
        parsed = sp.finish()
        self.assertEqual(parsed.say, "嗯。")
        self.assertIsNotNone(parsed.head)
        self.assertIsNotNone(parsed.tail)
        self.assertIsNotNone(parsed.tail.pending)


class TestDegenerate(unittest.TestCase):
    def test_bare_text_becomes_say(self):
        """模型完全不按协议 —— 整段当回复。"""
        sp, out = run(["今天挺累的吧。"])
        self.assertEqual("".join(out), "今天挺累的吧。")
        self.assertEqual(sp.finish().say, "今天挺累的吧。")

    def test_empty_chunks(self):
        sp, out = run(["", "", ""])
        self.assertEqual(out, [])

    def test_head_only(self):
        sp, out = run(list("<<<HEAD>>>need: a\nconfidence: 0.5\ndirect: x\ndisplace: none"))
        parsed = sp.finish()
        self.assertIsNotNone(parsed.head)
        self.assertEqual(parsed.say, "")

    def test_no_say_section(self):
        """HEAD 直接接 TAIL，SAY 缺失 —— 不能把 TAIL 内容当回复。"""
        sp, out = run(list("<<<HEAD>>>need: a\ndisplace: none\n<<<TAIL>>>settle: -"))
        parsed = sp.finish()
        self.assertNotIn("settle", parsed.say)

    def test_streaming_matches_batch(self):
        """流式解析和整段解析结果必须一致。"""
        from persona.parse import parse

        raw = make_reply(
            "need: 要被推一把\nconfidence: 0.62\ndirect: 你早就想辞了\ndisplace: 悬置",
            "那你是想听我说辞了吧？",
            "settle: p_0001 | refuted | 用户直接问怎么办",
        )
        sp, out = run([raw[i:i + 7] for i in range(0, len(raw), 7)])
        a = sp.finish()
        b = parse(raw)
        self.assertEqual(a.say, b.say)
        self.assertEqual(a.head.need, b.head.need)
        self.assertEqual(a.head.displace, b.head.displace)
        self.assertEqual(len(a.tail.settle), len(b.tail.settle))


if __name__ == "__main__":
    unittest.main(verbosity=2)
