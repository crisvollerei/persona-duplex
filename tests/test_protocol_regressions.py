"""协议层审查发现的缺陷 —— 回归测试。

这一批的核心是同一件事：**DIRECT 永不进入输出**。
实测有四种漏法，每一种都单独覆盖：
  1. 完全没标记，只写四个字段
  2. 标记写错（少一个 >、小写、空格）
  3. 流式路径（engine 默认走这条）
  4. clean_phase2 的兜底
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.parse import parse, safe_say                    # noqa: E402
from persona.protocol import (                                 # noqa: E402
    StreamCleanPhase2,
    clean_phase2,
    parse_tail,
    tail_tool,
)
from persona.stream import StreamParser                        # noqa: E402

SECRET = "你其实早就想辞职了，只是在等一个人说可以。"


def _stream(raw: str):
    sp = StreamParser()
    out = []
    for ch in raw:
        out += sp.feed(ch)
    sp.finish()
    return sp.say_text


class TestDirectNeverLeaks(unittest.TestCase):
    """四种漏法都要堵住。"""

    def test_bare_fields_without_markers(self):
        """模型完全忘了写标记，只把四个字段列出来。"""
        raw = ("need: x\nconfidence: 0.7\ndirect: " + SECRET +
               "\ndisplace: 降级\n\n你都想过哪些了。")
        self.assertNotIn(SECRET, parse(raw).say)

    def test_malformed_marker(self):
        """标记写错 —— 少一个 >。"""
        raw = ("<<<HEAD>>\nneed: x\ndirect: " + SECRET +
               "\ndisplace: none\n<<<SAY>>>\n回复。")
        self.assertNotIn(SECRET, parse(raw).say)

    def test_stream_path_also_blocked(self):
        """流式是 engine 的默认路径 —— 这一侧漏了等于没修。"""
        raw = ("need: x\nconfidence: 0.7\ndirect: " + SECRET +
               "\ndisplace: 降级\n<<<SAY>>>\n你都想过哪些了。")
        self.assertNotIn(SECRET, _stream(raw))

    def test_clean_phase2_direct_before_marker(self):
        self.assertNotIn(SECRET, clean_phase2("direct: " + SECRET + "\n<<<SAY>>>"))

    def test_clean_phase2_scaffold_echo(self):
        """模型把 prompt 里的脚手架回显出来。"""
        got = clean_phase2("（内心：" + SECRET + "／位移方式：降级）你都想过哪些了。")
        self.assertNotIn(SECRET, got)

    def test_clean_phase2_head_fragment_after_marker(self):
        got = clean_phase2("你都想过哪些了。<<<HEAD>>>" + SECRET)
        self.assertNotIn(SECRET, got)

    def test_normal_say_passes(self):
        """正常回复不能被误杀。"""
        for s in ("你都想过哪些了。", "嗯。", "Then what happened?"):
            self.assertEqual(safe_say(s), s)

    def test_special_token_stripped(self):
        """模型漏出的特殊 token 不能给用户看到。

        实测 DeepSeek 会在回复末尾带 "<｜end▁of▁sentence｜>"。
        """
        got = safe_say("想好了，也跟我说一声。<｜end▁of▁sentence｜>")
        self.assertEqual(got, "想好了，也跟我说一声。")
        self.assertEqual(safe_say("hi<|endoftext|>"), "hi")


class TestStreamCleanPhase2(unittest.TestCase):
    """逐字流式清洗不能吞字。"""

    @staticmethod
    def _run(s: str) -> str:
        sp = StreamCleanPhase2()
        out = []
        for ch in s:
            out += sp.feed(ch)
        out += sp.finish()
        return "".join(out)

    def test_english_not_swallowed(self):
        """英文前缀会撞上 settle/delta 等字段名的首字母。

        实测 "yes" 只输出 "ye"、"Sounds good" 只输出 "Sounds goo"，
        中文正常 —— 所以中文测试抓不到。
        """
        for s in ("Sounds good", "OK then", "yes",
                  "Need to get some rest.", "Direct hit"):
            self.assertEqual(self._run(s), s, "吞字了：" + s)

    def test_chinese_not_swallowed(self):
        for s in ("你好。", "好的。", "嗯。"):
            self.assertEqual(self._run(s), s)


class TestTailParsing(unittest.TestCase):
    def test_multi_line_settle(self):
        """一行一条 settle，不能只读第一条。"""
        raw = ("x\n<<<TAIL>>>\nsettle: p_0001 | confirmed | a\n"
               "settle: p_0002 | refuted | b\n")
        got = parse(raw).tail.settle
        self.assertEqual(len(got), 2)
        self.assertEqual([g.pending_id for g in got], ["p_0001", "p_0002"])

    def test_watch_preserved(self):
        """模型明确判 watch 要保留，不能被丢掉退回字面判定。"""
        raw = "x\n<<<TAIL>>>\nsettle: p_0001 | watch | 只是附和\n"
        tail = parse(raw).tail
        self.assertEqual(tail.watch, ["p_0001"])
        self.assertEqual(tail.settle, [])

    def test_tool_tail_has_delta_channel(self):
        """tool 协议没有 delta 通道时，关系账本永远是初始值。"""
        tool = tail_tool([{"id": "p_0001", "claim": "c", "predicts": "p"}])
        props = tool["function"]["parameters"]["properties"]
        self.assertIn("delta", props)

    def test_tool_delta_parsed(self):
        tail = parse_tail({"settle": [], "pending": None,
                           "delta": {"closeness": 0.05, "unresolved": ["累"]}})
        self.assertAlmostEqual(tail.delta["closeness"], 0.05)
        self.assertEqual(tail.delta["unresolved"], ["累"])


class TestBufferNotLost(unittest.TestCase):
    """drain 之后处理失败，事件不能丢。"""

    def test_restore_on_error(self):
        from persona.config import PersonaConfig
        from persona.consolidate import Consolidator
        from persona.store import Store

        tmp = tempfile.TemporaryDirectory()
        try:
            st = Store(Path(tmp.name))
            st.init_traits(PersonaConfig.load(
                ROOT / "personas" / "shorekeeper.toml").role.traits_state())
            for i in range(3):
                st.push_buffer({"turn_id": "t%d" % i, "user": "x", "say": "y",
                                "direct": "z", "need": "只想被听",
                                "displace": "none"})
            st.add_judgment = lambda *a, **k: (_ for _ in ()).throw(
                OSError("disk full"))
            with self.assertRaises(OSError):
                Consolidator(st).run_once()
            self.assertEqual(len(list(st.buffer_dir.glob("*.json"))), 3,
                             "处理失败后事件被丢了")
        finally:
            tmp.cleanup()

    def test_commit_on_success(self):
        from persona.config import PersonaConfig
        from persona.consolidate import Consolidator
        from persona.store import Store

        tmp = tempfile.TemporaryDirectory()
        try:
            st = Store(Path(tmp.name))
            st.init_traits(PersonaConfig.load(
                ROOT / "personas" / "shorekeeper.toml").role.traits_state())
            for i in range(2):
                st.push_buffer({"turn_id": "t%d" % i, "user": "x", "say": "y",
                                "direct": "z", "need": "只想被听",
                                "displace": "none"})
            Consolidator(st).run_once()
            self.assertEqual(len(list(st.buffer_dir.glob("*.json"))), 0)
            self.assertEqual(len(list(st.processing_dir.glob("*.json"))), 0)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)