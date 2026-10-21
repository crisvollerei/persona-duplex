"""会话与后台线程测试。

重点：后台必须真的跑起来，退出时必须把积压清掉。
慢层一直是空的，那「成长」就不存在。
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.config import PersonaConfig                       # noqa: E402
from persona.llm import ScriptedLLM, make_reply                # noqa: E402
from persona.session import Session, SessionConfig             # noqa: E402
from persona.store import Store                                # noqa: E402


def _cfg() -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


def _reply(need="只想被听", say="嗯。"):
    return make_reply(f"need: {need}\nconfidence: 0.6\ndirect: x\ndisplace: none", say)


class TestSessionLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_say_returns_result(self):
        with Session(_cfg(), self.store, ScriptedLLM([_reply()]),
                     options=SessionConfig(consolidate_interval=999)) as s:
            res = s.say("今天好累")
        self.assertEqual(res.say, "嗯。")

    def test_close_is_idempotent(self):
        s = Session(_cfg(), self.store, ScriptedLLM([_reply()]),
                    options=SessionConfig(consolidate_interval=999))
        s.close()
        s.close()          # 不能抛

    def test_closed_session_refuses(self):
        s = Session(_cfg(), self.store, ScriptedLLM([_reply()]),
                    options=SessionConfig(consolidate_interval=999))
        s.close()
        with self.assertRaises(RuntimeError):
            s.say("hi")

    def test_exit_consolidates_backlog(self):
        """退出时必须把积压清掉 —— 否则慢层永远是空的。"""
        replies = [_reply("只想被听"), _reply("只想被听")]
        with Session(_cfg(), self.store, ScriptedLLM(replies),
                     options=SessionConfig(consolidate_interval=999,
                                           consolidate_on_exit=True)) as s:
            s.say("今天好累")
            s.say("路上还堵了")
            self.assertEqual(list(self.store.buffer_dir.glob("*.json")) != [], True,
                             "buffer 里应该有积压")
        # 退出后
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [],
                         "退出后 buffer 应该清空")
        self.assertTrue((self.store.root / "self-story.md").exists(),
                        "退出后应该写出自我叙事")

    def test_no_consolidate_on_exit_when_disabled(self):
        with Session(_cfg(), self.store, ScriptedLLM([_reply()]),
                     options=SessionConfig(consolidate_interval=999,
                                           consolidate_on_exit=False)) as s:
            s.say("今天好累")
        self.assertTrue((self.store.buffer_dir / "").exists())
        self.assertNotEqual(list(self.store.buffer_dir.glob("*.json")), [])

    def test_background_thread_runs(self):
        """后台线程真的在跑，不只是摆设。"""
        with Session(_cfg(), self.store, ScriptedLLM([_reply()] * 2),
                     options=SessionConfig(consolidate_interval=0.05,
                                           consolidate_on_exit=False)) as s:
            s.say("今天好累")
            s.say("路上还堵了")
            deadline = time.time() + 3
            while time.time() < deadline:
                if (self.store.root / "self-story.md").exists():
                    break
                time.sleep(0.02)
        self.assertTrue((self.store.root / "self-story.md").exists(),
                        "后台线程没有产出")

    def test_consolidate_now(self):
        with Session(_cfg(), self.store, ScriptedLLM([_reply()] * 2),
                     options=SessionConfig(consolidate_interval=999,
                                           consolidate_on_exit=False)) as s:
            s.say("今天好累")
            s.say("路上还堵了")
            out = s.consolidate_now()
        self.assertGreaterEqual(out["judgments"], 1)


class TestStreamingSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_stream_delivers_say_before_finish(self):
        """流式：SAY 必须边生成边到，不能等整段。"""
        raw = make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none", "嗯，然后呢。")

        def gen(messages):
            for i in range(0, len(raw), 5):
                yield raw[i:i + 5]

        seen: list[str] = []
        with Session(_cfg(), self.store, ScriptedLLM(["不应被调用"]),
                     stream_llm=gen,
                     options=SessionConfig(consolidate_interval=999)) as s:
            res = s.say("hi", on_delta=lambda p: seen.append(p))

        self.assertEqual("".join(seen), "嗯，然后呢。")
        self.assertEqual(res.say, "嗯，然后呢。")

    def test_stream_falls_back_on_error(self):
        """流断了要退回整段，不能吞掉整轮。"""
        raw = make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none", "好的。")

        def broken(messages):
            yield "<<<SAY>>>好"
            raise RuntimeError("connection reset")

        with Session(_cfg(), self.store, ScriptedLLM([raw]), stream_llm=broken,
                     options=SessionConfig(consolidate_interval=999)) as s:
            res = s.say("hi")
        self.assertEqual(res.say, "好的。")


if __name__ == "__main__":
    unittest.main(verbosity=2)
