"""会话与后台线程测试。

重点：后台必须真的跑起来，退出时必须把积压清掉。
消费成功不保证必须产出新判断；记忆不是自动人格成长。
"""
from __future__ import annotations

import sys
import tempfile
import threading
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
    return make_reply(f"need: {need}\nconfidence: 0.6\ndirect: x\nbaseline: 嗯。\ndisplace: none",
                      say, "settle: -")


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
            deadline = time.monotonic() + 3
            consumed = set()
            while time.monotonic() < deadline:
                consumed = {eid for row in self.store.read_ledger()
                            if row.get("kind") == "memory_batch_commit"
                            for eid in row.get("event_ids", [])}
                if len(consumed) == 2:
                    break
                time.sleep(0.02)
        # The worker may consume the two turns in separate batches, neither of
        # which has enough repeated evidence to justify a new judgment/story.
        self.assertEqual(len(consumed), 2, "后台线程没有处理全部积压")

    def test_consolidate_now(self):
        with Session(_cfg(), self.store, ScriptedLLM([_reply()] * 2),
                     options=SessionConfig(consolidate_interval=999,
                                           consolidate_on_exit=False)) as s:
            s.say("今天好累")
            s.say("路上还堵了")
            out = s.consolidate_now()
        self.assertGreaterEqual(out["judgments"], 1)

    def test_say_returns_while_background_memory_model_is_still_running(self):
        entered = threading.Event()
        release = threading.Event()
        returned = threading.Event()
        memory_threads = []
        errors = []
        results = []
        llm = ScriptedLLM([_reply(), _reply()])

        def slow_memory(prompt):
            memory_threads.append(threading.current_thread().name)
            entered.set()
            release.wait()
            return "[]"

        def foreground():
            try:
                results.append(session.say("今天好累"))
            except Exception as exc:
                errors.append(exc)
            finally:
                returned.set()

        session = Session(_cfg(), self.store, llm, consolidate_llm=slow_memory,
                          options=SessionConfig(consolidate_interval=0.01,
                                                consolidate_on_exit=False))
        worker = threading.Thread(target=foreground, daemon=True)
        try:
            worker.start()
            self.assertTrue(returned.wait(3), "前台不应等待记忆提炼")
            self.assertTrue(entered.wait(3), "后台没有开始提炼")
            worker.join(timeout=3)
            self.assertEqual(errors, [])
            self.assertEqual(memory_threads, ["persona-consolidator"])

            returned.clear()
            worker = threading.Thread(target=foreground, daemon=True)
            worker.start()
            self.assertTrue(returned.wait(3), "后台尚未完成时下一轮也应返回")
            self.assertFalse(release.is_set())
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(len(llm.calls), 2)
        finally:
            release.set()
            worker.join(timeout=3)
            session.close()


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

    def test_stream_error_keeps_partial_without_another_model_call(self):
        """流断了保留已发出的内容，不自动补发整段请求。"""

        def broken(messages):
            yield "<<<SAY>>>好"
            raise RuntimeError("connection reset")

        llm = ScriptedLLM(["不应被调用"])
        seen = []
        with Session(_cfg(), self.store, llm, stream_llm=broken,
                     options=SessionConfig(consolidate_interval=999)) as s:
            res = s.say("hi", on_delta=seen.append)
        self.assertEqual(res.say, "好")
        self.assertEqual("".join(seen), res.say)
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
