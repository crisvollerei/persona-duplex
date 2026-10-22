"""话题判定测试。

两个必须守住的性质：
  1. 异步不拖慢主链路 —— 生成花掉全部预算时，判定必须立刻返回兜底
  2. 任何失败都退回字面 —— 网络错、JSON 坏、模型胡写，都不能让对话卡住
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

from persona.config import PersonaConfig                        # noqa: E402
from persona.engine import TurnEngine                            # noqa: E402
from persona.llm import ScriptedLLM, make_reply                  # noqa: E402
from persona.store import Store                                  # noqa: E402
from persona.topicjudge import (                                 # noqa: E402
    LEFT,
    RETURNED,
    UNCLEAR,
    AsyncJudge,
    BigramJudge,
    JudgeRequest,
    LLMJudge,
    build_judge,
)


def _cfg() -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


def _req(user="投了两家简历", pending_id="p_1", claim="需要被允许",
         predicts="会说出某个具体方案") -> JudgeRequest:
    return JudgeRequest(
        user_text=user,
        prev_ai_say="你每次说到这一步就跑",
        pendings=[{"id": pending_id, "claim": claim, "predicts": predicts}],
    )


class TestLLMJudge(unittest.TestCase):
    def test_parses_returned(self):
        llm = ScriptedLLM(['[{"id":"p_1","verdict":"returned","why":"说了具体方案"}]'])
        out = LLMJudge(llm).judge(_req())
        self.assertEqual(out["p_1"].verdict, RETURNED)
        self.assertIn("具体方案", out["p_1"].why)

    def test_parses_left(self):
        llm = ScriptedLLM(['[{"id":"p_1","verdict":"left","why":"换成吃饭了"}]'])
        self.assertEqual(LLMJudge(llm).judge(_req())["p_1"].verdict, LEFT)

    def test_code_fence_tolerated(self):
        llm = ScriptedLLM(['```json\n[{"id":"p_1","verdict":"returned"}]\n```'])
        self.assertEqual(LLMJudge(llm).judge(_req())["p_1"].verdict, RETURNED)

    def test_garbage_becomes_unclear(self):
        """模型胡写不能当成「没回来」—— 那是把噪声当信号。"""
        for raw in ["我不知道", "", "{坏的", "[{]", "null"]:
            out = LLMJudge(ScriptedLLM([raw])).judge(_req())
            self.assertEqual(out["p_1"].verdict, UNCLEAR, f"输入 {raw!r}")

    def test_unknown_id_ignored(self):
        llm = ScriptedLLM(['[{"id":"p_999","verdict":"returned"}]'])
        out = LLMJudge(llm).judge(_req())
        self.assertEqual(out["p_1"].verdict, UNCLEAR)
        self.assertNotIn("p_999", out)

    def test_bad_verdict_ignored(self):
        llm = ScriptedLLM(['[{"id":"p_1","verdict":"maybe"}]'])
        self.assertEqual(LLMJudge(llm).judge(_req())["p_1"].verdict, UNCLEAR)

    def test_no_pending_skips_call(self):
        llm = ScriptedLLM([])
        req = JudgeRequest(user_text="hi", pendings=[])
        self.assertEqual(LLMJudge(llm).judge(req), {})
        self.assertEqual(len(llm.calls), 0)


class TestBigramFallback(unittest.TestCase):
    def test_returns_when_literal_overlap(self):
        req = _req(user="今天路上堵了一个半小时，好累", claim="很累",
                   predicts="会说路上堵")
        out = BigramJudge().judge(req)
        self.assertEqual(out["p_1"].verdict, RETURNED)

    def test_unclear_when_unrelated(self):
        """字面完全不重合 → unclear，不是 left。

        「用户没说这件事」和「用户说的和预言相反」是两回事。
        旧版把前者判成 left，等于把「还没验证」当成「否证」，
        实跑里把命中率压到 40% 上下。
        """
        req = _req(user="今天天气不错", claim="辞职", predicts="会说出方案")
        self.assertEqual(BigramJudge().judge(req)["p_1"].verdict, UNCLEAR)

    def test_returned_when_literal_overlap(self):
        req = _req(user="今天路上堵了一个半小时，好累", claim="很累", predicts="会说路上堵")
        self.assertEqual(BigramJudge().judge(req)["p_1"].verdict, RETURNED)

    def test_empty_pendings(self):
        self.assertEqual(BigramJudge().judge(JudgeRequest(user_text="hi")), {})


class TestAsyncJudge(unittest.TestCase):
    def test_returns_llm_result_when_fast(self):
        inner = LLMJudge(ScriptedLLM(['[{"id":"p_1","verdict":"returned"}]']))
        h = AsyncJudge(inner, grace=3.0).submit(_req())
        verdicts, source = h.collect()
        self.assertEqual(source, "llm")
        self.assertEqual(verdicts["p_1"].verdict, RETURNED)

    def test_timeout_falls_back_immediately(self):
        """超时必须立刻返回兜底，不能等。"""
        def slow(req):
            time.sleep(2.0)
            return {"p_1": None}

        h = AsyncJudge(slow, grace=0.05).submit(_req(user="今天天气不错"))
        t0 = time.monotonic()
        verdicts, source = h.collect()
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.5, "超时后还在等")
        self.assertEqual(source, "late")
        # 兜底判不出「否证」，只能判「没验证」
        self.assertEqual(verdicts["p_1"].verdict, UNCLEAR)

    def test_exception_falls_back(self):
        def boom(req):
            raise RuntimeError("网络断了")

        h = AsyncJudge(boom, grace=1.0).submit(_req(user="今天天气不错"))
        verdicts, source = h.collect()
        self.assertEqual(source, "late")
        self.assertIn(verdicts["p_1"].verdict, (LEFT, RETURNED, UNCLEAR))

    def test_zero_budget_never_waits(self):
        """生成已经花光预算时，collect 必须零等待。"""
        def slow(req):
            time.sleep(1.0)
            return {}

        h = AsyncJudge(slow, grace=0.0).submit(_req())
        t0 = time.monotonic()
        h.collect()
        self.assertLess(time.monotonic() - t0, 0.2)


class TestBuildJudge(unittest.TestCase):
    def test_none_when_disabled(self):
        cfg = _cfg()
        cfg.judge_enabled = False
        self.assertIsNone(build_judge(cfg, ScriptedLLM([])))

    def test_none_without_llm(self):
        self.assertIsNone(build_judge(_cfg(), None))

    def test_built_when_enabled(self):
        j = build_judge(_cfg(), ScriptedLLM([]))
        self.assertIsInstance(j, AsyncJudge)


class TestEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _seed_pending(self):
        cfg = _cfg()
        return self.store.add_pending(
            "需要被允许", 0.6, "会说出某个具体方案", "t0", cfg.limits
        )

    def test_semantic_returned_settles_even_without_literal_overlap(self):
        """这是接 LLM 的全部意义：字面不重合但语义对上了。"""
        p = self._seed_pending()
        judge = AsyncJudge(
            LLMJudge(ScriptedLLM([f'[{{"id":"{p["id"]}","verdict":"returned"}}]'])),
            grace=3.0,
        )
        e = TurnEngine(_cfg(), self.store,
                       ScriptedLLM([make_reply("need: a\nconfidence: 0.6\n"
                                               "direct: x\ndisplace: none", "嗯。")]),
                       judge=judge)
        res = e.turn("其实我已经投了两家简历了。")
        self.assertTrue(any(s["id"] == p["id"] for s in res.settled),
                        "语义判 returned 却没结算")

    def test_semantic_left_keeps_watching(self):
        p = self._seed_pending()
        judge = AsyncJudge(
            LLMJudge(ScriptedLLM([f'[{{"id":"{p["id"]}","verdict":"left"}}]'])),
            grace=3.0,
        )
        e = TurnEngine(_cfg(), self.store,
                       ScriptedLLM([make_reply("need: a\nconfidence: 0.6\n"
                                               "direct: x\ndisplace: none", "嗯。")]),
                       judge=judge)
        res = e.turn("……算了，不聊这个。")
        self.assertFalse([s for s in res.settled if s["id"] == p["id"]])
        self.assertEqual(len(self.store.pendings()), 1)

    def test_no_judge_still_works(self):
        p = self._seed_pending()
        e = TurnEngine(_cfg(), self.store,
                       ScriptedLLM([make_reply("need: a\nconfidence: 0.6\n"
                                               "direct: x\ndisplace: none", "嗯。")]))
        res = e.turn("今天天气不错")
        self.assertFalse([s for s in res.settled if s["id"] == p["id"]])

    def test_turn_not_slowed_by_slow_judge(self):
        """判定再慢，也不能拖慢这一轮。"""
        p = self._seed_pending()

        def slow(req):
            time.sleep(1.5)
            return {}

        e = TurnEngine(_cfg(), self.store,
                       ScriptedLLM([make_reply("need: a\nconfidence: 0.6\n"
                                               "direct: x\ndisplace: none", "嗯。")]),
                       judge=AsyncJudge(slow, grace=0.05))
        t0 = time.monotonic()
        e.turn("嗯")
        self.assertLess(time.monotonic() - t0, 0.8, "慢判定拖慢了主链路")

    def test_judge_sees_pendings_snapshot(self):
        """判定必须看到挂起时的预言，否则判不了。"""
        p = self._seed_pending()
        seen = {}

        def capture(req):
            seen["ids"] = [x["id"] for x in req.pendings]
            return {p["id"]: type("V", (), {"verdict": "unclear", "why": ""})()}

        e = TurnEngine(_cfg(), self.store,
                       ScriptedLLM([make_reply("need: a\nconfidence: 0.6\n"
                                               "direct: x\ndisplace: none", "嗯。")]),
                       judge=AsyncJudge(capture, grace=2.0))
        e.turn("嗯")
        self.assertIn(p["id"], seen.get("ids", []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
