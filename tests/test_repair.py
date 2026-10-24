"""受伤修复测试。

这是全系统最贵的一条链路：检测靠语域、修复只有一次机会、
修错了要么变成敷衍要么变成猜错。
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
from persona.repair import (                                     # noqa: E402
    DIRECT,
    HEALED,
    OPEN,
    PLAIN,
    REPAIRING,
    RESIDUAL,
    SOFT,
    RepairBook,
    compose_repair,
    is_repair_attempt,
    judge_repair_response,
    pick_level,
    repair_level,
)
from persona.store import Store                                  # noqa: E402


def _cfg() -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


class TestDetection(unittest.TestCase):
    def test_soft_repair_recognised(self):
        self.assertTrue(is_repair_attempt("刚才那句是不是有点重。"))

    def test_plain_repair_recognised(self):
        self.assertTrue(is_repair_attempt("刚才那句……我不该那么说。"))

    def test_direct_repair_recognised(self):
        self.assertTrue(is_repair_attempt("刚才那句我说重了，对不起。"))

    def test_ordinary_say_is_not_repair(self):
        self.assertFalse(is_repair_attempt("今天是不是特别长啊。"))

    def test_apology_without_reference_is_not_repair(self):
        """光说对不起但没指向刚才，不算修复 —— 可能是别的语境。"""
        self.assertFalse(is_repair_attempt("对不起，我这边网络卡了。"))

    def test_repair_allowed_to_refer_to_itself(self):
        """修复是「位移不许声明」的唯一例外。"""
        say = "我刚才那句话是不是有点重。"
        self.assertTrue(is_repair_attempt(say))


class TestLevel(unittest.TestCase):
    def test_soft_level(self):
        self.assertEqual(repair_level("刚才那句是不是有点重"), SOFT)

    def test_direct_level(self):
        self.assertEqual(repair_level("对不起，刚才那句是我不对"), DIRECT)

    def test_pick_light_when_unsure(self):
        """线索弱时要轻 —— 把没发生的事说成发生了比不修更糟。"""
        self.assertEqual(pick_level(confidence=0.3, formality_rise=0.1), SOFT)

    def test_pick_heavy_when_clear(self):
        self.assertEqual(pick_level(confidence=0.9, formality_rise=0.5), DIRECT)

    def test_compose_is_short(self):
        for lv in (SOFT, PLAIN, DIRECT):
            self.assertLess(len(compose_repair(lv, what="你最近挺忙的")), 40,
                            "修复太长会变成辩解")


class TestResponseJudging(unittest.TestCase):
    def test_explicit_forgiveness(self):
        self.assertEqual(judge_repair_response("没事，我也没往心里去"), "healed")

    def test_explicit_refusal(self):
        self.assertEqual(judge_repair_response("不用说了，没必要"), "rejected")

    def test_register_relaxed_is_not_explicit_forgiveness(self):
        """语气变化不证明用户接受了修复。"""
        self.assertEqual(
            judge_repair_response("哈哈其实我也觉得我有点敏感了，刚才那事儿吧……"),
            "unclear",
        )

    def test_short_reply_is_unknown(self):
        self.assertEqual(judge_repair_response("知道了。"), "unclear")

    def test_empty_is_unclear(self):
        self.assertEqual(judge_repair_response(""), "unclear")


class TestRepairBook(unittest.TestCase):
    def test_open_is_idempotent(self):
        book = RepairBook()
        book.open("n1", 1, "t", PLAIN)
        book.open("n1", 2, "t2", PLAIN)
        self.assertEqual(len(book.active()), 1)

    def test_mark_repairing(self):
        book = RepairBook()
        book.open("n1", 1, "t", PLAIN)
        book.mark_repairing("n1")
        self.assertEqual(book.items[0].state, REPAIRING)
        self.assertEqual(book.items[0].attempts, 1)

    def test_healed_closes(self):
        book = RepairBook()
        book.open("n1", 1, "t", PLAIN)
        book.close("n1", healed=True)
        self.assertEqual(book.items[0].state, HEALED)
        self.assertEqual(book.active(), [])

    def test_roundtrip(self):
        book = RepairBook()
        book.open("n1", 1, "t", PLAIN)
        book.mark_repairing("n1")
        clone = RepairBook.from_list(book.to_list())
        self.assertEqual(clone.items[0].state, REPAIRING)


class TestEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _engine(self, replies):
        return TurnEngine(_cfg(), self.store, ScriptedLLM(replies))

    def test_hurt_opens_repair(self):
        """检测到受伤 → 开一条修复，并记进未解决。"""
        e = self._engine([
            make_reply("need: 认同\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: 需要认同\n"
                                "  confidence: 0.7\n  predicts: 会继续说"),
            # 第二轮语域突变 → 受伤
            make_reply("need: 认同\nconfidence: 0.6\ndirect: y\ndisplace: none", "嗯。"),
        ])
        e.turn("我今天很累，老板又骂我了。")
        e.turn("谢谢，我知道了。")
        book = e._repair_book()
        self.assertTrue(book.active(), "受伤了但没开修复")
        rel = self.store.relation()
        self.assertTrue(any("伤到" in u for u in rel["unresolved"]))

    def test_repair_attempt_recorded(self):
        e = self._engine([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: 需要认同\n"
                                "  confidence: 0.7\n  predicts: 会说"),
            make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none", "嗯。"),
            make_reply("need: a\nconfidence: 0.6\ndirect: z\ndisplace: none",
                       "刚才那句是不是有点重。"),
        ])
        e.turn("我很累。")
        e.turn("谢谢，我知道了。")
        res = e.turn("……")
        self.assertIsNotNone(res.repair_opened)
        self.assertEqual(res.repair_opened["action"], "attempted")

    def test_healed_clears_unresolved(self):
        e = self._engine([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: 需要认同\n"
                                "  confidence: 0.7\n  predicts: 会说"),
            make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none", "嗯。"),
            make_reply("need: a\nconfidence: 0.6\ndirect: z\ndisplace: none",
                       "刚才那句是不是有点重。"),
            make_reply("need: a\nconfidence: 0.6\ndirect: w\ndisplace: none", "嗯。"),
        ])
        e.turn("我很累。")
        e.turn("谢谢，我知道了。")
        e.turn("……")
        res = e.turn("没事，我也没往心里去，哈哈。")
        self.assertIsNotNone(res.repair_closed)
        self.assertEqual(res.repair_closed["verdict"], "healed")
        rel = self.store.relation()
        self.assertFalse(any("伤到" in u for u in rel["unresolved"]),
                         "修复成功了但未解决事项没销账")

    def test_rejected_pauses_without_inventing_sore_point(self):
        """停止请求结束修复话题，不把暂定信号固化成长期禁区。"""
        e = self._engine([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: 需要认同\n"
                                "  confidence: 0.7\n  predicts: 会说"),
            make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none", "嗯。"),
            make_reply("need: a\nconfidence: 0.6\ndirect: z\ndisplace: none",
                       "刚才那句是不是有点重。"),
            make_reply("need: a\nconfidence: 0.6\ndirect: w\ndisplace: none", "嗯。"),
        ])
        e.turn("我很累。")
        e.turn("谢谢，我知道了。")
        e.turn("……")
        res = e.turn("不用说了，没必要。")
        self.assertEqual(res.repair_closed["verdict"], "rejected")
        rel = self.store.relation()
        self.assertEqual(rel["sore_points"], [])
        self.assertTrue(any(r["state"] == "paused" for r in self.store.load_view("repair", {})["items"]))

    def test_repair_note_injected_into_context(self):
        """待修复的事必须进上下文，否则模型根本不知道该道歉。"""
        e = self._engine([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: 需要认同\n"
                                "  confidence: 0.7\n  predicts: 会说"),
            make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none", "嗯。"),
        ])
        e.turn("我很累。")
        e.turn("谢谢，我知道了。")
        note = e._repair_note()
        self.assertIn("伤到", note)
        self.assertIn("别解释", note)


if __name__ == "__main__":
    unittest.main(verbosity=2)
