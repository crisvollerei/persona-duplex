"""端到端：账本、结算、限幅、重建。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.config import PersonaConfig                       # noqa: E402
from persona.consolidate import Consolidator                    # noqa: E402
from persona.engine import TurnEngine                           # noqa: E402
from persona.llm import ScriptedLLM, make_reply                 # noqa: E402
from persona.settle import extract_features, judge              # noqa: E402
from persona.store import Store, rebuild_views                  # noqa: E402


def _cfg() -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()


class TestLedger(Base):
    def test_append_only_and_replayable(self):
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: 认同\nconfidence: 0.6\ndirect: 好\ndisplace: none",
                       "嗯。", "settle: -\ndelta:\n  closeness: 0.05"),
        ]))
        e.turn("我今天很累")
        n1 = len(self.store.read_ledger())
        e2 = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: 认同\nconfidence: 0.6\ndirect: 好\ndisplace: none", "嗯。"),
        ]))
        e2.turn("是啊")
        self.assertGreater(len(self.store.read_ledger()), n1)

    def test_rebuild_matches_runtime(self):
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\ndelta:\n  closeness: 0.05\n  unresolved: 那件事"),
        ]))
        e.turn("hi")
        live = self.store.relation()
        rebuilt = rebuild_views(self.store)
        self.assertAlmostEqual(live["closeness"],
                              rebuilt["relation"]["closeness"], places=2)
        self.assertIn("那件事", rebuilt["relation"]["unresolved"])

    def test_corrupt_state_recovers(self):
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\ndelta:\n  closeness: 0.05"),
        ]))
        e.turn("hi")
        (self.store.state_dir / "relation.json").write_text("{坏掉的", encoding="utf-8")
        self.store.relation()                      # 不能抛
        rebuild_views(self.store)
        self.assertIsInstance(self.store.relation()["closeness"], float)


class TestLimits(Base):
    def test_closeness_step_capped(self):
        """模型想一口气加 0.9，必须被限额截住。"""
        cfg = _cfg()
        self.store.apply_relation_delta({"closeness": 0.9}, cfg.limits)
        self.assertLessEqual(self.store.relation()["closeness"], 0.3 + 0.05 + 1e-9)

    def test_unresolved_capped(self):
        cfg = _cfg()
        for i in range(6):
            self.store.apply_relation_delta({"unresolved": [f"事{i}"]}, cfg.limits)
        self.assertLessEqual(len(self.store.relation()["unresolved"]),
                             cfg.limits.unresolved_max)

    def test_unknown_field_rejected(self):
        """慢层字段不能在线写。"""
        cfg = _cfg()
        e = TurnEngine(cfg, self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none", "嗯。",
                       "settle: -\ndelta:\n  judgments: 篡改\n  self_story: 篡改"),
        ]))
        e.turn("hi")
        self.assertFalse((self.store.root / "judgments.jsonl").exists())

    def test_pending_capped_per_turn(self):
        """pending_max 是「每轮最多挂几条」，不是总数。

        按总数卡的话，只要还有一条 WATCH 挂在 72 小时窗口里，
        预测链路就整个停摆 —— 那就再也不会产生新的校准样本了。
        """
        cfg = _cfg()
        p1 = self.store.add_pending("a", 0.6, "x", "t1", cfg.limits)
        p2 = self.store.add_pending("b", 0.6, "y", "t1", cfg.limits)
        self.assertIsNotNone(p1)
        self.assertIsNone(p2, "同一轮挂第二条应该被挡住")

        p3 = self.store.add_pending("c", 0.6, "z", "t2", cfg.limits)
        self.assertIsNotNone(p3, "下一轮应该还能挂")

    def test_pending_total_capped(self):
        """但总量要有闸，不能让 WATCH 无限堆积。"""
        cfg = _cfg()
        added = [
            self.store.add_pending(f"c{i}", 0.6, "p", f"t{i}", cfg.limits)
            for i in range(cfg.limits.pending_total_max + 3)
        ]
        self.assertEqual(len(self.store.pendings()), cfg.limits.pending_total_max)
        self.assertTrue(all(added[:cfg.limits.pending_total_max]))
        self.assertIsNone(added[-1])


class TestSettle(Base):
    def test_watch_when_topic_ignored(self):
        feat = extract_features("算了不说了",
                                prev_ai_say="今天是不是特别长啊")
        v = judge(feat, baseline_formality=0.3)
        self.assertEqual(v.kind, "WATCH")
        self.assertFalse(v.settle, "不接话题不能记账")

    def test_same_topic_without_evidence_stays_unknown(self):
        feat = extract_features("是啊，路上堵了一个半小时，还没吃饭",
                                prev_ai_say="路上堵吗")
        v = judge(feat, baseline_formality=0.3)
        self.assertEqual(v.kind, "WATCH")

    def test_hurt_detected_by_register(self):
        """用户不说被冒犯，但语域变了 —— 这是全系统最贵的信号。"""
        feat = extract_features("谢谢，我会自己处理的。")
        v = judge(feat, baseline_formality=0.2)
        self.assertEqual(v.kind, "WATCH")
        self.assertFalse(v.settle)
        self.assertTrue(v.relation_signal["suspected_hurt"])

    def test_semantic_settle_wins(self):
        """模型自己在 TAIL 里判了 —— 那是主路径，优先于启发式。"""
        feat = extract_features("嗯", prev_ai_say="今天是不是特别长啊",
                                semantic="confirmed")
        v = judge(feat, baseline_formality=0.3)
        self.assertEqual(v.kind, "CONFIRMED")
        self.assertTrue(v.settle)

    def test_hurt_signal_is_independent_of_prediction(self):
        """疑似受伤不覆盖预测判定。"""
        feat = extract_features("谢谢，我知道了。", semantic="confirmed")
        v = judge(feat, baseline_formality=0.2)
        self.assertEqual(v.kind, "CONFIRMED")
        self.assertTrue(v.relation_signal["suspected_hurt"])

    def test_expire(self):
        cfg = _cfg()
        p = self.store.add_pending("a", 0.6, "x", "t1", cfg.limits)
        self.store._save_pendings([{**p, "created": "2020-01-01T00:00:00+08:00"}])
        gone = self.store.sweep_expired(cfg.watch_ttl_hours)
        self.assertEqual(len(gone), 1)
        self.assertEqual(self.store.pendings(), [])


class TestEngineEndToEnd(Base):
    def test_declaration_flagged_not_swallowed(self):
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: 截断",
                       "我本来想说点别的，算了。"),
        ]))
        res = e.turn("hi")
        self.assertTrue(any("声明" in w or "硬失败" in w for w in res.warnings))

    def test_bare_reply_does_not_crash(self):
        """模型完全不守协议，也要能拿到回复。"""
        e = TurnEngine(_cfg(), self.store, ScriptedLLM(["今天挺累的吧。"]))
        res = e.turn("累死了")
        self.assertEqual(res.say, "今天挺累的吧。")

    def test_llm_exception_propagates_to_caller_only(self):
        """生成失败不该留下半条账。"""
        def boom(messages):
            raise RuntimeError("network")

        e = TurnEngine(_cfg(), self.store, boom)
        with self.assertRaises(RuntimeError):
            e.turn("hi")
        self.assertEqual([r for r in self.store.read_ledger() if r["kind"] == "turn"], [])


class TestWarningConsistency(Base):
    def test_settled_pending_leaves_no_stale_warning(self):
        """模型在 TAIL 里判掉了的预言，不能还留着「挂起观察」的提示。

        启发式先跑、模型判定后跑，如果不过滤就会同时出现
        「已结算」和「仍挂起」两条互相矛盾的输出。
        """
        cfg = _cfg()
        e = TurnEngine(cfg, self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: 需要认同\n"
                                "  confidence: 0.7\n  predicts: 会继续说"),
            make_reply("need: a\nconfidence: 0.7\ndirect: y\ndisplace: none",
                       "嗯。", "settle: p_0001 | confirmed | 用户继续说了"),
        ]))
        e.turn("我很累。")
        res = e.turn("是啊，路上还堵了。")

        settled_ids = {s["id"] for s in res.settled}
        self.assertIn("p_0001", settled_ids)
        for w in res.warnings:
            self.assertNotIn("p_0001", w,
                             "已结算的预言不该再出现挂起提示")


class TestConsolidator(Base):
    def test_heuristic_needs_repetition(self):
        """出现一次不算倾向。

        必须显式指定活跃度：3 条事件在「中等」档要 3 次重复才认，
        在「稀疏」档 2 次就够。不指定的话测的就不是「重复」而是「档位」。
        这里用 sparse（阈值 2）来验「1 次不够、2 次够」这个核心逻辑。
        """
        cfg = _cfg()
        e = TurnEngine(cfg, self.store, ScriptedLLM([
            make_reply("need: 只想被听\nconfidence: 0.6\ndirect: x\nbaseline: 嗯。\ndisplace: none", "嗯。", "settle: -"),
            make_reply("need: 只想被听\nconfidence: 0.6\ndirect: y\nbaseline: 嗯。\ndisplace: none", "嗯。", "settle: -"),
            make_reply("need: 要被逗\nconfidence: 0.6\ndirect: z\nbaseline: 哈。\ndisplace: none", "哈。", "settle: -"),
        ]))
        for t in ("a啊", "b啊", "c啊"):
            e.turn(t)
        out = Consolidator(self.store, gap_hours=48).run_once()
        self.assertEqual(out["activity"], "sparse")
        self.assertGreaterEqual(out["judgments"], 1)
        joined = " ".join(j["judgment"] for j in self.store.judgments())
        self.assertIn("只想被听", joined)
        self.assertNotIn("要被逗", joined, "只出现一次的不该成为判断")

    def test_empty_buffer_is_noop(self):
        out = Consolidator(self.store).run_once()
        self.assertEqual(out["judgments"], 0)
        self.assertFalse(out["self_story"])

    def test_self_story_created(self):
        cfg = _cfg()
        e = TurnEngine(cfg, self.store, ScriptedLLM([
            make_reply("need: 只想被听\nconfidence: 0.6\ndirect: x\nbaseline: 嗯。\ndisplace: none", "嗯。", "settle: -"),
            make_reply("need: 只想被听\nconfidence: 0.6\ndirect: y\nbaseline: 嗯。\ndisplace: none", "嗯。", "settle: -"),
        ]))
        for t in ("a啊", "b啊"):
            e.turn(t)
        Consolidator(self.store).run_once()
        story = (self.store.root / "self-story.md").read_text(encoding="utf-8")
        self.assertIn("人格的陪伴记录", story)
        self.assertIn("原因", story)


if __name__ == "__main__":
    unittest.main(verbosity=2)
