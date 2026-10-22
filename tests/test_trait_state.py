"""人格参数的三层记忆。

    traits_init    首次载入的快照（人设是种子，活状态在 data/）
    trait_update   中期层：多条判断一致 -> 推动 value
    anchor_update  长期层：自我叙事 -> 推动 anchor（必须有依据）

三层的目的不同：
    中期层回答「这段关系让我怎么调整对他的反应」
    长期层回答「我变成了一个什么样的人」
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.config import PersonaConfig, Trait                  # noqa: E402
from persona.consolidate import Consolidator                    # noqa: E402
from persona.store import Store, rebuild_views, replay_traits   # noqa: E402


def _cfg(name="shorekeeper") -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / f"{name}.toml")


class TestTraitState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))
        self.cfg = _cfg()
        self.store.init_traits(self.cfg.role.traits_state())

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_snapshots_anchor(self):
        state = self.store.traits()
        self.assertIn("warmth", state)
        self.assertAlmostEqual(state["warmth"]["anchor"], 0.70, places=3)

    def test_init_is_idempotent(self):
        before = self.store.traits()
        self.store.init_traits(self.cfg.role.traits_state())
        self.assertEqual(before, self.store.traits())
        inits = [r for r in self.store.read_ledger() if r["kind"] == "traits_init"]
        self.assertEqual(len(inits), 1)

    def test_trait_update_records_delta_not_value(self):
        """账本记增量不记结果 —— 否则重放会把结果当增量再加一遍。"""
        self.store.update_trait("warmth", -0.01, reason="测试")
        recs = [r for r in self.store.read_ledger() if r["kind"] == "trait_update"]
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0]["delta"], -0.01)
        self.assertNotIn("value", recs[0])

    def test_replay_matches_runtime(self):
        self.store.update_trait("warmth", -0.01, reason="a")
        self.store.update_trait("warmth", -0.01, reason="b")
        runtime = self.store.traits()
        rebuilt = replay_traits(self.store.read_ledger())
        self.assertAlmostEqual(runtime["warmth"]["value"],
                               rebuilt["warmth"]["value"], places=5)

    def test_rebuild_views_restores_traits(self):
        self.store.update_trait("intervention", -0.005, reason="x")
        live = self.store.traits()
        for p in self.store.state_dir.glob("*"):
            p.unlink()
        rebuild_views(self.store)
        self.assertAlmostEqual(self.store.traits()["intervention"]["value"],
                               live["intervention"]["value"], places=5)

    def test_drift_clamps_on_replay(self):
        """重放也要夹边界，否则账本里的历史脏数据会把参数推飞。"""
        for _ in range(100):
            self.store.update_trait("intervention", -0.05, reason="extreme")
        state = self.store.traits()["intervention"]
        lo = max(state["lo"], state["anchor"] - state["drift"])
        self.assertGreaterEqual(state["value"], lo - 1e-9)


class TestMidTermPush(unittest.TestCase):
    """中期层：多条判断一致才动参数。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))
        self.cfg = _cfg()
        self.store.init_traits(self.cfg.role.traits_state())

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_judgment_does_not_move(self):
        self.store.add_judgment("他不想被建议", 0.7, affects={"intervention": -1})
        changes = Consolidator(self.store).push_traits()
        self.assertEqual(changes, [])
        self.assertAlmostEqual(self.store.traits()["intervention"]["value"], 0.20, places=4)

    def test_two_consistent_judgments_move(self):
        self.store.add_judgment("他不想被建议", 0.7, affects={"intervention": -1})
        self.store.add_judgment("他讨厌被安排", 0.7, affects={"intervention": -1})
        changes = Consolidator(self.store).push_traits()
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["name"], "intervention")
        self.assertLess(self.store.traits()["intervention"]["value"], 0.20)

    def test_opposing_judgments_cancel(self):
        """两条说 -1、两条说 +1 -> 净值为零，不动。"""
        for _ in range(2):
            self.store.add_judgment("不想被建议", 0.7, affects={"intervention": -1})
            self.store.add_judgment("想被建议", 0.7, affects={"intervention": 1})
        changes = Consolidator(self.store).push_traits()
        self.assertEqual(changes, [])

    def test_judgment_without_affects_never_moves(self):
        for _ in range(5):
            self.store.add_judgment("他是个内向的人", 0.7)
        changes = Consolidator(self.store).push_traits()
        self.assertEqual(changes, [])

    def test_zero_plasticity_never_moves(self):
        """plasticity=0 的参数是「材质」，任何判断都推不动。

        注意：plasticity 存在 traits_init 的快照里。要改它必须重新初始化，
        光改 state view 没用 —— push_traits 从账本重放。
        """
        tmp = tempfile.TemporaryDirectory()
        try:
            store = Store(Path(tmp.name))
            cfg = _cfg()
            snapshot = cfg.role.traits_state()
            snapshot["intervention"]["plasticity"] = 0.0
            store.init_traits(snapshot)
            self.assertEqual(store.traits()["intervention"]["plasticity"], 0.0)

            for _ in range(5):
                store.add_judgment("不想被建议", 0.7, affects={"intervention": -1})
            self.assertEqual(Consolidator(store).push_traits(), [])
        finally:
            tmp.cleanup()


class TestLongTermAnchor(unittest.TestCase):
    """长期层：叙事推动基线，且必须有依据。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))
        self.cfg = _cfg()
        self.store.init_traits(self.cfg.role.traits_state())

    def tearDown(self):
        self.tmp.cleanup()

    def test_anchor_update_requires_reason(self):
        """改基线必须有依据 —— 没有理由的改动直接拒绝。"""
        self.assertIsNone(self.store.update_anchor("warmth", 0.01, reason=""))
        self.assertIsNone(self.store.update_anchor("warmth", 0.01, reason="   "))

    def test_anchor_update_with_reason_works(self):
        rec = self.store.update_anchor("warmth", 0.01,
                                       reason="她说「我原来不会主动开口」")
        self.assertIsNotNone(rec)

    def test_value_follows_anchor(self):
        """anchor 移动时 value 保持相对位置，不能突变。"""
        before = self.store.traits()["warmth"]
        gap = before["value"] - before["anchor"]
        self.store.update_anchor("warmth", 0.01, reason="测试依据")
        after = self.store.traits()["warmth"]
        self.assertAlmostEqual(after["value"] - after["anchor"], gap, places=5)

    def test_anchor_step_is_tiny(self):
        """一次基线改动不超过 drift 的 5%。"""
        c = Consolidator(self.store)
        state = self.store.traits()
        spec = state["warmth"]
        max_step = spec["drift"] * c.ANCHOR_FRACTION
        self.assertLess(max_step, 0.01, "基线单次改动应该极小")

    def test_no_story_no_anchor_change(self):
        """没有叙事就不动基线。"""
        self.assertEqual(Consolidator(self.store).push_anchors(), [])

    def test_story_consumed_once(self):
        """同一段叙事只能推动一次。

        实测踩过：push_anchors 每次都读全文，同一条「我原来累了会说，
        现在话到嘴边就缩回去」被用了两次，基线被推了两遍。
        叙事是一次事件，不能当常驻依据反复消费。
        """
        calls = {"n": 0}

        def fake_llm(prompt):
            calls["n"] += 1
            return '[{"name": "warmth", "direction": 1, "evidence": "我原来不会开口"}]'

        self.store.self_story_path.write_text("我原来不会主动开口，现在会了。" * 60,
                                              encoding="utf-8")
        c = Consolidator(self.store, llm=fake_llm)
        first = c.push_anchors()
        self.assertEqual(len(first), 1)

        second = c.push_anchors()
        self.assertEqual(second, [], "同一段叙事被用了两次")
        self.assertEqual(calls["n"], 1, "没有新叙事却还在调 LLM")

    def test_new_story_can_push_again(self):
        """有新叙事时可以继续推动。"""
        def fake_llm(prompt):
            return '[{"name": "warmth", "direction": 1, "evidence": "现在会开口了"}]'

        self.store.self_story_path.write_text("我原来不会开口，话都咽回去。" * 60,
                                              encoding="utf-8")
        c = Consolidator(self.store, llm=fake_llm)
        self.assertEqual(len(c.push_anchors()), 1)

        with self.store.self_story_path.open("a", encoding="utf-8") as f:
            f.write("\n后来又过了一阵，我发现自己会先开口了。" * 20)
        pushed = c.push_anchors()
        self.assertEqual(len(pushed), 1, "新叙事应该能再推一次")

    def test_short_new_story_ignored(self):
        """新增叙事太短不算「变化」。"""
        self.store.self_story_path.write_text("原来 X，什么都不说。" * 60, encoding="utf-8")
        c = Consolidator(self.store, llm=lambda p: "[]")
        c.push_anchors()          # 消费掉
        with self.store.self_story_path.open("a", encoding="utf-8") as f:
            f.write("\n变了一点。")
        self.assertEqual(c.push_anchors(), [])


class TestSessionRestoresTraits(unittest.TestCase):
    def test_traits_survive_restart(self):
        from persona.llm import ScriptedLLM
        from persona.session import Session, SessionConfig
        tmp = tempfile.TemporaryDirectory()
        try:
            store = Store(Path(tmp.name))
            cfg = _cfg()
            with Session(cfg, store, ScriptedLLM([]),
                         options=SessionConfig(consolidate_interval=999)) as s:
                pass
            store.update_trait("warmth", -0.02, reason="跑过的痕迹")
            learned = store.traits()["warmth"]["value"]

            cfg2 = _cfg()
            with Session(cfg2, Store(Path(tmp.name)), ScriptedLLM([]),
                         options=SessionConfig(consolidate_interval=999)) as s2:
                pass
            self.assertAlmostEqual(cfg2.role.trait("warmth"), learned, places=5,
                                   msg="重启后人格参数被重置了")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
