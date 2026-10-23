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
        """有新叙事时可以继续推动 —— 但**要等频率因子放行**。

        刚推过一次的话，间隔为 0，f1 = 1 - exp(0) = 0，推不动。
        这是对的：连续两次固化之间不该有基线漂移。
        """
        def fake_llm(prompt):
            return '[{"name": "warmth", "direction": 1, "evidence": "现在会开口了"}]'

        self.store.self_story_path.write_text("我原来不会开口，话都咽回去。" * 60,
                                              encoding="utf-8")
        c = Consolidator(self.store, llm=fake_llm)
        self.assertEqual(len(c.push_anchors()), 1)

        with self.store.self_story_path.open("a", encoding="utf-8") as f:
            f.write("\n后来又过了一阵，我发现自己会先开口了。" * 20)
        # 紧接着再推：频率因子挡住（间隔 0）
        self.assertEqual(c.push_anchors(), [], "间隔为 0 时不该推进基线")

        # 把上次推动时间往前挪 24 小时，模拟「过了一天」
        pushes = self.store.load_view("anchor_pushes", {"times": []})
        from persona.store import _LOCAL_TZ
        from datetime import datetime, timedelta
        old = (datetime.now(_LOCAL_TZ) - timedelta(hours=24)).isoformat(timespec="seconds")
        self.store.save_view("anchor_pushes", {"times": [old]})
        # 给一段新叙事（游标已经走到末尾了）
        with self.store.self_story_path.open("a", encoding="utf-8") as f:
            f.write("\n再后来又过了一阵，我确认自己真的变了。" * 20)
        self.assertEqual(len(c.push_anchors()), 1, "隔一天应该能再推")

    def test_short_total_story_still_works(self):
        """叙事总长不足 400 字也该能判读 —— 只要新增够 200。

        原来有个 `len(story) < 400` 门槛，和「新增 >= 200」重复了，
        而且让短会话永远进不了判读。实测副作用：新角色跑十几轮
        写出的叙事只有 300 多字，长期层完全不启动。
        """
        class Always:
            def __call__(self, p):
                return '[{"name": "warmth", "direction": 1, "evidence": "e"}]'

        # 300 字，不到旧的 400 门槛，但超过新增的 200
        self.store.self_story_path.write_text("我变了。" * 75, encoding="utf-8")
        self.assertEqual(len(self.store.self_story_path.read_text(
            encoding="utf-8")), 300)
        c = Consolidator(self.store, llm=Always())
        self.assertEqual(len(c.push_anchors()), 1, "300 字叙事应该能判读")

    def test_short_new_story_ignored(self):
        """新增叙事太短不算「变化」。"""
        self.store.self_story_path.write_text("原来 X，什么都不说。" * 60, encoding="utf-8")
        c = Consolidator(self.store, llm=lambda p: "[]")
        c.push_anchors()          # 消费掉
        with self.store.self_story_path.open("a", encoding="utf-8") as f:
            f.write("\n变了一点。")
        self.assertEqual(c.push_anchors(), [])


class TestJudgmentConsumption(unittest.TestCase):
    """每条判断只能推动参数一次。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))
        self.cfg = _cfg()
        self.store.init_traits(self.cfg.role.traits_state())

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_judgments_not_reused(self):
        """3 条判断连续 push 10 次，参数只该动一次。

        实测踩过：没有消费标记时，同 3 条判断被推了 20 次，
        intervention 从 0.20 掉到 0.1775 —— 同一条证据用了 20 遍。
        """
        for i in range(3):
            self.store.add_judgment("他不想被建议%d" % i, 0.7,
                                    affects={"intervention": -1})
        c = Consolidator(self.store)
        c.push_traits()
        first = self.store.traits()["intervention"]["value"]
        for _ in range(10):
            c.push_traits()
        self.assertAlmostEqual(self.store.traits()["intervention"]["value"],
                               first, places=6,
                               msg="同一批判断被重复消费了")

    def test_new_judgments_can_push_again(self):
        c = Consolidator(self.store)
        for i in range(2):
            self.store.add_judgment("A%d" % i, 0.7, affects={"intervention": -1})
        c.push_traits()
        v1 = self.store.traits()["intervention"]["value"]
        for i in range(2):
            self.store.add_judgment("B%d" % i, 0.7, affects={"intervention": -1})
        c.push_traits()
        self.assertLess(self.store.traits()["intervention"]["value"], v1)

    def test_insufficient_judgments_accumulate(self):
        """不够阈值的判断要留着，等后续证据一起凑。

        实测踩过：第 1 条判断不够 1.4 就被标记成已消费，
        第 2 条来时从零开始 —— 连续加 11 条同向判断，参数一步都没动。
        """
        c = Consolidator(self.store)
        self.store.add_judgment("单条 0.7", 0.7, affects={"intervention": -1})
        self.assertEqual(c.push_traits(), [], "单条不该推")
        # 第二条凑够 1.4，应该推了
        self.store.add_judgment("第二条 0.7", 0.7, affects={"intervention": -1})
        changes = c.push_traits()
        self.assertEqual(len(changes), 1, "两条凑够阈值应该推")
        self.assertLess(self.store.traits()["intervention"]["value"], 0.20)

    def test_accumulation_over_many_judgments(self):
        """连续加同向判断，参数应该逐步下移而不是一步不动。"""
        c = Consolidator(self.store)
        values = []
        for i in range(8):
            self.store.add_judgment("判断%d" % i, 0.7, affects={"intervention": -1})
            c.push_traits()
            values.append(self.store.traits()["intervention"]["value"])
        self.assertLess(values[-1], 0.20, "8 条判断后应该有漂移")
        self.assertGreater(values[-1], 0.19, "不该一步跳到很远")
        # 单调不增
        for a, b in zip(values, values[1:]):
            self.assertLessEqual(b, a + 1e-9)

    def test_pushed_judgments_not_reused(self):
        """推过的判断不该继续参与投票。"""
        c = Consolidator(self.store)
        for i in range(2):
            self.store.add_judgment("A%d" % i, 0.7, affects={"intervention": -1})
        c.push_traits()
        v1 = self.store.traits()["intervention"]["value"]
        c.push_traits()
        self.assertAlmostEqual(self.store.traits()["intervention"]["value"], v1,
                               places=6, msg="已推过的判断还在投票")


class TestPersonaSwitch(unittest.TestCase):
    """同一个 data 目录换人格，参数不能串。"""

    def test_switching_persona_resets_traits(self):
        from persona.llm import ScriptedLLM
        from persona.session import Session, SessionConfig

        tmp = tempfile.TemporaryDirectory()
        try:
            cfg_a = _cfg("shorekeeper")
            with Session(cfg_a, Store(Path(tmp.name)), ScriptedLLM([]),
                         options=SessionConfig(consolidate_interval=999)):
                pass
            self.assertAlmostEqual(Store(Path(tmp.name)).traits()["warmth"]["value"],
                                   0.70, places=3)

            cfg_b = _cfg("ache")
            with Session(cfg_b, Store(Path(tmp.name)), ScriptedLLM([]),
                         options=SessionConfig(consolidate_interval=999)):
                pass
            self.assertAlmostEqual(cfg_b.role.trait("warmth"), 0.30, places=3)
        finally:
            tmp.cleanup()

    def test_same_persona_keeps_drift(self):
        """同一人格重开，漂移要保住。"""
        from persona.llm import ScriptedLLM
        from persona.session import Session, SessionConfig

        tmp = tempfile.TemporaryDirectory()
        try:
            cfg = _cfg()
            with Session(cfg, Store(Path(tmp.name)), ScriptedLLM([]),
                         options=SessionConfig(consolidate_interval=999)):
                pass
            Store(Path(tmp.name)).update_trait("warmth", -0.02, reason="x")
            learned = Store(Path(tmp.name)).traits()["warmth"]["value"]

            cfg2 = _cfg()
            with Session(cfg2, Store(Path(tmp.name)), ScriptedLLM([]),
                         options=SessionConfig(consolidate_interval=999)):
                pass
            self.assertAlmostEqual(cfg2.role.trait("warmth"), learned, places=5)
        finally:
            tmp.cleanup()

    def test_traits_reset_recorded(self):
        from persona.llm import ScriptedLLM
        from persona.session import Session, SessionConfig

        tmp = tempfile.TemporaryDirectory()
        try:
            for name in ("shorekeeper", "ache"):
                with Session(_cfg(name), Store(Path(tmp.name)), ScriptedLLM([]),
                             options=SessionConfig(consolidate_interval=999)):
                    pass
            kinds = [r["kind"] for r in Store(Path(tmp.name)).read_ledger()
                     if "traits" in r["kind"]]
            self.assertEqual(kinds, ["traits_init", "traits_reset", "traits_init"])
        finally:
            tmp.cleanup()


class TestAnchorFrequency(unittest.TestCase):
    """基线推动的频率衰减。

    没有它的时候：连续 20 次固化就能把 anchor 推到 drift 边界
    （实测 warmth 0.70 -> 0.82）。后台间隔 300 秒，也就是
    **一次两小时的对话就能把人格基线推到极限**。

    两个因子：
      f1 = 1 - exp(-gap / TAU)    距上次推动越久，越接近满额
      f2 = 1 / (1 + n)            窗口内推得越多，每次越弱
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))
        self.store.init_traits(_cfg().role.traits_state())

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_push_is_full_strength(self):
        """从没推过 -> 满额。"""
        self.assertAlmostEqual(Consolidator(self.store)._anchor_frequency(), 1.0, places=6)

    def test_immediate_second_push_is_blocked(self):
        """刚推过就跑第二次 -> 系数接近 0。"""
        c = Consolidator(self.store)
        c._record_anchor_push()
        self.assertLess(c._anchor_frequency(), 0.01)

    def test_factor_one_grows_with_gap(self):
        """间隔越久，系数越接近 1。"""
        from datetime import datetime, timedelta
        from persona.store import _LOCAL_TZ
        c = Consolidator(self.store)

        def freq_at(hours):
            old = (datetime.now(_LOCAL_TZ) - timedelta(hours=hours)
                   ).isoformat(timespec="seconds")
            self.store.save_view("anchor_pushes", {"times": [old]})
            return c._anchor_frequency()

        f1h, f6h, f24h = freq_at(1), freq_at(6), freq_at(24)
        self.assertLess(f1h, f6h)
        self.assertLess(f6h, f24h)
        self.assertGreater(f24h, 0.9)

    def test_factor_two_weakens_with_recent_count(self):
        """同一窗口里已推次数越多，系数越低。"""
        from datetime import datetime, timedelta
        from persona.store import _LOCAL_TZ
        c = Consolidator(self.store)
        now = datetime.now(_LOCAL_TZ)

        def freq_with(n_recent):
            times = [(now - timedelta(minutes=30)).isoformat(timespec="seconds")
                     for _ in range(n_recent)]
            self.store.save_view("anchor_pushes", {"times": times})
            return c._anchor_frequency()

        self.assertGreater(freq_with(1), freq_with(3))
        self.assertGreater(freq_with(3), freq_with(8))

    def test_dense_consolidation_barely_moves_anchor(self):
        """两小时内反复固化 20 次，anchor 只走第一步。

        这是这次修的核心 —— 修之前 20 次会走满 drift（0.70 -> 0.82）。
        现在只有第一次拿满额（此前从没推过），之后 19 次全被挡住。

        预期总量 ≈ 1 步 = drift × ANCHOR_FRACTION = 0.12 × 0.05 = 0.006。
        """
        class Always:
            def __call__(self, p):
                return '[{"name": "warmth", "direction": 1, "evidence": "e"}]'

        c = Consolidator(self.store, llm=Always())
        before = self.store.traits()["warmth"]["anchor"]
        for _ in range(20):
            self.store.self_story_path.write_text("我变了。" * 300, encoding="utf-8")
            self.store.set_cursor("story", 0, "offset")
            c.push_anchors()
        after = self.store.traits()["warmth"]["anchor"]
        moved = abs(after - before)
        self.assertLess(moved, 0.008,
                        "密集固化把基线推动了 %.4f（应该只走一步 0.006）" % moved)
        self.assertGreater(moved, 0.004, "第一次推动应该是满额的")

        # 关键对比：修之前 20 次会跑到 0.82（drift 边界）
        self.assertLess(after, 0.71, "不应该接近 drift 边界")

    def test_daily_consolidation_does_move(self):
        """每天固化一次，累积效果明显。"""
        from datetime import datetime, timedelta
        from persona.store import _LOCAL_TZ

        class Always:
            def __call__(self, p):
                return '[{"name": "warmth", "direction": 1, "evidence": "e"}]'

        c = Consolidator(self.store, llm=Always())
        before = self.store.traits()["warmth"]["anchor"]
        now = datetime.now(_LOCAL_TZ)
        for day in range(5):
            # 模拟「距上次推动已过一天」
            t = (now - timedelta(days=1)).isoformat(timespec="seconds")
            self.store.save_view("anchor_pushes", {"times": [t]})
            self.store.self_story_path.write_text(
                ("我变了。第%d天。" % day) * 300, encoding="utf-8")
            self.store.set_cursor("story", 0, "offset")
            c.push_anchors()
        after = self.store.traits()["warmth"]["anchor"]
        self.assertGreater(after - before, 0.01, "每天一次应该累积出可见漂移")

    def test_low_frequency_push_not_recorded(self):
        """系数太低时不写账本 —— 否则被 1e-5 级噪声填满。"""
        class Always:
            def __call__(self, p):
                return '[{"name": "warmth", "direction": 1, "evidence": "e"}]'

        c = Consolidator(self.store, llm=Always())
        self.store.self_story_path.write_text("我变了。" * 300, encoding="utf-8")
        c._record_anchor_push()          # 刚推过
        n_before = len([r for r in self.store.read_ledger()
                        if r["kind"] == "anchor_update"])
        self.store.set_cursor("story", 0, "offset")
        c.push_anchors()
        n_after = len([r for r in self.store.read_ledger()
                       if r["kind"] == "anchor_update"])
        self.assertEqual(n_before, n_after, "系数不够时不该写账本")


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
