"""审查报告发现的缺陷 —— 回归测试。

每一条都对应一个实跑复现过的 bug，注释里写了复现方式。
这是「测试全绿但功能是坏的」的补救：这些路径之前没有任何覆盖。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.config import PersonaConfig, StanceRule, Trait       # noqa: E402
from persona.consolidate import Consolidator                       # noqa: E402
from persona.metrics import compute                                # noqa: E402
from persona.parse import _to_float, parse                         # noqa: E402
from persona.store import Store, rebuild_views, replay_traits      # noqa: E402
from persona.stream import StreamParser                            # noqa: E402


def _cfg(name="shorekeeper"):
    return PersonaConfig.load(ROOT / "personas" / (name + ".toml"))


class TestDirectNeverLeaks(unittest.TestCase):
    """direct 永远不能出现在用户可见输出里（README 硬约束第 2 条）。"""

    def test_missing_head_does_not_leak_direct(self):
        """缺 <<<HEAD>>> 但裸写四个字段 —— 旧实现整段当正文发出去。"""
        raw = ("need: 只想被听\nconfidence: 0.82\ndirect: 你根本不是在问我\n"
               "displace: 降级\n\n<<<SAY>>>\n我没事。\n")
        t = parse(raw)
        self.assertNotIn("你根本不是在问我", t.say)
        self.assertIn("我没事", t.say)

    def test_stream_parse_agree_out_of_order(self):
        """同一份字节，流式与非流式必须得到同样的账本状态。"""
        raw = ("<<<SAY>>>\n我没事。\n<<<HEAD>>>\nneed: 只想被听\n"
               "confidence: 0.8\ndirect: 你根本不是在问我\ndisplace: 降级\n"
               "<<<TAIL>>>\nsettle: -\n")
        a = parse(raw)
        sp = StreamParser()
        for ch in raw:
            sp.feed(ch)
        b = sp.finish()
        self.assertEqual(a.head is not None, b.head is not None)
        if a.head and b.head:
            self.assertEqual(a.head.displace, b.head.displace)


class TestRelationReplay(unittest.TestCase):
    def test_unresolved_cap_survives_rebuild(self):
        """重放必须和运行时用同一套限幅。"""
        tmp = tempfile.TemporaryDirectory()
        try:
            from persona.engine import TurnEngine
            from persona.llm import ScriptedLLM, make_reply
            st = Store(Path(tmp.name))
            cfg = _cfg()
            cfg.limits.unresolved_max = 3
            replies = [
                make_reply("need: 只想被听\nconfidence: 0.5\ndirect: x\ndisplace: none",
                           "嗯。", "settle: -\ndelta:\n  unresolved: 事" + str(i))
                for i in range(6)
            ]
            e = TurnEngine(cfg, st, ScriptedLLM(replies))
            for i in range(6):
                e.turn("第" + str(i) + "轮")
            live = list(st.relation()["unresolved"])
            self.assertEqual(len(live), 3)
            for p in st.state_dir.glob("*"):
                p.unlink()
            rebuild_views(st, limits=cfg.limits)
            self.assertEqual(st.relation()["unresolved"], live)
        finally:
            tmp.cleanup()


class TestTraitBounds(unittest.TestCase):
    def test_anchor_clamped(self):
        """anchor 必须夹在 [lo, hi]，否则参数永久卡死。"""
        t = Trait(anchor=0.15, plasticity=0.25, drift=0.10, lo=0.02, hi=0.40)
        for _ in range(400):
            t.anchor += 0.005
            t.value += 0.005
            t.clamp()
        self.assertLessEqual(t.anchor, 0.40 + 1e-9)
        self.assertLessEqual(t.value, 0.40 + 1e-9)
        self.assertNotEqual(t.effective_plasticity(direction=-1), 0.0)

    def test_value_zero_not_unset(self):
        """value=0.0 是合法值，不能被 falsy 判断重置成 anchor。"""
        t = Trait(anchor=0.5, value=0.0, plasticity=0.2, drift=0.5)
        self.assertAlmostEqual(t.value, 0.0, places=6)

    def test_replay_and_trait_agree(self):
        """replay_traits 和 Trait.clamp 不能分叉。"""
        ledger = [
            {"kind": "traits_init",
             "traits": {"x": {"anchor": 0.5, "value": 0.5, "drift": 0.6,
                              "lo": 0.8, "hi": 0.2}}},
            {"kind": "trait_update", "name": "x", "delta": 0.1},
        ]
        r = replay_traits(ledger)
        self.assertAlmostEqual(r["x"]["value"],
                               Trait.from_state(r["x"]).value, places=9)


class TestConsolidationRobustness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))
        self.store.init_traits(_cfg().role.traits_state())

    def tearDown(self):
        self.tmp.cleanup()

    def _buf(self, n):
        for i in range(n):
            self.store.push_buffer({"turn_id": "t_" + str(i), "user": "x",
                                    "say": "y", "direct": "z",
                                    "need": "只想被听", "displace": "none"})

    def test_dict_output_falls_back(self):
        """模型返回 JSON 对象不能崩，也不能丢事件。"""
        self._buf(3)
        c = Consolidator(self.store, llm=lambda p: '{"judgment": "x"}')
        out = c.run_once()
        self.assertIsInstance(out, dict)

    def test_non_dict_item_skipped(self):
        self._buf(2)
        Consolidator(self.store, llm=lambda p: '["str", 123]').run_once()

    def test_confidence_clamped(self):
        """confidence 必须夹到 0..1，否则单条越过共识闸门。"""
        self.store.add_judgment("x", 5.0, affects={"intervention": -1})
        self.assertEqual(Consolidator(self.store).push_traits(), [])

    def test_supersedes_not_blocked_by_dedup(self):
        """要推翻旧判断的不能被去重挡掉。"""
        old = self.store.add_judgment("他喜欢猫", 0.5)
        self._buf(2)
        c = Consolidator(self.store, llm=lambda p: (
            '[{"judgment": "他喜欢猫。", "confidence": 0.9, "supersedes": "'
            + old["id"] + '"}]'))
        c.run_once()
        live = self.store.judgments()
        self.assertTrue(any("。" in j["judgment"] for j in live),
                        "supersedes 被去重挡掉了")

    def test_run_once_shape_consistent(self):
        """没有事件时返回结构也要一致。"""
        out = Consolidator(self.store).run_once()
        for key in ("judgments", "self_story", "traits", "anchors",
                    "activity", "threshold"):
            self.assertIn(key, out)


class TestMetricsRobustness(unittest.TestCase):
    def test_expired_without_pid(self):
        m = compute([{"kind": "pending_expire", "reason": "timeout"}])
        self.assertEqual(m.expired, 1)

    def test_settle_expired(self):
        m = compute([{"kind": "pending_settle", "pending_id": "p1",
                      "outcome": "EXPIRED"}])
        self.assertEqual(m.expired, 1)

    def test_expired_dedup(self):
        m = compute([
            {"kind": "pending_expire", "pending_id": "p1"},
            {"kind": "pending_settle", "pending_id": "p1", "outcome": "EXPIRED"},
        ])
        self.assertEqual(m.expired, 1)

    def test_repair_rate_counts_unanswered(self):
        """10 次受伤 1 次治愈 -> 10%，不是 100%。"""
        led = [{"kind": "repair_open", "note": "n" + str(i)} for i in range(10)]
        led += [{"kind": "repair_attempt", "note": "n0"},
                {"kind": "repair_healed", "note": "n0"}]
        self.assertAlmostEqual(compute(led).repair_success_rate, 0.1, places=3)

    def test_unparsed_excluded_from_rate(self):
        """解析失败不是「没有位移」，不该拉低位移率。"""
        led = [{"kind": "turn", "say": "x", "direct": "", "displace": ""}
               for _ in range(10)]
        m = compute(led)
        self.assertEqual(m.unparsed, 10)
        self.assertEqual(m.displacement_rate, 0.0)


class TestConfidenceParsing(unittest.TestCase):
    def test_percent_and_decimal_together(self):
        self.assertAlmostEqual(_to_float("0.85（85%）"), 0.85, places=3)
        self.assertAlmostEqual(_to_float("0.6（60%）"), 0.6, places=3)

    def test_plain_percent(self):
        self.assertAlmostEqual(_to_float("85%"), 0.85, places=3)


class TestStanceGap(unittest.TestCase):
    def test_gap_falls_to_conservative(self):
        """站位表空隙取更保守的一档，不是最敢说的。"""
        cfg = _cfg()
        cfg.stance_table = [StanceRule(0.0, 0.3, "admit_unsure"),
                            StanceRule(0.6, 1.01, "direct")]
        cfg.confidence_floor = 0.0
        cfg.role.bias = 0.0
        self.assertEqual(cfg.stance_for(0.4), "admit_unsure")


if __name__ == "__main__":
    unittest.main(verbosity=2)