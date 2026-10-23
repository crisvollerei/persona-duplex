"""人格参数：anchor / plasticity / drift 的语义。

三个数说的是三件事，混成一个就分不出来了：
    anchor      从哪开始
    plasticity  每次经历能推动多少（推多快）
    drift       总共能偏离多远（能推多远）

为什么不合并：
    plasticity 高 + drift 小 = 容易被触动，但骨子里不变
    plasticity 低 + drift 大 = 平时很稳，遇到大事会变很多
这两个是不同的人。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.config import PersonaConfig, Trait                # noqa: E402


class TestTrait(unittest.TestCase):
    def test_starts_at_anchor(self):
        t = Trait(anchor=0.7, plasticity=0.2, drift=0.1)
        self.assertAlmostEqual(t.value, 0.7)

    def test_plasticity_scales_signal(self):
        """信号一样，可塑性不同，推动量不同 —— 这就是「记忆不承载人格」。"""
        soft = Trait(anchor=0.5, plasticity=0.05, drift=0.5)
        hard = Trait(anchor=0.5, plasticity=0.40, drift=0.5)
        self.assertGreater(hard.nudge(1.0), soft.nudge(1.0) * 5)
        self.assertGreater(soft.nudge(1.0), 0.0)

    def test_zero_plasticity_never_moves(self):
        t = Trait(anchor=0.5, plasticity=0.0, drift=0.5)
        for _ in range(100):
            self.assertEqual(t.nudge(1.0), 0.0)
        self.assertAlmostEqual(t.value, 0.5)

    def test_zero_drift_never_moves(self):
        t = Trait(anchor=0.5, plasticity=1.0, drift=0.0)
        self.assertEqual(t.nudge(1.0), 0.0)

    def test_drift_caps_total_movement(self):
        """可塑性再高，也出不了 drift 圈。"""
        t = Trait(anchor=0.5, plasticity=1.0, drift=0.1)
        for _ in range(3000):
            t.nudge(1.0)
        self.assertLessEqual(t.value, 0.6 + 1e-9, "越过 drift 上限了")
        self.assertGreater(t.value, 0.5998, "3000 轮应该几乎贴到上限")
        self.assertAlmostEqual(t.drift_room[1], 0.6)

    def test_cosine_annealing_slows_near_boundary(self):
        """越靠近边界推得越慢 —— 余弦退火。

        没有退火时，参数会「撞到墙就停住」，之后所有经历都不再有影响。
        有了退火，它在边界附近变硬：还能动，但越来越难动。
        """
        t = Trait(anchor=0.5, plasticity=0.5, drift=0.40)
        self.assertAlmostEqual(t.effective_plasticity(direction=1.0), 0.5, places=6)
        for _ in range(20):
            t.nudge(1.0)
        self.assertLess(t.effective_plasticity(direction=1.0), 0.15, "没有减速")
        self.assertGreater(t.saturation, 0.6)

    def test_single_step_is_a_small_fraction(self):
        """一次推动只占区间一小部分 —— 否则退火来不及展开。

        实测踩过：把 plasticity 直接当步长，drift 小于 plasticity 时
        一次就撞墙，两个不同人格都在 5 轮饱和到 1.00，退火完全失效。
        """
        t = Trait(anchor=0.5, plasticity=1.0, drift=0.10)
        step = t.nudge(1.0)
        self.assertLess(step, 0.10 * 0.25, "单次推动过大")
        self.assertGreater(step, 0.0)

    def test_plasticity_still_orders_speed(self):
        """步长归一化之后，plasticity 仍然决定快慢。"""
        slow = Trait(anchor=0.5, plasticity=0.1, drift=0.4)
        fast = Trait(anchor=0.5, plasticity=0.5, drift=0.4)
        for _ in range(5):
            slow.nudge(1.0); fast.nudge(1.0)
        self.assertGreater(fast.value, slow.value)

    def test_annealing_approaches_boundary_asymptotically(self):
        """退火让它渐近边界，而不是撞上去。

        「人格不会变成另一个人」的数学保证：越靠近极限，
        需要越多次推动才能再往前一点。收敛是几何级数的。
        """
        t = Trait(anchor=0.5, plasticity=0.5, drift=0.20)
        gains = []
        for i in range(60):
            before = t.value
            t.nudge(1.0)
            gains.append(t.value - before)
        # 前半段必然比后半段推进得多
        early = sum(gains[:10])
        late = sum(gains[-10:])
        self.assertGreater(early, late * 3, "越靠近边界推进应该越少")
        self.assertLess(t.value, 0.70 + 1e-6, "不该越过 drift 上限")
        self.assertGreater(t.saturation, 0.5, "60 轮应该走了过半")

    def test_annealing_is_direction_aware(self):
        """贴着上限时，往上几乎推不动，往下还很自由。"""
        t = Trait(anchor=0.6, plasticity=1.0, drift=0.2)
        for _ in range(500):
            t.nudge(1.0)
        self.assertGreater(t.saturation, 0.99, "应该已经贴上上限")
        up_near = t.effective_plasticity(direction=1.0)
        down_near = t.effective_plasticity(direction=-1.0)
        self.assertLess(up_near, 0.01, "向上应该推不动了")
        self.assertGreater(down_near, 0.4, "向下应该还很自由")

    def test_saturation_at_anchor_is_zero(self):
        t = Trait(anchor=0.5, plasticity=0.1, drift=0.2)
        self.assertAlmostEqual(t.saturation, 0.0)

    def test_zero_drift_means_no_annealing_room(self):
        t = Trait(anchor=0.5, plasticity=0.5, drift=0.0)
        self.assertEqual(t.effective_plasticity(direction=1.0), 0.0)
        self.assertEqual(t.nudge(1.0), 0.0)

    def test_drift_and_physical_bounds_both_apply(self):
        """物理边界和漂移范围取交集，谁更紧听谁的。"""
        t = Trait(anchor=0.5, plasticity=1.0, drift=0.6, hi=0.6)
        for _ in range(500):
            t.nudge(1.0)
        self.assertLessEqual(t.value, 0.6 + 1e-9)   # hi 更紧

    def test_signal_clamped(self):
        """信号超过 ±1 也只按 1 算。"""
        a = Trait(anchor=0.5, plasticity=0.2, drift=0.5)
        b = Trait(anchor=0.5, plasticity=0.2, drift=0.5)
        self.assertAlmostEqual(a.nudge(5.0), b.nudge(1.0))
        self.assertAlmostEqual(a.nudge(-5.0), b.nudge(-1.0))

    def test_two_profiles_diverge_under_same_history(self):
        """同一段经历，两个人格分得开 —— 这是整个参数系统的目的。

        注意漂移上限会兜住长期差异：给足轮数，两者最终都撞到同一个天花板。
        差异体现在**到达速度**上，而不是最终位置。
        这也说明 drift 是护栏，不是区分度来源 —— 区分度来自 plasticity。
        """
        stoic = Trait(anchor=0.15, plasticity=0.10, drift=0.10)   # 守岸人式
        volatile = Trait(anchor=0.15, plasticity=0.50, drift=0.10)
        for _ in range(10):
            stoic.nudge(1.0)
            volatile.nudge(1.0)
        self.assertLess(stoic.value, volatile.value, "同一信号下可塑性高的应该推得多")
        self.assertGreater(volatile.value, stoic.value * 1.15)

    def test_drift_differs_across_profiles(self):
        """不同人格不仅速率不同，可漂移的范围也不同。"""
        narrow = Trait(anchor=0.5, plasticity=0.5, drift=0.05)
        wide = Trait(anchor=0.5, plasticity=0.5, drift=0.40)
        for _ in range(400):
            narrow.nudge(1.0)
            wide.nudge(1.0)
        self.assertLessEqual(narrow.value, 0.55 + 1e-9)
        self.assertGreater(wide.value, narrow.value * 1.5)


class TestPersonaLoad(unittest.TestCase):
    def setUp(self):
        self.cfg = PersonaConfig.load(ROOT / "personas" / "shorekeeper.toml")

    def test_traits_loaded(self):
        self.assertIn("warmth", self.cfg.role.traits)
        self.assertIn("learning_openness", self.cfg.role.traits)
        self.assertGreater(len(self.cfg.role.traits), 5)

    def test_emotional_traits_less_plastic(self):
        """守岸人对情感迟钝 —— 情感相关参数应比学习能力更不可塑。"""
        t = self.cfg.role.traits
        self.assertLess(t["affect_range"].plasticity,
                        t["learning_openness"].plasticity)

    def test_intervention_is_a_principle_not_a_habit(self):
        """「不会替他们做决定」是行为准则，应该最不可塑。"""
        t = self.cfg.role.traits
        self.assertLessEqual(t["intervention"].plasticity, 0.15)

    def test_identity_text_loaded(self):
        self.assertIn("守岸人", self.cfg.role.identity)
        self.assertIn("黑海岸", self.cfg.role.identity)

    def test_phrases_by_scene(self):
        p = self.cfg.role.phrases
        self.assertIn("greeting", p)
        self.assertIn("comfort", p)
        self.assertTrue(any("你回来了" in x for x in p["greeting"]))

    def test_special_triggers(self):
        notes = " ".join(s["note"] for s in self.cfg.role.special)
        self.assertIn("弹琴", notes)
        self.assertIn("蓝蝴蝶", notes)


class TestPromptInjection(unittest.TestCase):
    """方案 B：句式按场景注入，不是全塞进 prompt。"""

    def setUp(self):
        self.cfg = PersonaConfig.load(ROOT / "personas" / "shorekeeper.toml")

    def test_greeting_only_on_first_turn(self):
        from persona.prompt import build_system_prompt
        first = build_system_prompt(self.cfg, first_turn=True)
        later = build_system_prompt(self.cfg, first_turn=False)
        self.assertIn("你回来了", first)
        self.assertNotIn("你回来了", later)

    def test_comfort_only_when_needed(self):
        from persona.prompt import build_system_prompt
        comfort = build_system_prompt(self.cfg, need="只想被听")
        other = build_system_prompt(self.cfg, need="要被推一把")
        self.assertIn("不用怕，我在这里", comfort)
        self.assertNotIn("不用怕，我在这里", other)

    def test_identity_always_present(self):
        from persona.prompt import build_system_prompt
        for need in ("", "只想被听", "要被推一把"):
            self.assertIn("守岸人", build_system_prompt(self.cfg, need=need))

    def test_trait_hints_reflect_values(self):
        from persona.prompt import build_system_prompt
        p = build_system_prompt(self.cfg)
        # brevity 0.85 -> 短
        self.assertIn("尽量短", p)
        # intervention 0.20 -> 不给建议
        self.assertIn("不要给建议", p)
        # affect_range 0.15 -> 平稳
        self.assertIn("不要有情绪起伏", p)





class TestPersonaProtocol(unittest.TestCase):
    """三层协议：素回复（无人设）→ 人设化 → 结算。

    为什么需要它：
    文本协议里 direct 和人格在同一个 prompt。实测发现，哪怕明确写了
    「direct 不要带人格」，产出的 direct 仍被污染 —— 两个人格对同一句话的
    「素回复」相似度 0.00，等于人格已经渗进底层了。
    隔离之后素回复相似度 0.29，say 相似度 0.13，两层各司其职。
    """

    def test_raw_layer_sees_no_persona(self):
        """素回复的 prompt 里不能出现任何人设内容。"""
        from persona import protocol
        msgs = protocol.build_raw_messages([], "今天好累")
        joined = " ".join(str(m.get("content")) for m in msgs)
        for word in ("守岸人", "黑海岸", "阿澈", "便利店"):
            self.assertNotIn(word, joined, f"素回复 prompt 里混进了人设：{word}")

    def test_raw_tool_has_no_persona_fields(self):
        """素回复的 schema 只该有「事情怎么办」，不该有角色相关的东西。"""
        from persona import protocol
        cfg = PersonaConfig.load(ROOT / "personas" / "shorekeeper.toml")
        tool = protocol.raw_tool(cfg)
        props = set(tool["function"]["parameters"]["properties"])
        self.assertEqual(props, {"need", "confidence", "raw"})
        self.assertNotIn("say", props)

    def test_raw_tool_need_enum_from_config(self):
        from persona import protocol
        cfg = PersonaConfig.load(ROOT / "personas" / "shorekeeper.toml")
        tool = protocol.raw_tool(cfg)
        enum = tool["function"]["parameters"]["properties"]["need"]["enum"]
        self.assertEqual(enum, list(cfg.needs))

    def test_raw_tool_needs_list(self):
        """素回复这一层需要一次会话来跑。这里只验组装不炸。"""
        # build_raw_messages 只要历史 + 用户输入
        from persona import protocol
        msgs = protocol.build_raw_messages(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "嗯。"}], "今天好累")
        self.assertEqual(len(msgs), 3)
        self.assertNotIn("嗯。", str(msgs))
        self.assertEqual(msgs[-1]["role"], "user")


class TestLengthGuard(unittest.TestCase):
    """brevity 的提示在 prompt 前部，但后面跟着几十行长回复示例 ——
    规则输给示例。所以长度约束要放到最靠近输出的位置。"""

    def test_short_persona_gets_guard(self):
        cfg = PersonaConfig.load(ROOT / "personas" / "shorekeeper.toml")
        self.assertGreaterEqual(cfg.role.traits["brevity"].value, 0.75)
        # 这个断言靠 engine 的 _generate_persona 加 guard，这里只验参数值

    def test_long_persona_does_not(self):
        cfg = PersonaConfig.load(ROOT / "personas" / "ache.toml")
        self.assertLess(cfg.role.traits["brevity"].value, 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
