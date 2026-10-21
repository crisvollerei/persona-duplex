"""站位、配置、以及「只改一个数就换个人」的验证。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.config import (  # noqa: E402
    STANCE_ASK,
    STANCE_DIRECT,
    STANCE_UNSURE,
    PersonaConfig,
)


class TestStance(unittest.TestCase):
    def setUp(self):
        self.cfg = PersonaConfig.load(ROOT / "personas" / "default.toml")

    def test_high_confidence_is_direct(self):
        self.assertEqual(self.cfg.stance_for(0.95), STANCE_DIRECT)

    def test_low_confidence_admits(self):
        self.assertEqual(self.cfg.stance_for(0.1), STANCE_UNSURE)

    def test_mid_is_ask(self):
        self.assertEqual(self.cfg.stance_for(0.35), STANCE_ASK)

    def test_bias_is_the_cheapest_personality_knob(self):
        """同一个把握度，只改 bias，站位就变了。

        这是 P1 唯一的验收点：改 bias 从 +0.2 到 -0.2，
        人应该能明显感觉到是两个不同的存在。
        """
        c = PersonaConfig.load(ROOT / "personas" / "default.toml")
        c.role.bias = 0.2
        bold = c.stance_for(0.42)
        c.role.bias = -0.2
        timid = c.stance_for(0.42)
        self.assertNotEqual(bold, timid,
                            "bias 改了但站位没变，人格旋钮失效")

    def test_bias_clamped(self):
        c = PersonaConfig.load(ROOT / "personas" / "default.toml")
        c.role.bias = 5.0
        self.assertLessEqual(c.effective_confidence(0.9), 1.0)
        c.role.bias = -5.0
        self.assertGreaterEqual(c.effective_confidence(0.9), 0.0)


class TestLegibility(unittest.TestCase):
    def test_service_role_cannot_displace(self):
        """客服型 legibility=0，不允许任何位移。"""
        c = PersonaConfig.load(ROOT / "personas" / "service.toml")
        self.assertFalse(c.can_displace("降级"))

    def test_default_can_displace(self):
        c = PersonaConfig.load(ROOT / "personas" / "default.toml")
        self.assertTrue(c.can_displace("降级"))

    def test_none_always_allowed(self):
        c = PersonaConfig.load(ROOT / "personas" / "service.toml")
        self.assertTrue(c.can_displace("none"))


class TestLimits(unittest.TestCase):
    def test_defaults(self):
        c = PersonaConfig.load(ROOT / "personas" / "default.toml")
        self.assertGreater(c.limits.closeness_step, 0)
        self.assertEqual(c.limits.pending_max, 1)

    def test_watch_ttl_parameterised(self):
        """WATCH 挂起时长是参数，默认三天。"""
        c = PersonaConfig.load(ROOT / "personas" / "default.toml")
        self.assertEqual(c.watch_ttl_hours, 72.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
