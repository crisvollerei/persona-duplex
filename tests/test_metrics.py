"""指标测试：确认评测表是真的在算，不是摆设。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.metrics import compute, compare_with_without_history, verdicts  # noqa: E402


def _turn(say="嗯。", direct="好", displace="none", need="认同", baseline=None):
    return {"kind": "turn", "say": say, "direct": direct,
            "baseline": direct if baseline is None else baseline,
            "displacement_basis": "persona_baseline", "displace": displace, "need": need}


class TestDisplacementRate(unittest.TestCase):
    def test_counts(self):
        m = compute([_turn(displace="none"), _turn(displace="降级")])
        self.assertEqual(m.turns, 2)
        self.assertEqual(m.displaced, 1)
        self.assertAlmostEqual(m.displacement_rate, 0.5)

    def test_all_none_is_zero(self):
        m = compute([_turn() for _ in range(5)])
        self.assertEqual(m.displacement_rate, 0.0)

    def test_rate_is_descriptive(self):
        ledger = [_turn(displace="降级", say="你最近挺忙的？", direct="我需要你")
                  for _ in range(10)]
        levels = {name: lv for lv, name, _ in verdicts(compute(ledger))}
        self.assertEqual(levels["位移率"], "info")

    def test_low_rate_does_not_imply_no_personality(self):
        ledger = [_turn(displace="none") for _ in range(10)]
        levels = {name: lv for lv, name, _ in verdicts(compute(ledger))}
        self.assertEqual(levels["位移率"], "info")


class TestDeclaration(unittest.TestCase):
    def test_declaration_is_failure(self):
        ledger = [_turn(say="我本来想说点别的，算了。", displace="截断",
                        direct="你根本不在乎")]
        m = compute(ledger)
        self.assertEqual(m.declarations, 1)
        levels = {name: lv for lv, name, _ in verdicts(m)}
        self.assertEqual(levels["声明违规"], "fail")

    def test_clean_has_none(self):
        m = compute([_turn(say="你最近挺忙的？", displace="降级", direct="我需要你")])
        self.assertEqual(m.declarations, 0)


class TestInconsistency(unittest.TestCase):
    def test_claimed_but_identical(self):
        """声称位移，但 baseline 和 say 一样。"""
        m = compute([_turn(say="今天真长", direct="今天真长", displace="降级")])
        self.assertEqual(m.inconsistent, 1)

    def test_real_displacement_ok(self):
        m = compute([_turn(say="今天是不是特别长啊。",
                           direct="你只是想让人知道你有多累", displace="降级")])
        self.assertEqual(m.inconsistent, 0)


class TestCalibration(unittest.TestCase):
    def _ledger(self, pairs):
        out = []
        for i, (conf, outcome) in enumerate(pairs):
            out.append({"kind": "pending_add", "pending_id": f"p_{i}",
                        "confidence": conf})
            out.append({"kind": "pending_settle", "pending_id": f"p_{i}",
                        "outcome": outcome})
        return out

    def test_gap_detects_overconfidence(self):
        """全说 0.9，实际全错 —— 这就是在读空气。"""
        m = compute(self._ledger([(0.9, "REFUTED")] * 5))
        self.assertGreater(m.calibration_gap(), 0.2)

    def test_good_calibration(self):
        m = compute(self._ledger([(0.8, "CONFIRMED")] * 4 + [(0.2, "REFUTED")]))
        self.assertLess(m.calibration_gap(), 0.2)

    def test_delayed_counts_as_hit(self):
        m = compute(self._ledger([(0.6, "CONFIRMED'")]))
        self.assertEqual(m.hits, 1)

    def test_brier(self):
        m = compute(self._ledger([(1.0, "CONFIRMED"), (0.0, "REFUTED")]))
        self.assertAlmostEqual(m.brier, 0.0, places=6)

    def test_buckets(self):
        m = compute(self._ledger([(0.9, "CONFIRMED"), (0.95, "REFUTED"),
                                  (0.4, "CONFIRMED")]))
        labels = [b[0] for b in m.buckets()]
        self.assertIn("0.9–1.0", labels)


class TestHistoryLift(unittest.TestCase):
    def test_lift_positive_when_memory_helps(self):
        full = compute([{"kind": "pending_add", "pending_id": f"p{i}", "confidence": 0.7}
                        for i in range(4)]
                       + [{"kind": "pending_settle", "pending_id": f"p{i}",
                           "outcome": "CONFIRMED"} for i in range(4)])
        bare = compute([{"kind": "pending_add", "pending_id": f"q{i}", "confidence": 0.7}
                        for i in range(4)]
                       + [{"kind": "pending_settle", "pending_id": f"q{i}",
                           "outcome": "REFUTED"} for i in range(4)])
        out = compare_with_without_history(full, bare)
        self.assertGreater(out["lift"], 0)

    def test_zero_difference_does_not_prove_memory_is_useless(self):
        same = [{"kind": "pending_add", "pending_id": "a", "confidence": 0.7},
                {"kind": "pending_settle", "pending_id": "a", "outcome": "CONFIRMED"}]
        out = compare_with_without_history(compute(same), compute(same))
        self.assertEqual(out["lift"], 0.0)
        self.assertFalse(out["causal"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
