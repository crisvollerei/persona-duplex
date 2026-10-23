"""校准报告测试。

这个模块的价值在于「能不能真的指出该调什么」。
所以测试重点是：给它造出已知的偏差，看它能不能诊断出来。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.calibrate import (                                 # noqa: E402
    Calibration,
    SettleSample,
    collect,
    findings,
    format_report,
)


def _turn(settles=None, judge=None):
    return {
        "kind": "turn",
        "probe": {
            "judge": judge or {},
            "settle": settles or [],
        },
    }


def _s(overlap=0.2, verdict="CONFIRMED", settle=True, formality=0.3,
       base=0.3, length=20, topic=None, pid="p_1"):
    return {
        "pid": pid, "overlap": overlap, "formality": formality,
        "formality_base": base, "length": length,
        "topic": topic, "verdict": verdict, "settle": settle,
    }


class TestCollect(unittest.TestCase):
    def test_reads_probe(self):
        cal = collect([_turn([_s()])])
        self.assertEqual(len(cal.samples), 1)
        self.assertAlmostEqual(cal.samples[0].overlap, 0.2)

    def test_missing_probe_is_skipped(self):
        cal = collect([{"kind": "turn"}])
        self.assertEqual(cal.samples, [])

    def test_old_ledger_without_probe_does_not_crash(self):
        """升级前的账本没有 probe 字段，不能因此报错。"""
        cal = collect([{"kind": "turn", "need": "认同"},
                       {"kind": "pending_settle", "outcome": "CONFIRMED"}])
        self.assertEqual(cal.samples, [])

    def test_bad_values_skipped(self):
        bad = {"pid": "p", "overlap": "坏", "formality": 0.3,
               "formality_base": 0.3, "length": 10, "topic": None,
               "verdict": "WATCH", "settle": False}
        cal = collect([_turn([bad])])
        self.assertEqual(cal.samples, [])

    def test_judge_sources(self):
        cal = collect([
            _turn(judge={"source": "llm", "elapsed": 0.4}),
            _turn(judge={"source": "late", "elapsed": 1.2}),
        ])
        self.assertEqual(cal.judge_sources, {"llm": 1, "late": 1})
        self.assertAlmostEqual(cal.judge_success_rate(), 0.5)


class TestFindings(unittest.TestCase):
    def _cal(self, samples, judge=None) -> Calibration:
        return collect([_turn(samples, judge=judge)])

    def test_high_overlap_does_not_recommend_forcing_settlement(self):
        """字面重合高也可能正好在否定，不能据此建议降低阈值。

        每条样本必须用**不同 pid** —— collect 按 pid 去重（同一条预言
        挂多轮只记最后一次判定），同一个 pid 会被合并成 1 条。
        """
        samples = [_s(overlap=0.9, verdict="WATCH", settle=False, pid=f"p{i}")
                   for i in range(6)]
        fs = findings(self._cal(samples))
        areas = {f.area: f for f in fs}
        self.assertNotIn("OVERLAP_KEEP", areas)
        self.assertEqual(areas["字面相关性"].level, "info")

    def test_self_labeled_groups_do_not_validate_threshold(self):
        samples = [_s(overlap=0.3, verdict="CONFIRMED", pid=f"c{i}") for i in range(4)]
        samples += [_s(overlap=0.0, verdict="WATCH", settle=False, pid=f"w{i}")
                    for i in range(4)]
        fs = findings(self._cal(samples))
        areas = {f.area: f for f in fs}
        self.assertEqual(areas["字面相关性"].level, "info")

    def test_hurt_threshold_overlap_detected(self):
        """受伤和正常样本的语域升幅重叠 → 会误判。

        门槛：受伤 ≥3、正常 ≥5。
        """
        samples = [_s(verdict="SUSPECTED_HURT", formality=0.5, base=0.3,
                      pid=f"h{i}") for i in range(3)]
        samples += [_s(verdict="CONFIRMED", formality=0.55, base=0.3, pid=f"n{i}")
                    for i in range(5)]
        fs = findings(self._cal(samples))
        areas = {f.area: f for f in fs}
        self.assertIn("语域阈值", areas)
        self.assertEqual(areas["语域阈值"].level, "info")
        self.assertIn("独立", areas["语域阈值"].detail)

    def test_hurt_threshold_clean(self):
        samples = [_s(verdict="SUSPECTED_HURT", formality=0.9, base=0.3,
                      pid=f"h{i}") for i in range(3)]
        samples += [_s(verdict="CONFIRMED", formality=0.35, base=0.3, pid=f"n{i}")
                    for i in range(5)]
        fs = findings(self._cal(samples))
        areas = {f.area: f for f in fs}
        self.assertEqual(areas["语域阈值"].level, "info")

    def test_grace_too_tight(self):
        """判定经常超时 → grace 该放宽。

        样本要 ≥10 条才下结论（少量样本报 info）。
        """
        cal = Calibration(samples=[],
                          judge_elapsed=[1.0, 2.0, 0.5, 3.0] * 3,
                          judge_sources={"late": 12})
        fs = findings(cal)
        areas = {f.area: f for f in fs}
        self.assertEqual(areas["grace"].level, "warn")

    def test_grace_healthy(self):
        cal = Calibration(samples=[], judge_elapsed=[0.1, 0.2, 0.15] * 5,
                          judge_sources={"llm": 15})
        areas = {f.area: f for f in findings(cal)}
        self.assertEqual(areas["grace"].level, "ok")

    def test_grace_small_sample_gives_info(self):
        """样本不足时只报 info，不给结论。"""
        cal = Calibration(samples=[], judge_elapsed=[0.1, 0.2],
                          judge_sources={"llm": 2})
        areas = {f.area: f for f in findings(cal)}
        self.assertEqual(areas["grace"].level, "info")

    def test_semantic_judge_underused(self):
        cal = Calibration(samples=[], judge_elapsed=[0.2] * 10,
                          judge_sources={"late": 8, "llm": 2})
        areas = {f.area: f for f in findings(cal)}
        self.assertEqual(areas["语义判定"].level, "warn")

    def test_empty_gives_hint_not_crash(self):
        fs = findings(Calibration())
        self.assertTrue(any(f.area == "样本" for f in fs))


class TestReport(unittest.TestCase):
    def test_renders_without_crash(self):
        cal = collect([_turn([_s(), _s(verdict="WATCH", settle=False)],
                             judge={"source": "llm", "elapsed": 0.2})])
        text = format_report(cal)
        self.assertIn("校准报告", text)
        self.assertIn("诊断", text)

    def test_empty_report(self):
        text = format_report(Calibration())
        self.assertIn("校准报告", text)

    def test_percentile(self):
        cal = Calibration(samples=[SettleSample("p", 0.1, 0.3, 0.3, 10, None, "WATCH", False)])
        self.assertAlmostEqual(cal.pct([0.0, 0.5, 1.0], 0.5), 0.5)
        self.assertAlmostEqual(cal.pct([], 0.5), 0.0)
        self.assertAlmostEqual(cal.pct([0.7], 0.9), 0.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
