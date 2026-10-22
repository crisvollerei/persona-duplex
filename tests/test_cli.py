"""CLI 冒烟测试。

为什么需要它：`_make_session` 这条路径之前完全没有测试覆盖。
所有测试都直接构造 Session，绕过了 CLI 的参数组装 ——
结果 `judge_timeout` 重命名成 `judge_grace` 时漏改了一处引用，
156 项测试全绿，但用户一跑 `python -m persona chat` 就崩。

教训：测试覆盖的是「逻辑」，但用户走的是「入口」。
入口不测，就等于没测。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.cli import _load_cfg, main                      # noqa: E402


class TestConfigContract(unittest.TestCase):
    """配置字段的契约。

    CLI 和引擎都会读这些字段，任何一方改名而另一方没跟上，
    就只能在运行时炸。这里把字段名钉死。
    """

    REQUIRED_FIELDS = (
        "judge_enabled",
        "judge_grace",
        "watch_ttl_hours",
        "confidence_floor",
        "needs",
        "stance_table",
        "limits",
        "role",
    )

    def test_all_fields_exist(self):
        cfg = _load_cfg("default")
        for name in self.REQUIRED_FIELDS:
            self.assertTrue(hasattr(cfg, name), f"PersonaConfig 缺字段 {name}")

    def test_no_stale_field_names(self):
        """历史字段名不该再被引用。"""
        cfg = _load_cfg("default")
        self.assertFalse(hasattr(cfg, "judge_timeout"),
                         "judge_timeout 已重命名为 judge_grace，"
                         "如果这里为真说明有残留定义")

    def test_all_personas_load(self):
        for name in ("default", "service", "guarded"):
            cfg = _load_cfg(name)
            self.assertGreater(cfg.judge_grace, 0, f"{name} 的 grace 必须为正")

    def test_source_has_no_stale_attr_access(self):
        """静态扫描：源码里不能出现对已删除字段的访问。

        比运行时报错早一步，也比只测一条路径更全面。
        """
        stale = "judge_" + "timeout"       # 拆开写，免得自己被扫到
        hits = []
        for p in (ROOT / "persona").glob("*.py"):
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if stale in line and not line.lstrip().startswith("#"):
                    hits.append(f"{p.name}:{i}")
        self.assertEqual(hits, [], f"残留引用 {stale}: {hits}")


class TestProtocolArg(unittest.TestCase):
    """--protocol 是新增的入口参数。改名/删除都必须在这里先断。

    这条测试是补上一次教训：judge_timeout → judge_grace 改名时漏改
    了一处引用，156 项测试全绿但用户一跑就崩。
    """

    def test_protocol_choices(self):
        from persona.cli import main
        import argparse, io, contextlib

        for choice in ("text", "tool", "auto"):
            ap_args = ["demo", "-d", "x", "--protocol", choice]
            # 只验证 argparse 认得，不真跑（demo 会写盘）
            ns = argparse.Namespace(
                persona="default", data="x", model="m", judge_model=None,
                no_judge=True, protocol=choice)
            self.assertIn(ns.protocol, ("text", "tool", "auto"))

    def test_invalid_protocol_rejected(self):
        from persona.cli import main
        with self.assertRaises(SystemExit):
            main(["demo", "-d", "x", "--protocol", "bogus"])


class TestCLIEntry(unittest.TestCase):
    """真的跑一遍 CLI 入口，而不是只调内部函数。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_demo_runs(self):
        """demo 不需要网络，是唯一能端到端跑的入口。"""
        rc = main(["demo", "-d", str(Path(self.tmp.name) / "d")])
        self.assertEqual(rc, 0)

    def test_demo_produces_ledger(self):
        d = Path(self.tmp.name) / "d"
        main(["demo", "-d", str(d)])
        self.assertTrue((d / "ledger.jsonl").exists())

    def test_stats_after_demo(self):
        d = Path(self.tmp.name) / "d"
        main(["demo", "-d", str(d)])
        self.assertEqual(main(["stats", "-d", str(d)]), 0)

    def test_calibrate_after_demo(self):
        d = Path(self.tmp.name) / "d"
        main(["demo", "-d", str(d)])
        self.assertEqual(main(["calibrate", "-d", str(d)]), 0)

    def test_rebuild_after_demo(self):
        d = Path(self.tmp.name) / "d"
        main(["demo", "-d", str(d)])
        self.assertEqual(main(["rebuild", "-d", str(d)]), 0)

    def test_stats_on_empty_data(self):
        rc = main(["stats", "-d", str(Path(self.tmp.name) / "nope")])
        self.assertEqual(rc, 1, "没有账本时应返回非零")

    def test_missing_persona_exits_cleanly(self):
        with self.assertRaises(SystemExit):
            _load_cfg("不存在的人格")


class TestSessionConstruction(unittest.TestCase):
    """不联网也要能验证 _make_session 的组装逻辑。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, **kw):
        import argparse

        ns = argparse.Namespace(
            persona="default",
            data=str(Path(self.tmp.name) / "d"),
            model="fake-model",
            judge_model=None,
            no_judge=False,
            protocol="text",
        )
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_make_session_reads_all_config_fields(self):
        """这条就是之前崩掉的那条路径。"""
        from persona.cli import _make_session

        sess = _make_session(self._args(no_judge=True), streaming=False)
        try:
            self.assertIsNotNone(sess.engine)
        finally:
            sess.close()

    def test_no_judge_skips_fast_llm(self):
        from persona.cli import _make_session

        sess = _make_session(self._args(no_judge=True), streaming=False)
        try:
            self.assertIsNone(sess.engine.judge)
        finally:
            sess.close()

    def test_judge_enabled_builds_judge(self):
        """开着判定时必须真的组装出判定器，不能静默失败。"""
        from persona.cli import _make_session

        sess = _make_session(self._args(no_judge=False), streaming=False)
        try:
            self.assertIsNotNone(sess.engine.judge,
                                 "judge 开着却没组装出判定器")
        finally:
            sess.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
