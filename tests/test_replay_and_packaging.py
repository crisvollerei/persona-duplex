"""新增能力的回归测试：可重放的累积量、lift 命令、confidence_floor。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.cli import main                                    # noqa: E402
from persona.config import (                                    # noqa: E402
    STANCE_ASK,
    STANCE_DIRECT,
    PersonaConfig,
    StanceRule,
)
from persona.engine import TurnEngine                            # noqa: E402
from persona.llm import ScriptedLLM, make_reply                  # noqa: E402
from persona.store import Store, rebuild_views                   # noqa: E402


def _cfg() -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


class TestStyleReplays(unittest.TestCase):
    """语域基线是「受伤」的尺子。尺子丢了，整条链路都失效。"""

    def test_style_update_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            e = TurnEngine(_cfg(), store, ScriptedLLM([]))
            e.update_style_baseline("今天好累啊")
            recs = [r for r in store.read_ledger() if r["kind"] == "style_update"]
            self.assertTrue(recs, "语域基线变了却没落账")
            self.assertAlmostEqual(recs[-1]["formality"],
                                   store.load_view("style")["formality"], places=4)

    def test_style_rebuilt_from_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            e = TurnEngine(_cfg(), store, ScriptedLLM([]))
            for text in ("今天好累啊", "谢谢，我知道了。", "嗯。"):
                e.update_style_baseline(text)
            live = store.load_view("style")
            for p in store.state_dir.glob("*"):
                p.unlink()
            rebuild_views(store)
            rebuilt = store.load_view("style")
            self.assertAlmostEqual(rebuilt["formality"], live["formality"], places=4)
            self.assertEqual(rebuilt["length_hist"], live["length_hist"])

    def test_old_ledger_without_style_records_keeps_view(self):
        """老账本没有 style_update，重建时不能把现有基线抹掉。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            store.save_view("style", {"formality": 0.4321, "length": 12.5,
                                      "length_hist": [12]})
            rebuild_views(store)
            self.assertAlmostEqual(store.load_view("style")["formality"], 0.4321,
                                   places=4)


class TestConfidenceFloor(unittest.TestCase):
    def test_floor_blocks_bold_tables(self):
        """换一套更敢说的站位表，地板仍然拦得住。"""
        cfg = _cfg()
        cfg.stance_table = [StanceRule(0.0, 1.01, STANCE_DIRECT)]
        self.assertEqual(cfg.stance_for(0.05), STANCE_ASK)
        self.assertEqual(cfg.stance_for(0.9), STANCE_DIRECT)

    def test_floor_zero_disables(self):
        cfg = _cfg()
        cfg.stance_table = [StanceRule(0.0, 1.01, STANCE_DIRECT)]
        cfg.confidence_floor = 0.0
        self.assertEqual(cfg.stance_for(0.05), STANCE_DIRECT)

    def test_default_table_unaffected(self):
        cfg = _cfg()
        self.assertEqual(cfg.stance_for(0.95), STANCE_DIRECT)
        self.assertEqual(cfg.stance_for(0.35), STANCE_ASK)


class TestLiftCommand(unittest.TestCase):
    def test_lift_runs_and_labels_small_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "d"
            self.assertEqual(main(["demo", "-d", str(d)]), 0)
            self.assertEqual(main(["lift", "-d", str(d)]), 0)

    def test_lift_on_empty_data_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["lift", "-d", str(Path(tmp) / "nope")]), 1)


class TestPersonasPackage(unittest.TestCase):
    def test_personas_importable_as_package(self):
        import personas
        self.assertEqual(personas.available(),
                         ["ache", "default", "guarded", "service", "shorekeeper"])
        for name in personas.available():
            self.assertTrue(personas.path_of(name).exists())

    def test_cli_loads_from_package(self):
        from persona.cli import _load_cfg
        for name in ("default", "guarded", "service"):
            self.assertTrue(_load_cfg(name).role.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
