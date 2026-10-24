"""Keep generated readings separate from evidence about the user."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from persona.config import PersonaConfig
from persona.calibrate import collect, findings
from persona.consolidate import Consolidator
from persona.metrics import compute, verdicts
from persona.session import Session, SessionConfig
from persona.store import Store, is_diagnostic_judgment, rebuild_views


ROOT = Path(__file__).resolve().parent.parent


class MemoryDesignAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name))

    def diagnostic(self, text="same model reading", **kwargs):
        return self.store.add_judgment(text, .8, source="heuristic",
                                       evidence_scope="need_count", **kwargs)

    def events(self):
        return [dict(turn_id=f"t_{i}", user=f"user evidence {i}",
                     need="wants advice", direct="an interpretation", say="advice")
                for i in range(2)]

    def test_diagnostic_selector_recognizes_old_and_new_provenance(self):
        self.assertTrue(is_diagnostic_judgment({"source": "heuristic"}))
        self.assertTrue(is_diagnostic_judgment({"source": "heuristic_fallback"}))
        self.assertTrue(is_diagnostic_judgment({"evidence_scope": "need_count"}))
        self.assertFalse(is_diagnostic_judgment({"source": "explicit"}))
        self.assertFalse(is_diagnostic_judgment({"source": "legacy_unknown"}))

    def test_foreground_filters_before_limit_without_removing_ledger_records(self):
        expected = [self.store.add_judgment(f"user preference {i}", .6)["id"]
                    for i in range(6)]
        for i in range(8):
            self.diagnostic(f"generated reading {i}")
        self.assertTrue(all(is_diagnostic_judgment(j) for j in self.store.judgments()))
        self.assertEqual([j["id"] for j in self.store.user_judgments()], expected[-5:])
        self.assertEqual(len(self.store.judgments(limit=None)), 14)
        self.assertEqual(len(self.store.user_judgments(limit=None)), 6)
        self.assertEqual(self.store.user_judgments(limit=0), [])
        rebuild_views(self.store)
        self.assertEqual([j["id"] for j in self.store.user_judgments()], expected[-5:])

    def test_legacy_import_preserves_diagnostic_provenance(self):
        records = [{"id": "j_0001", "judgment": "old heuristic reading",
                    "confidence": .8, "source": "heuristic_fallback",
                    "evidence_scope": "need_count"},
                   {"id": "j_0002", "judgment": "explicit preference", "confidence": .6}]
        self.store.judgments_path.write_text(
            "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
        self.assertEqual(len(self.store.judgments()), 2)
        self.assertEqual([j["judgment"] for j in self.store.user_judgments()],
                         ["explicit preference"])

    def test_frozen_memory_snapshots_only_user_judgments(self):
        expected = self.store.add_judgment("a tentative preference", .6)
        for i in range(6):
            self.diagnostic(f"generated reading {i}")
        cfg = PersonaConfig.load(ROOT / "personas" / "default.toml")
        with Session(cfg, self.store, lambda messages: "unused",
                     options=SessionConfig(memory_mode="frozen", consolidate_on_exit=False)) as session:
            self.assertEqual([j["id"] for j in session.engine.memory_snapshot], [expected["id"]])
            self.store.add_judgment("later preference", .6)
            self.assertEqual([j["id"] for j in session.engine.memory_snapshot], [expected["id"]])

    def test_heuristic_records_are_diagnostics_not_user_confidence(self):
        for event in self.events():
            self.store.push_buffer(event)
        result = Consolidator(self.store).run_once()
        self.assertEqual(result["judgments"], 1)
        record, = self.store.judgments()
        self.assertTrue(is_diagnostic_judgment(record))
        self.assertEqual(record["confidence"], 0.0)
        self.assertEqual(record["evidence_ids"], ["t_0", "t_1"])
        self.assertIn("wants advice", record["judgment"])
        self.assertIn("2", record["judgment"])
        self.assertEqual(self.store.user_judgments(), [])
        story = self.store.self_story_path.read_text(encoding="utf-8")
        self.assertIn("读法诊断", story)
        self.assertIn("不是用户倾向的证据", story)
        self.assertNotIn("我开始觉得", story)

    def test_diagnostic_record_does_not_deduplicate_a_real_judgment(self):
        text = "user prefers listening"
        self.diagnostic(text)
        consolidator = Consolidator(self.store, llm=lambda prompt: json.dumps([
            {"judgment": text, "confidence": .7, "evidence_ids": ["t_0", "t_1"]}]))
        added = consolidator._consolidate(self.events())
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["source"], "llm")
        self.assertEqual(len(self.store.judgments()), 2)

    def test_diagnostic_cannot_supersede_user_judgment(self):
        user_record = self.store.add_judgment("preference", .6)
        diagnostic = self.diagnostic("reading", supersedes=user_record["id"])
        self.assertIsNone(diagnostic["supersedes"])
        self.assertEqual([j["id"] for j in self.store.user_judgments()], [user_record["id"]])

    def test_model_memory_prompt_keeps_user_evidence_and_provenance_distinct(self):
        self.store.add_judgment("known tentative preference", .6, source="explicit")
        self.diagnostic("SHOULD_NOT_APPEAR_AS_USER_MEMORY")
        events = self.events()
        events[0]["user"] = "a" * 100 + " I explicitly reject that interpretation"
        prompt = Consolidator(self.store)._consolidation_prompt(events)
        self.assertNotIn("SHOULD_NOT_APPEAR_AS_USER_MEMORY", prompt)
        self.assertIn("known tentative preference", prompt)
        self.assertIn("来源 explicit", prompt)
        self.assertIn(events[0]["user"], prompt)
        self.assertIn("反复出现同一个模型读法，不算重复的用户证据", prompt)
        self.assertIn("need（非用户证据）", prompt)
        self.assertIn("direct（非用户证据）", prompt)
        self.assertIn("SAY（非用户证据）", prompt)

    def test_story_model_does_not_treat_generated_text_as_new_user_evidence(self):
        prompts = []
        consolidator = Consolidator(self.store, llm=lambda prompt: prompts.append(prompt) or "story")
        consolidator._story_entry_llm(self.events(), [{"judgment": "tentative preference"}])
        self.assertIn("不是新增的用户证据", prompts[0])
        self.assertIn("诊断读法不能改写成用户倾向", prompts[0])
        self.assertIn("模型表达，非用户证据", prompts[0])
        self.assertNotIn("你是系统扮演", prompts[0])

    def test_legacy_diagnostics_do_not_vote_in_explicit_trait_experiment(self):
        cfg = PersonaConfig.load(ROOT / "personas" / "default.toml")
        self.store.init_traits(cfg.role.traits_state())
        before = self.store.traits()
        self.diagnostic("one", affects={"intervention": -1})
        self.diagnostic("two", affects={"intervention": -1})
        self.assertEqual(Consolidator(self.store).push_traits(), [])
        self.assertEqual(self.store.traits(), before)


class MemoryReportingAlignment(unittest.TestCase):
    def test_reports_count_diagnostics_separately_from_user_judgments(self):
        records = [{"kind": "judgment_add", "source": "heuristic"},
                   {"kind": "judgment_add", "source": "heuristic_fallback", "supersedes": "j_1"},
                   {"kind": "judgment_add", "evidence_scope": "need_count"},
                   {"kind": "judgment_add", "source": "llm", "supersedes": "j_2"}]
        metrics = compute(records)
        self.assertEqual(metrics.judgments, 1)
        self.assertEqual(metrics.judgments_superseded, 1)
        self.assertEqual(metrics.diagnostic_judgments, 3)
        self.assertIn("读法诊断", [item[1] for item in verdicts(metrics)])
        calibration = collect(records + [{"kind": "turn"}] * 10)
        self.assertEqual(calibration.judgments, 1)
        self.assertEqual(calibration.diagnostic_judgments, 3)
        self.assertTrue(any("不计用户判断产出" in f.detail for f in findings(calibration)))

    def test_untagged_legacy_records_keep_their_existing_classification(self):
        records = [{"kind": "judgment_add"}]
        self.assertEqual(compute(records).judgments, 1)
        self.assertEqual(collect(records).judgments, 1)
        self.assertEqual(compute(records).diagnostic_judgments, 0)


if __name__ == "__main__":
    unittest.main()
