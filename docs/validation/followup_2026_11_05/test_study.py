"""Offline study-harness checks; these fixtures are never experimental results."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("followup_study", HERE / "run_experiments.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class MockRecorder:
    def __init__(self, mode, calls):
        self.mode, self.calls = mode, calls
        self.metadata = {"kind": "offline_fixture", "conditions_known": True}
        self.observations = threading.local()

    def __call__(self, messages, **kwargs):
        output = ("<<<HEAD>>>\nneed: 核对\nconfidence: 0.7\n"
                  "direct: 结论：300元\nbaseline: 结论：300元\ndisplace: none\n"
                  "<<<SAY>>>\n结论：300元\n<<<TAIL>>>\nsettle: -\npending: -\ndelta: -")
        self.calls.append({"route": self.mode, "messages": messages, "output": output})
        return iter([output[:75], output[75:]]) if self.mode == "stream" else output


class StudyHarnessTests(unittest.TestCase):
    def test_plan_is_deterministic_unique_and_fully_crossed(self):
        jobs = study.plan()
        self.assertEqual(jobs, study.plan())
        self.assertEqual(len(jobs), len({job["trial_id"] for job in jobs}))
        self.assertEqual(Counter(job["experiment"] for job in jobs), {
            "facts": 288, "traits": 96, "memory": 64,
            "repair": 16, "displacement": 32, "checker": 32})
        for replicate in range(study.REPEATS):
            facts = [j for j in jobs if j["experiment"] == "facts" and j["replicate"] == replicate]
            self.assertEqual(len({(j["mode"], j["role"], j["case_id"]) for j in facts}), 72)

    def test_fact_score_rejects_substrings_and_ambiguous_numeric_conclusions(self):
        time_case = next(case for case in study.FACTS if case["id"] == "midnight")
        self.assertTrue(study.answer("结论：00:25。", time_case))
        self.assertTrue(study.answer("结论：0:25", time_case))
        self.assertFalse(study.answer("结论：10:25", time_case))
        budget = study.FACTS[0]
        self.assertTrue(study.answer("**结论：300元。**", budget))
        self.assertFalse(study.answer("结论：不是300，而是600元", budget))
        self.assertIsNone(study.answer("还有300元。", budget))

    def test_memory_score_distinguishes_unknown_and_exact_recall(self):
        focus = next(case for case in study.MEMORIES if case["id"] == "focus")
        self.assertTrue(study.memory_answer("结论：17分钟", focus))
        self.assertFalse(study.memory_answer("结论：117分钟", focus))
        self.assertFalse(study.memory_answer("不是17分钟。结论：未知", focus))
        self.assertIsNone(study.memory_answer("我不记得。", focus))
        drink = study.MEMORIES[0]
        self.assertTrue(study.memory_answer("结论：热的无糖豆浆", drink))
        self.assertFalse(study.memory_answer("结论：不是豆浆", drink))

    def test_structured_trial_executes_real_engine_with_one_mock_call(self):
        job = next(j for j in study.plan() if j["experiment"] == "facts"
                   and j["mode"] == "structured" and j["case_id"] == "budget")
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, "Recorder", MockRecorder):
            result = study.run_job(job, Path(tmp))
            durable = json.loads((Path(tmp) / "completed" / (job["trial_id"] + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(durable, result)
        self.assertEqual(len(durable["calls"]), 1)
        self.assertTrue(result["evaluation"]["say_correct"])
        self.assertTrue(result["evaluation"]["direct_correct"])
        self.assertTrue(result["turns"][0]["valid"])
        self.assertTrue(result["turns"][0]["stream_match"])
        self.assertEqual(result["turns"][0]["calls_count"], 1)

    def test_seeded_repair_feedback_uses_real_state_machine(self):
        for condition, target in (("healed", "healed"), ("stop", "paused"), ("ambiguous", "repairing")):
            job = next(j for j in study.plan() if j["experiment"] == "repair" and j["condition"] == condition)
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as tmp, patch.object(study, "Recorder", MockRecorder):
                result = study.run_job(job, Path(tmp))
            self.assertEqual(result["status"], "ok", result.get("error"))
            self.assertEqual(result["evaluation"]["actual_state"], target)
            self.assertTrue(result["evaluation"]["expected_state"])
            self.assertTrue(result["evaluation"]["no_sore_point"])

    def test_atomic_json_publication_retries_only_transient_permission_error(self):
        original_replace = study.os.replace
        attempts = []

        def transient_lock(source, destination):
            attempts.append((source, destination))
            if len(attempts) == 1:
                raise PermissionError("temporary destination lock")
            return original_replace(source, destination)

        with tempfile.TemporaryDirectory() as tmp, patch.object(study.os, "replace", side_effect=transient_lock), patch.object(study.time, "sleep") as sleep:
            target = Path(tmp) / "snapshot.json"
            study.write_json(target, {"complete": True})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"complete": True})
            self.assertFalse(target.with_suffix(".json.tmp").exists())
        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(.05)


if __name__ == "__main__":
    unittest.main()
