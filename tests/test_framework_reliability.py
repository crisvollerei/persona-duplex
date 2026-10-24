"""Fault injection and experiment contracts, without external model requests."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from persona.audit import model_info
from persona import protocol
from persona.calibrate import collect, findings
from persona.config import PersonaConfig
from persona.consolidate import Consolidator
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply, openai_compatible, openai_tool_call
from persona.metrics import compute, verdicts
from persona.parse import parse
from persona.prompt import build_system_prompt, phrase_block
from persona.repair import judge_repair_response
from persona.session import Session, SessionConfig
from persona.store import Store, rebuild_views
from persona.topicjudge import AsyncJudge, JudgeRequest, LLMJudge


ROOT = Path(__file__).resolve().parent.parent


def config(name="default"):
    return PersonaConfig.load(ROOT / "personas" / f"{name}.toml")


def reply(say="Hello.", tail="settle: -", displace="none"):
    return make_reply("need: listen\nconfidence: .6\ndirect: neutral\n"
                      f"baseline: ordinary\ndisplace: {displace}", say, tail)


class Case(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(tmp.name)

    def rows(self, kind):
        return [r for r in self.store.read_ledger() if r["kind"] == kind]

    def push(self, count=2):
        for i in range(count):
            self.store.push_buffer({"turn_id": f"t_{i}", "need": "listen",
                                    "user": "hello", "say": "Hello."})

    def buffers(self):
        return list(self.store.buffer_dir.glob("*.json"))


class TestDurableCommit(Case):
    def test_partial_tail_does_not_swallow_next_record(self):
        self.store.ledger_path.write_bytes(b'{"kind":"broken","text":"\xe4')
        self.store.append("probe", {"number": 1})
        self.store.append("probe", {"number": 2})
        self.assertEqual([r["number"] for r in self.rows("probe")], [1, 2])

    def test_complete_line_without_newline_is_preserved(self):
        self.store.ledger_path.write_bytes(b'{"kind":"probe","number":0}')
        self.store.append("probe", {"number": 1})
        self.assertEqual([r["number"] for r in self.rows("probe")], [0, 1])

    def test_invalid_utf8_is_not_silently_replaced_in_an_event(self):
        self.store.ledger_path.write_bytes(b'{"kind":"probe","text":"\xff"}\n')
        self.store.append("probe", {"number": 1})
        self.assertEqual(len(self.rows("probe")), 1)

    def test_complete_write_followed_by_fsync_error_is_recovered_as_committed(self):
        original = self.store._append_record
        def ambiguous(rec):
            original(rec)
            raise OSError("write completed but caller did not receive acknowledgment")
        with patch.object(self.store, "_append_record", side_effect=ambiguous):
            with self.assertRaises(OSError):
                with self.store.transaction():
                    self.store.append("probe", {"number": 1})
                    self.store.save_view("relation", {"closeness": .4})
        recovered = Store(self.store.root)
        self.assertEqual(recovered.relation()["closeness"], .4)
        self.assertEqual(len(self.rows("probe")), 1)

    def test_transaction_exception_rolls_back_views_events_and_outbox(self):
        self.store.save_view("relation", {"closeness": .3})
        with self.assertRaises(RuntimeError):
            with self.store.transaction():
                self.store.save_view("relation", {"closeness": .4})
                self.store.append("probe", {})
                self.store.push_buffer({"turn_id": "t_1"})
                self.assertEqual(self.store.relation()["closeness"], .4)
                raise RuntimeError("before commit")
        self.assertEqual(self.store.relation()["closeness"], .3)
        self.assertEqual(self.rows("probe"), [])
        self.assertEqual(self.buffers(), [])

    def test_restart_finishes_a_durable_commit_after_publication_failure(self):
        original = self.store._write_view
        def fail(name, data):
            if name == "relation":
                raise OSError("temporary publication failure")
            return original(name, data)
        with patch.object(self.store, "_write_view", side_effect=fail):
            with self.assertRaises(OSError):
                with self.store.transaction():
                    self.store.save_view("relation", {"closeness": .4})
                    self.store.append("probe", {"number": 1})
                    self.store.push_buffer({"turn_id": "t_1"})
        recovered = Store(self.store.root)
        self.assertEqual(recovered.relation()["closeness"], .4)
        self.assertEqual(len(self.buffers()), 1)
        Store(self.store.root)
        self.assertEqual(len(self.buffers()), 1)
        self.assertEqual(len(self.rows("probe")), 1)

    def test_failed_turn_append_cannot_leave_relation_or_pending_changes(self):
        original = self.store.append
        def fail(kind, payload):
            if kind == "turn":
                raise OSError("turn write failed")
            return original(kind, payload)
        tail = "settle: -\ndelta:\n  closeness: .05\npending:\n  claim: c\n  confidence: .7\n  predicts: p"
        llm = ScriptedLLM([reply(tail=tail)])
        engine = TurnEngine(config(), self.store, llm)
        with patch.object(self.store, "append", side_effect=fail):
            with self.assertRaises(OSError):
                engine.turn("hello")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(self.store.relation()["closeness"], .3)
        self.assertEqual(self.store.pendings(), [])
        self.assertEqual(self.buffers(), [])
        self.assertEqual(self.rows("turn"), [])
        self.assertEqual(len(self.rows("generation_failure")), 1)
        rebuild_views(self.store)
        self.assertEqual(self.store.relation()["closeness"], .3)

    def test_partial_commit_bytes_are_not_replayed(self):
        original = self.store._append_record
        def fail(rec):
            with self.store.ledger_path.open("ab") as f:
                f.write(json.dumps(rec).encode()[:40])
            raise OSError("interrupted append")
        with patch.object(self.store, "_append_record", side_effect=fail):
            with self.assertRaises(OSError):
                with self.store.transaction():
                    self.store.append("probe", {})
                    self.store.save_view("relation", {"closeness": .9})
        recovered = Store(self.store.root)
        self.assertEqual(recovered.relation()["closeness"], .3)
        recovered.append("probe", {"number": 2})
        self.assertEqual(len(self.rows("probe")), 1)

    def test_rebuild_uses_actual_custom_limits_without_cli_configuration(self):
        cfg = config("service")
        engine = TurnEngine(cfg, self.store, ScriptedLLM([
            reply(tail=f"settle: -\ndelta:\n  unresolved: issue{i}") for i in range(4)]))
        for _ in range(4):
            engine.turn("hello")
        before = self.store.relation()
        self.assertEqual(len(before["unresolved"]), 2)
        rebuild_views(self.store)
        self.assertEqual(self.store.relation(), before)

    def test_repair_rebuild_preserves_capacity_and_attempts(self):
        cfg = config()
        cfg.limits.unresolved_max = 2
        engine = TurnEngine(cfg, self.store, ScriptedLLM([]))
        for seq in range(1, 5):
            engine._turn_seq = seq
            engine._open_repair({})
        before = self.store.relation()
        self.assertEqual(len(before["unresolved"]), 2)
        rebuild_views(self.store)
        self.assertEqual(self.store.relation(), before)

    def test_consumed_memory_is_not_requeued_when_checkpoints_are_lost(self):
        engine = TurnEngine(config(), self.store, ScriptedLLM([reply(), reply()]))
        engine.turn("hello")
        engine.turn("hello again")
        Consolidator(self.store).run_once()
        self.assertEqual(self.buffers(), [])
        self.store._view_path("commit_checkpoint").unlink()
        recovered = Store(self.store.root)
        self.assertEqual(list(recovered.buffer_dir.glob("*.json")), [])
        self.assertEqual(len(self.rows("judgment_add")), 1)


class TestMemoryBatches(Case):
    def test_invalid_memory_fields_cannot_turn_into_invented_confidence_or_text(self):
        self.push()
        for data in ({"judgment": None, "confidence": .6}, {"judgment": "x"},
                     {"judgment": "x", "confidence": float("nan")},
                     {"judgment": "x", "confidence": True}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                Consolidator(self.store, llm=lambda p: json.dumps([data])).run_once()
            self.assertEqual(self.store.judgments(), [])
            self.assertEqual(len(self.buffers()), 2)

    def test_model_failure_does_not_switch_to_counting(self):
        self.push()
        calls = []
        def fail(prompt):
            calls.append(prompt)
            raise RuntimeError("network")
        with self.assertRaises(RuntimeError):
            Consolidator(self.store, llm=fail).run_once()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.buffers()), 2)
        self.assertEqual(self.store.judgments(), [])
        self.assertEqual(len(self.rows("consolidate_error")), 1)
        self.assertFalse(self.store.self_story_path.exists())

    def test_invalid_json_preserves_evidence(self):
        self.push()
        with self.assertRaises(ValueError):
            Consolidator(self.store, llm=lambda prompt: "not json").run_once()
        self.assertEqual(len(self.buffers()), 2)
        self.assertEqual(self.rows("memory_batch_commit"), [])

    def test_story_request_failure_commits_neither_half(self):
        self.push()
        responses = iter(['[{"judgment":"user likes listening","confidence":0.7}]'])
        con = Consolidator(self.store, llm=lambda prompt: next(responses))
        with self.assertRaises(StopIteration):
            con.run_once()
        self.assertEqual(self.store.judgments(), [])
        self.assertEqual(len(self.buffers()), 2)
        con.llm = lambda prompt: ('[{"judgment":"user likes listening","confidence":0.7}]'
                                 if "supersedes" in prompt else "An observation.")
        out = con.run_once()
        self.assertEqual(out["judgments"], 1)
        self.assertTrue(out["self_story"])
        self.assertEqual(len(self.rows("story_entry")), 1)
        self.assertEqual(self.buffers(), [])

    def test_story_commit_failure_rolls_back_judgments_before_retry(self):
        self.push()
        con = Consolidator(self.store)
        with patch.object(con, "_rewrite_story", side_effect=OSError("temporary disk error")):
            with self.assertRaises(OSError):
                con.run_once()
        self.assertEqual(self.store.judgments(), [])
        self.assertEqual(len(self.buffers()), 2)
        con.run_once()
        self.assertEqual(len(self.store.judgments()), 1)
        self.assertEqual(len(self.rows("story_entry")), 1)

    def test_cleanup_failure_cannot_double_consume(self):
        self.push()
        con = Consolidator(self.store)
        with patch.object(self.store, "commit_drained", side_effect=OSError("cleanup")):
            with self.assertRaises(OSError):
                con.run_once()
        con.run_once()
        self.assertEqual(len(self.rows("memory_batch_commit")), 1)
        self.assertEqual(len(self.store.judgments()), 1)
        self.assertEqual(len(self.rows("story_entry")), 1)
        self.assertEqual(self.buffers(), [])

    def test_judgment_and_story_have_batch_and_turn_provenance(self):
        self.push()
        Consolidator(self.store).run_once()
        judgment = self.store.judgments()[0]
        batch = self.rows("memory_batch_commit")[0]
        self.assertEqual(judgment["evidence_ids"], ["t_0", "t_1"])
        self.assertEqual(judgment["source"], "heuristic")
        self.assertEqual(judgment["batch_id"], batch["batch_id"])
        self.assertEqual(self.rows("story_entry")[0]["batch_id"], batch["batch_id"])

    def test_foreign_evidence_ids_are_rejected(self):
        self.push()
        data = [{"judgment": "x", "confidence": .6, "evidence_ids": ["not-in-batch"]}]
        with self.assertRaises(ValueError):
            Consolidator(self.store, llm=lambda prompt: json.dumps(data)).run_once()
        self.assertEqual(self.store.judgments(), [])
        self.assertEqual(len(self.buffers()), 2)

    def test_malformed_buffer_is_quarantined_without_blocking_valid_events(self):
        (self.store.buffer_dir / "broken.json").write_text("[]", encoding="utf-8")
        self.push()
        Consolidator(self.store).run_once()
        self.assertTrue((self.store.buffer_dir / "broken.invalid").exists())
        self.assertEqual(len(self.rows("buffer_error")), 1)
        self.assertEqual(len(self.store.judgments()), 1)

    def test_story_model_does_not_hold_the_foreground_store_lock(self):
        self.push()
        entered, release, read_done = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def model(prompt):
            if "supersedes" in prompt:
                return '[{"judgment":"tentative preference","confidence":0.6}]'
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release story request")
            return "An observation."
        def run():
            try:
                Consolidator(self.store, llm=model).run_once()
            except Exception as exc:
                errors.append(exc)
        writer = threading.Thread(target=run)
        writer.start()
        reader = None
        try:
            self.assertTrue(entered.wait(2))
            reader = threading.Thread(target=lambda: (self.store.relation(), read_done.set()))
            reader.start()
            self.assertTrue(read_done.wait(1), "background model request held store lock")
        finally:
            release.set()
            writer.join(3)
            if reader:
                reader.join(3)
        self.assertEqual(errors, [])


class TestSampleValidity(Case):
    def test_garbage_tail_and_incomplete_pending_are_not_valid_samples(self):
        for tail in ("garbage", "pending:\n  claim: unfinished", "pending:\n  predicts: next"):
            with self.subTest(tail=tail):
                self.assertFalse(parse(reply(tail=tail)).validity()["valid"])

    def test_explicit_no_pending_is_valid(self):
        for tail in ("settle: -\npending: -", "settle: -\npending: none"):
            self.assertTrue(parse(reply(tail=tail)).validity()["valid"])

    def test_bad_confidence_is_invalid_not_a_clamped_valid_sample(self):
        for confidence in ("1.8", "-0.2", "nan", "maybe 0.8", "1e-3", "120%"):
            with self.subTest(confidence=confidence):
                result = parse(reply().replace("confidence: .6", "confidence: " + confidence))
                self.assertIn("invalid_confidence", result.validity()["reasons"])

    def test_decimal_percent_and_annotated_confidence_remain_supported(self):
        for confidence, value in ((".7", .7), ("85%", .85), ("1%", .01), ("0.85（85%）", .85)):
            with self.subTest(confidence=confidence):
                result = parse(reply().replace("confidence: .6", "confidence: " + confidence))
                self.assertTrue(result.validity()["valid"])
                self.assertAlmostEqual(result.head.confidence, value)

    def test_invalid_pending_confidence_never_enters_the_pending_book(self):
        for confidence in ("nan", "1.5", "not sure"):
            with self.subTest(confidence=confidence):
                tail = f"settle: -\npending:\n  claim: c\n  confidence: {confidence}\n  predicts: p"
                result = TurnEngine(config(), self.store, ScriptedLLM([reply(tail=tail)])).turn("hello")
                self.assertIsNone(result.pending_added)
                self.assertIn("invalid_pending_confidence", result.validity["reasons"])

    def test_tool_confidence_does_not_become_valid_by_clamping(self):
        for value in (True, float("nan"), 2, "0.7"):
            head = protocol.parse_inner({"need": "listen", "confidence": value, "direct": "neutral",
                                         "baseline": "ordinary", "displace": "none"})
            self.assertIn("invalid_confidence", head.errors)
            tail = protocol.parse_tail({"pending": {"claim": "c", "confidence": value, "predicts": "p"}})
            self.assertIn("invalid_pending_confidence", tail.pending.errors)

    def test_missing_head_fields_are_unknown_not_normal_none(self):
        raw = make_reply("baseline: ordinary", "Hello.", "settle: -")
        result = TurnEngine(config(), self.store, ScriptedLLM([raw])).turn("hello")
        self.assertEqual(result.say, "Hello.")
        self.assertFalse(result.validity["valid"])
        self.assertIn("missing_displace", result.validity["reasons"])
        self.assertEqual(compute(self.store.read_ledger()).invalid_turns, 1)
        self.assertEqual(self.buffers(), [])

    def test_invalid_enum_is_flagged_even_after_compatibility_normalization(self):
        result = TurnEngine(config(), self.store, ScriptedLLM([reply(displace="invalid")])).turn("hello")
        self.assertFalse(result.validity["valid"])
        self.assertIn("invalid_displace", result.validity["reasons"])

    def test_interrupted_displacement_cannot_enter_the_rate(self):
        def broken(messages):
            yield reply(displace="悬置")
            raise RuntimeError("disconnect")
        seen = []
        result = TurnEngine(config(), self.store, ScriptedLLM([]), stream_llm=broken).turn(
            "hello", on_delta=seen.append)
        self.assertEqual(result.say, "".join(seen))
        self.assertFalse(result.validity["valid"])
        metrics = compute(self.store.read_ledger())
        self.assertEqual(metrics.invalid_turns, 1)
        self.assertEqual(metrics.displaced, 0)
        self.assertEqual(metrics.generation_failures, 1)

    def test_failure_does_not_reuse_turn_id_on_restart(self):
        engine = TurnEngine(config(), self.store, ScriptedLLM([]))
        with self.assertRaises(RuntimeError):
            engine.turn("hello")
        new = TurnEngine(config(), Store(self.store.root), ScriptedLLM([reply()]))
        result = new.turn("hello again")
        self.assertEqual(result.turn_id, "t_0002")

    def test_metrics_keep_delivery_failure_and_valid_turn_counts_separate(self):
        row = {"kind": "turn", "baseline": "ordinary", "displacement_basis": "persona_baseline",
               "say": "Hello.", "displace": "悬置"}
        metrics = compute([row, {**row, "validity": {"valid": False}},
                           {**row, "probe": {"output_blocked": True}}])
        self.assertEqual(metrics.turns, 3)
        self.assertEqual(metrics.invalid_turns, 2)
        self.assertEqual(metrics.displaced, 1)
        self.assertEqual(metrics.displacement_rate, 1)


class TestExperimentInputs(Case):
    def test_extra_cannot_replace_audited_input_or_routing(self):
        for key in ("model", "messages", "stream", "tools", "tool_choice"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                openai_compatible("fake", extra={key: "override"})

    def test_tool_response_must_match_the_requested_stage(self):
        payload = {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "wrong_stage", "arguments": "{}"}}]}}]}
        with patch("persona.llm._post", return_value=payload):
            with self.assertRaises(RuntimeError):
                openai_tool_call("fake")([], tool={"function": {"name": "inner"}})

    def test_run_and_actual_request_are_recorded_without_extra_calls(self):
        llm = ScriptedLLM([reply()])
        engine = TurnEngine(config(), self.store, llm)
        engine.turn("hello")
        row = self.rows("turn")[0]
        run = self.rows("run_start")[0]
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(row["run_id"], run["run_id"])
        self.assertEqual(row["calls"][0]["messages"], llm.calls[0])
        self.assertEqual(row["calls"][0]["status"], "complete")
        self.assertEqual(row["calls"][0]["output"], reply())
        self.assertIn("config_hash", run)
        self.assertIn("source_hash", run)
        self.assertFalse(run["models"]["text"]["conditions_known"])

    def test_transport_conditions_do_not_contain_credentials(self):
        fn = openai_compatible("fake", base_url="https://user:secret@example.test/v1?token=hidden",
                               api_key="api-secret", temperature=.2)
        text = json.dumps(model_info(fn))
        for secret in ("secret", "hidden", "token="):
            self.assertNotIn(secret, text)
        self.assertEqual(model_info(fn)["temperature"], .2)

    def test_restart_restores_history_and_next_prompt(self):
        control = TurnEngine(config(), self.store, ScriptedLLM([reply(), reply()]))
        control.turn("first")
        resumed_llm = ScriptedLLM([reply()])
        resumed = TurnEngine(config(), Store(self.store.root), resumed_llm)
        original_llm = control.llm.__wrapped__
        control.turn("second")
        resumed.turn("second")
        self.assertEqual(original_llm.calls[-1], resumed_llm.calls[-1])

    def test_persona_a_b_a_restores_the_requested_initial_parameters(self):
        for name in ("shorekeeper", "ache", "shorekeeper"):
            cfg = config(name)
            with Session(cfg, self.store, ScriptedLLM([]), options=SessionConfig(
                    consolidate_interval=999, consolidate_on_exit=False)) as session:
                session.engine._sync_traits()
                self.assertEqual(self.store.persona_name(), cfg.role.name)
        self.assertAlmostEqual(cfg.role.trait("warmth"), .7)

    def test_frozen_memory_does_not_drift_or_run_background(self):
        self.store.add_judgment("old", .6)
        llm = ScriptedLLM([reply()])
        with Session(config(), self.store, llm, options=SessionConfig(
                memory_mode="frozen", condition="fixed-memory")) as session:
            self.assertIsNone(session.consolidator._thread)
            self.store.add_judgment("new", .7)
            session.say("hello")
            self.assertNotIn("- new", llm.calls[-1][0]["content"])
            with self.assertRaises(RuntimeError):
                session.consolidate_now()
        self.assertEqual(len(self.buffers()), 1)
        self.assertEqual(self.rows("run_start")[0]["options"]["session"]["condition"], "fixed-memory")


class TestDeliveryLifecycle(Case):
    def test_callback_failure_never_retries_or_starts_observation_in_any_protocol(self):
        for mode in ("text", "persona", "tool"):
            with self.subTest(mode=mode):
                calls = []
                def tool(messages, **kwargs):
                    calls.append(kwargs["tool"]["function"]["name"])
                    return {"need": "listen", "confidence": .6, "raw": "neutral", "constraints": [],
                            "direct": "neutral", "baseline": "ordinary", "displace": "none"}
                def consumer(piece):
                    raise ValueError("display failed")
                llm = ScriptedLLM([])
                engine = TurnEngine(config(), self.store, llm, tool_llm=tool, protocol=mode,
                                    stream_llm=lambda messages: iter([reply()]))
                with self.assertRaisesRegex(ValueError, "display failed"):
                    engine.turn("hello", on_delta=consumer)
                self.assertEqual(len(calls), 0 if mode == "text" else 1)
                self.assertEqual(llm.calls, [])
                self.assertEqual(self.rows("turn"), [])
                failure = self.rows("generation_failure")[-1]
                self.assertEqual(failure["stage"], "delivery")
                self.assertEqual(failure["delivered"], "")

    def test_close_waits_for_active_turn_then_flushes_its_outbox(self):
        entered, release, closing, closed = [threading.Event() for _ in range(4)]
        errors = []
        def model(messages):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release generation")
            return reply()
        session = Session(config(), self.store, model, options=SessionConfig(consolidate_interval=999))
        self.addCleanup(session.close)
        def say():
            try:
                session.say("hello")
            except Exception as exc:
                errors.append(exc)
        def close():
            closing.set()
            session.close()
            closed.set()
        speaker, closer = threading.Thread(target=say), threading.Thread(target=close)
        speaker.start()
        try:
            self.assertTrue(entered.wait(2))
            closer.start()
            self.assertTrue(closing.wait(1))
            self.assertFalse(closed.wait(.05))
        finally:
            release.set()
            speaker.join(3)
            if closer.ident is not None:
                closer.join(3)
        self.assertEqual(errors, [])
        self.assertTrue(closed.is_set())
        self.assertEqual(self.buffers(), [])
        self.assertEqual(len(self.rows("memory_batch_commit")), 1)

    def test_callback_cannot_reenter_or_close_an_active_session(self):
        llm = ScriptedLLM([])
        with Session(config(), self.store, llm, stream_llm=lambda messages: iter([reply()]),
                     options=SessionConfig(consolidate_on_exit=False)) as session:
            def consumer(piece):
                with self.assertRaises(RuntimeError):
                    session.close()
                with self.assertRaises(RuntimeError):
                    session.say("nested")
            session.say("hello", on_delta=consumer)
        self.assertEqual(len(self.rows("turn")), 1)


class TestUncertaintyAndControls(Case):
    def test_minimal_configuration_has_a_valid_default_stance_table(self):
        cfg = PersonaConfig.from_dict({})
        cfg.validate()
        self.assertEqual(cfg.stance_for(.9), "direct")

    def test_register_signal_never_invents_confidence(self):
        engine = TurnEngine(config(), self.store, ScriptedLLM([]))
        engine._turn_seq = 3
        engine._last_formality_rise = .5
        engine._open_repair({})
        row = self.rows("repair_open")[0]
        self.assertEqual(row["level"], "soft")
        self.assertEqual(row["confidence"], 0)
        self.assertEqual(row["source_turn_seq"], 2)
        self.assertEqual(row["status"], "tentative")

    def test_prediction_confidence_is_not_hurt_confidence(self):
        engine = TurnEngine(config(), self.store, ScriptedLLM([]))
        engine._turn_seq = 2
        engine._last_formality_rise = .7
        engine._open_repair({"confidence": .99})
        row = self.rows("repair_open")[0]
        self.assertEqual(row["level"], "soft")
        self.assertEqual(row["confidence"], 0)
        self.assertEqual(row["pending_confidence"], .99)

    def test_short_reply_topic_change_or_generic_refusal_remains_unknown(self):
        for text in ("知道了。", "好的。", "不用推荐这个软件", "哈哈今天阳光很好呀", "不是没事"):
            self.assertEqual(judge_repair_response(text), "unclear", text)

    def test_explicit_stop_does_not_create_a_permanent_sore_point(self):
        engine = TurnEngine(config(), self.store, ScriptedLLM([]))
        engine._turn_seq = 1
        engine._open_repair({})
        engine._turn_seq = 2
        engine._maybe_repair("刚才那句是不是有点重。", [])
        engine._judge_repair_response("别再道歉了。", [])
        self.assertEqual(self.store.relation()["sore_points"], [])
        self.assertEqual(self.rows("repair_paused")[0]["reason"], "explicit_stop")
        metrics = compute(self.store.read_ledger())
        self.assertEqual(metrics.repair_paused, 1)
        report = " ".join(detail for _, _, detail in verdicts(metrics))
        self.assertIn("不推断长期雷区", report)
        before = self.store.relation()
        rebuild_views(self.store)
        self.assertEqual(self.store.relation(), before)

    def test_legibility_levels_have_distinct_input_policies(self):
        cfg = config()
        cfg.role.legibility_max = 1
        mild = build_system_prompt(cfg)
        cfg.role.legibility_max = 2
        self.assertNotEqual(mild, build_system_prompt(cfg))

    def test_each_phrase_scene_has_a_reachable_trigger(self):
        cfg = config("shorekeeper")
        for scene, text in (("promise", "你会一直在吗"), ("answer", "为什么？"), ("farewell", "再见")):
            prompt = phrase_block(cfg, user_text=text, first_turn=False)
            self.assertIn(cfg.role.phrases[scene][0], prompt)

    def test_specials_are_injected_only_on_explicit_keyword_matches(self):
        cfg = config("shorekeeper")
        plain = build_system_prompt(cfg, user_text="hello")
        music = build_system_prompt(cfg, user_text="聊聊音乐")
        self.assertNotIn("可以多说几句，你弹琴", plain)
        self.assertIn("可以多说几句，你弹琴", music)
        self.assertNotIn("你对这两个词特别敏感", music)

    def test_invalid_configurations_fail_before_generation(self):
        for raw in ({"params": {"warmht": .6}}, {"role": {"bias": float("nan")}},
                    {"limits": {"pending_max": -1}}, {"role": {"legibility_max": 1.5}},
                    {"params": {"warmth": {"anchor": .3, "lo": .8, "hi": .2}}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                PersonaConfig.from_dict(raw)

    def test_judge_internal_type_error_is_not_a_retry(self):
        calls = []
        def fail(messages, *, max_tokens=None):
            calls.append(messages)
            raise TypeError("inside model adapter")
        handle = AsyncJudge(LLMJudge(fail), grace=1).submit(JudgeRequest(
            "hello", pendings=[{"id": "p_1", "claim": "c", "predicts": "p"}]))
        _, source = handle.collect()
        self.assertEqual(source, "error")
        self.assertEqual(len(calls), 1)

    def test_invalid_judge_json_is_not_reported_as_model_success(self):
        handle = AsyncJudge(LLMJudge(lambda messages: "garbage"), grace=1).submit(JudgeRequest(
            "hello", pendings=[{"id": "p_1", "claim": "c", "predicts": "p"}]))
        verdicts, source = handle.collect()
        self.assertEqual(source, "invalid")
        self.assertEqual(verdicts["p_1"].verdict, "unclear")

    def test_evidenceless_semantic_verdict_cannot_become_a_prediction_hit(self):
        cfg = config()
        p = self.store.add_pending("c", .7, "p", "t_0", cfg.limits)
        judge = AsyncJudge(LLMJudge(lambda messages: json.dumps([
            {"id": p["id"], "verdict": "returned"}])), grace=1)
        result = TurnEngine(cfg, self.store, ScriptedLLM([reply()]), judge=judge).turn("hello")
        self.assertEqual(result.settled, [])
        self.assertEqual(len(self.store.pendings()), 1)
        self.assertEqual(self.rows("turn")[0]["probe"]["judge"]["source"], "invalid")

    def test_slow_background_judge_available_at_collection_is_not_timeout(self):
        ledger = [{"kind": "turn", "probe": {"judge": {
            "source": "llm", "judge_elapsed": 1.2, "grace": .01}}} for _ in range(12)]
        report = {item.area: item for item in findings(collect(ledger))}
        self.assertEqual(report["grace"].level, "ok")


if __name__ == "__main__":
    unittest.main()
