"""Implementation boundaries for single-call delivery and asynchronous memory."""
from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from persona.config import PersonaConfig, StanceRule
from persona.consolidate import Consolidator
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply, openai_stream
from persona.store import Store
from persona.topicjudge import AsyncJudge, TopicVerdict


ROOT = Path(__file__).resolve().parent.parent


def config():
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


def reply(say="Hello.", confidence=0.6, tail="settle: -"):
    return make_reply(
        f"need: listen\nconfidence: {confidence}\ndirect: neutral\n"
        "baseline: Hello.\ndisplace: none", say, tail)


class StoreCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(tmp.name)

    def push(self, turn_id="t_1"):
        self.store.push_buffer({"turn_id": turn_id, "need": "listen",
                                "user": "hello", "say": "Hello."})


class TestDefaultDelivery(StoreCase):
    def engine(self, stream):
        self.llm = ScriptedLLM([])
        return TurnEngine(config(), self.store, self.llm, stream_llm=stream)

    def test_interruption_does_not_apply_tail_or_create_memory(self):
        raw = reply(tail="delta:\n  closeness: 0.05\npending:\n"
                    "  claim: user will elaborate\n  confidence: .7\n  predicts: details")

        def broken(messages):
            yield raw
            raise RuntimeError("disconnect")

        seen = []
        result = self.engine(broken).turn("hi", on_delta=seen.append)
        self.assertEqual(result.say, "".join(seen))
        self.assertIsNone(result.parsed.tail)
        self.assertIsNone(result.pending_added)
        self.assertEqual(result.relation_changed, {})
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])
        self.assertTrue(any("中断" in warning for warning in result.warnings))
        failures = [r for r in self.store.read_ledger() if r["kind"] == "generation_failure"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["delivered"], result.say)
        self.assertEqual(self.llm.calls, [])

    def test_interruption_never_flushes_undelivered_text(self):
        def broken(messages):
            yield reply().split("<<<TAIL>>>")[0].rstrip() + "\nd"
            raise RuntimeError("disconnect")

        seen = []
        result = self.engine(broken).turn("hi", on_delta=seen.append)
        self.assertEqual(result.say, "Hello.")
        self.assertEqual(result.say, "".join(seen))
        turn = [r for r in self.store.read_ledger() if r["kind"] == "turn"][-1]
        self.assertEqual(turn["say"], result.say)

    def test_normal_finish_delivers_final_remainder_once(self):
        def stream(messages):
            yield reply().split("<<<TAIL>>>")[0].rstrip() + "\nd"

        seen = []
        result = self.engine(stream).turn("hi", on_delta=seen.append)
        self.assertEqual(result.say, "Hello.\nd")
        self.assertEqual(result.say, "".join(seen))

    def test_incomplete_internal_marker_is_not_delivered(self):
        for suffix in ("<<<HE", "<<<TA", "\ndirect: secret",
                       "\nsettle: p_1 | confirmed | secret", "\npending: secret"):
            with self.subTest(suffix=suffix):
                seen = []
                result = self.engine(lambda messages: iter([
                    reply().split("<<<TAIL>>>")[0].rstrip() + suffix
                ])).turn("hi", on_delta=seen.append)
                self.assertEqual(result.say, "Hello.")
                self.assertEqual("".join(seen), "Hello.")

    def test_internal_residue_is_blocked_before_callback_for_any_chunk_size(self):
        for residue in ("<<<head>>hidden secret", "<< ^ HEAD >>hidden secret",
                        "（内心：hidden secret）", "( 判断: hidden secret)",
                        "<|endoftext|>"):
            raw = reply().split("<<<TAIL>>>")[0].rstrip() + residue
            for size in (1, 2, 5, len(raw)):
                with self.subTest(residue=residue, size=size):
                    seen = []
                    stream = lambda messages: (raw[i:i + size] for i in range(0, len(raw), size))
                    result = self.engine(stream).turn("hi", on_delta=seen.append)
                    self.assertEqual("".join(seen), "Hello.")
                    self.assertEqual(result.say, "Hello.")
                    self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])

    def test_callback_errors_are_not_disguised_as_model_success(self):
        def consumer(piece):
            raise ValueError("display failed")

        with self.assertRaisesRegex(ValueError, "display failed"):
            self.engine(lambda messages: iter([reply()])).turn("hi", on_delta=consumer)
        self.assertFalse(any(row["kind"] == "turn" for row in self.store.read_ledger()))

    def test_explicit_tool_interruption_does_not_start_observation_call(self):
        calls = []

        def tool(messages, **kwargs):
            calls.append(kwargs["tool"]["function"]["name"])
            return {"need": "listen", "confidence": .6, "direct": "neutral",
                    "baseline": "Hello.", "displace": "none"}

        def broken(messages):
            yield "Hello."
            raise RuntimeError("disconnect")

        result = TurnEngine(config(), self.store, ScriptedLLM([]), tool_llm=tool,
                            stream_llm=broken, protocol="tool").turn("hi", on_delta=lambda p: None)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(result.parsed.tail)
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])

    def test_explicit_tool_uses_the_same_incremental_safety_boundary(self):
        def tool(messages, **kwargs):
            return {"need": "listen", "confidence": .6, "direct": "neutral",
                    "baseline": "Hello.", "displace": "none"}

        seen = []
        result = TurnEngine(config(), self.store, ScriptedLLM([]), tool_llm=tool,
                            stream_llm=lambda messages: iter("Hello.<<<head>>secret"),
                            protocol="tool").turn("hi", on_delta=seen.append)
        self.assertEqual("".join(seen), "Hello.")
        self.assertEqual(result.say, "Hello.")
        self.assertIsNone(result.parsed.tail)

    def test_normal_parentheses_and_comparisons_are_not_swallowed(self):
        say = "Hello (world). 2 < 3. （你好）"
        raw = reply(say=say)
        seen = []
        result = self.engine(lambda messages: iter(raw)).turn("hi", on_delta=seen.append)
        self.assertEqual("".join(seen), say)
        self.assertEqual(result.say, say)

    def test_user_evidence_can_settle_old_pending_despite_interruption(self):
        cfg = config()
        pending = self.store.add_pending("user will elaborate", .7, "details", "t_0", cfg.limits)

        class Judge:
            def judge(self, req):
                return {pending["id"]: TopicVerdict("returned", "user gave details")}

        def broken(messages):
            yield reply()
            raise RuntimeError("disconnect")

        engine = TurnEngine(cfg, self.store, ScriptedLLM([]), stream_llm=broken,
                            judge=AsyncJudge(Judge(), grace=1))
        result = engine.turn("Here are the details of my plan.", on_delta=lambda p: None)
        self.assertEqual(len(result.settled), 1)
        self.assertEqual(result.settled[0]["outcome"], "CONFIRMED")

    def test_zero_confidence_uses_current_stance(self):
        cfg = config()
        llm = ScriptedLLM([reply(confidence=.9), reply(confidence=0)])
        engine = TurnEngine(cfg, self.store, llm)
        engine.turn("first")
        result = engine.turn("second")
        self.assertEqual(result.stance, cfg.stance_for(0))
        self.assertEqual(llm.calls[-1][-1]["content"], "second")
        self.assertIn("本轮站位", llm.calls[-1][0]["content"])
        self.assertNotIn("[当前站位:", llm.calls[-1][-1]["content"])

    def test_stance_policy_uses_config_bias_floor_and_gaps(self):
        from persona.prompt import stance_policy

        cfg = config()
        cfg.role.bias = .1
        cfg.confidence_floor = .3
        cfg.stance_table = [StanceRule(0, .2, "admit_unsure"),
                            StanceRule(.4, 1.01, "direct")]
        policy = stance_policy(cfg)
        self.assertIn("0.1 <= confidence < 0.2: admit_unsure", policy)
        self.assertIn("0.2 <= confidence < 0.3: admit_unsure", policy)
        self.assertIn("0.3 <= confidence < 1: direct", policy)
        self.assertIn("confidence = 1: direct", policy)
        self.assertIn("事实", policy)


class TestMemoryBuffer(StoreCase):
    def test_restart_recovers_staged_events_before_profiling(self):
        for i in range(3):
            self.push(f"t_{i}")
        self.store.drain_buffer()
        fresh = Store(self.store.root)
        result = Consolidator(fresh).run_once()
        self.assertEqual(result["activity"], "medium")
        self.assertEqual(result["judgments"], 1)
        self.assertEqual(list(fresh.processing_dir.glob("*.json")), [])
        self.assertEqual(list(fresh.buffer_dir.glob("*.json")), [])

    def test_failed_claim_is_not_returned_as_drained(self):
        self.push()
        with patch.object(Path, "rename", side_effect=OSError("busy")):
            self.assertEqual(self.store.drain_buffer(), [])
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))), 1)

    def test_buffer_publication_is_atomic(self):
        import os

        replace = os.replace
        published = []

        def inspect_publication(src, dst):
            self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])
            published.append(json.loads(Path(src).read_text(encoding="utf-8")))
            replace(src, dst)

        with patch("persona.store.os.replace", side_effect=inspect_publication):
            self.push()
        self.assertEqual([r["turn_id"] for r in published], ["t_1"])
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))), 1)
        self.assertEqual(list(self.store.buffer_dir.glob("*.tmp")), [])

    def test_concurrent_runs_share_a_consumer_lock_without_blocking_writes(self):
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        second_drained = threading.Event()
        written = threading.Event()
        errors = []

        def slow_model(prompt):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release memory model")
            return "[]"

        def run(consolidator, started=None):
            if started:
                started.set()
            try:
                consolidator.run_once()
            except Exception as exc:
                errors.append(exc)

        first = Consolidator(self.store, llm=slow_model)
        second = Consolidator(self.store, llm=lambda prompt: "[]")
        original_drain = self.store.drain_buffer

        def observe_drain(*args, **kwargs):
            if threading.current_thread().name == "second":
                second_drained.set()
            return original_drain(*args, **kwargs)

        def foreground():
            try:
                self.push("t_2")
                self.store.append("foreground_probe", {})
                written.set()
            except Exception as exc:
                errors.append(exc)

        self.push()
        a = threading.Thread(target=run, args=(first,), daemon=True)
        b = threading.Thread(target=run, args=(second, second_started), name="second", daemon=True)
        writer = threading.Thread(target=foreground, daemon=True)
        with patch.object(self.store, "drain_buffer", side_effect=observe_drain):
            try:
                a.start()
                self.assertTrue(entered.wait(3))
                b.start()
                self.assertTrue(second_started.wait(3))
                writer.start()
                self.assertTrue(written.wait(3), "memory model must not hold the general store lock")
                self.assertFalse(second_drained.wait(.1), "consolidation batches must not overlap")
            finally:
                release.set()
                for worker in (a, b, writer):
                    if worker.ident:
                        worker.join(3)
        self.assertEqual(errors, [])
        self.assertFalse(a.is_alive())
        self.assertFalse(b.is_alive())
        self.assertEqual(list(self.store.processing_dir.glob("*.json")), [])
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])

    def test_stop_timeout_retains_live_worker(self):
        entered = threading.Event()
        release = threading.Event()

        def slow_model(prompt):
            entered.set()
            release.wait(3)
            return "[]"

        self.push()
        consolidator = Consolidator(self.store, llm=slow_model)
        consolidator.start(.01)
        try:
            self.assertTrue(entered.wait(3))
            worker = consolidator._thread
            consolidator.stop(timeout=0)
            self.assertIs(consolidator._thread, worker)
            consolidator.start(.01)
            self.assertIs(consolidator._thread, worker)
        finally:
            release.set()
            consolidator.stop(timeout=3)
        self.assertIsNone(consolidator._thread)

    def test_judgment_return_matches_normalized_ledger_value(self):
        for confidence, expected in ((5, 1), (-1, 0), ("invalid", .5),
                                     (float("nan"), .5), (float("inf"), .5)):
            with self.subTest(confidence=confidence):
                result = self.store.add_judgment(str(confidence), confidence)
                self.assertEqual(result["confidence"], expected)
                self.assertEqual(result["confidence"], self.store.judgments()[-1]["confidence"])

    def test_leading_dot_confidence_is_not_promoted_to_certainty(self):
        from persona.parse import parse

        parsed = parse(reply(confidence=".7", tail="pending:\n  claim: c\n"
                             "  confidence: .6\n  predicts: p"))
        self.assertEqual(parsed.head.confidence, .7)
        self.assertEqual(parsed.tail.pending.confidence, .6)

    def test_model_judgments_respect_same_hard_cap_as_prompt(self):
        self.push()
        items = [{"judgment": text, "confidence": .7}
                 for text in ("prefers cats", "avoids crowds", "enjoys reading", "works nights")]
        model = lambda prompt: json.dumps(items) if "supersedes" in prompt else "story"
        result = Consolidator(self.store, llm=model, max_judgments_per_run=2).run_once()
        self.assertEqual(result["judgments"], 2)


class TestStreamTransport(unittest.TestCase):
    def stream(self, *events):
        lines = ["data: " + (e if isinstance(e, str) else json.dumps(e)) + "\n\n"
                 for e in events]
        return io.BytesIO("".join(lines).encode("utf-8"))

    def run_stream(self, response):
        with patch("urllib.request.urlopen", return_value=response):
            return list(openai_stream("fake", api_key="test")([]))

    def test_eof_without_completion_is_an_interruption(self):
        response = self.stream({"choices": [{"delta": {"content": "partial"}}]})
        with self.assertRaisesRegex(RuntimeError, "中断"):
            self.run_stream(response)

    def test_done_completes_normally(self):
        response = self.stream({"choices": [{"delta": {"content": "hello"}}]}, "[DONE]")
        self.assertEqual(self.run_stream(response), ["hello"])

    def test_explicit_stop_can_complete_without_done(self):
        response = self.stream({"choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]})
        self.assertEqual(self.run_stream(response), ["hello"])

    def test_token_truncation_is_not_a_completed_turn(self):
        response = self.stream({"choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]}, "[DONE]")
        with self.assertRaisesRegex(RuntimeError, "length"):
            self.run_stream(response)

    def test_stream_error_event_is_not_silently_skipped(self):
        response = self.stream({"error": {"message": "upstream failed"}}, "[DONE]")
        with self.assertRaisesRegex(RuntimeError, "upstream failed"):
            self.run_stream(response)

    def test_nonstream_truncation_does_not_commit_a_partial_tail(self):
        from persona.llm import openai_compatible

        payload = {"choices": [{"message": {"content": reply(tail="delta:\n  closeness: .05")},
                                "finish_reason": "length"}]}
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            engine = TurnEngine(config(), store, openai_compatible("fake", api_key="test"))
            with patch("persona.llm._post", return_value=payload) as post:
                with self.assertRaisesRegex(RuntimeError, "length"):
                    engine.turn("hi")
            self.assertEqual(post.call_count, 1)
            self.assertEqual([r for r in store.read_ledger() if r["kind"] == "turn"], [])
            self.assertTrue(any(r["kind"] == "generation_failure" for r in store.read_ledger()))
            self.assertEqual(list(store.buffer_dir.glob("*.json")), [])

    def test_tool_truncation_is_rejected_even_if_arguments_are_valid_json(self):
        from persona.llm import openai_tool_call

        payload = {"choices": [{"message": {"tool_calls": [
            {"function": {"arguments": '{"need": "listen"}'}}]}, "finish_reason": "length"}]}
        with patch("persona.llm._post", return_value=payload):
            with self.assertRaisesRegex(RuntimeError, "length"):
                openai_tool_call("fake", api_key="test")(
                    [], tool={"function": {"name": "inner"}})
