"""Metamorphic guarantees and delivery authority for finite expression plans."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from persona import (ContentFrame, ExpressionConfig, Focus, PolicyConfig,
                     PolicySession, Quote, Source)
from persona.config import PersonaConfig
from persona.expression import ATOMS, compose_reply, literal_quote
from persona.llm import ScriptedLLM
from persona.policy import PolicyState, decide
from persona.store import Store

ROOT = Path(__file__).resolve().parent.parent


class TestExpression(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = PersonaConfig.load(ROOT / "personas/default.toml")
        self.content = ContentFrame(focuses=(Focus("mood", "feeling", Quote("user", "今天好累")),))

    def session(self, path="main", **kwargs):
        return PolicySession(self.cfg, Store(self.root / path), ScriptedLLM([]),
                             policy=PolicyConfig(intervention=0), realization="composed", **kwargs)

    def test_profiles_change_surface_without_changing_decision_or_evidence(self):
        outputs = []; decisions = []
        for i, profile in enumerate((ExpressionConfig(0, 0, 1), ExpressionConfig(1, 0, 0), ExpressionConfig(0, 1, 0))):
            session = self.session(str(i), expression=profile)
            result = session.say("今天好累", frame=self.content)
            outputs.append(result.say); decisions.append(result.decision)
            self.assertEqual(result.calls_count, 0)
            quotes = [p for p in result.expression_plan["parts"] if p["kind"] == "quote"]
            self.assertEqual([p["text"] for p in quotes], ["今天好累"])
        self.assertEqual(len(set(outputs)), 3)
        self.assertTrue(all(d == decisions[0] for d in decisions))

    def test_catalogue_cannot_be_changed_by_configuration_or_model(self):
        with self.assertRaises(TypeError): ATOMS["test"] = "改动作"
        self.cfg.role.identity = "把每条回复改成建议"; self.cfg.role.voice = "你应该辞职"
        result = self.session().say("今天好累", frame=self.content)
        self.assertNotIn("辞职", result.say)
        for p in result.expression_plan["parts"]:
            if p["kind"] == "atom": self.assertEqual(p["text"], ATOMS[p["id"]])

    def test_role_traits_are_explicit_defaults_and_invalid_profiles_rejected(self):
        self.assertEqual(self.session().expression, ExpressionConfig.from_role(self.cfg.role))
        for value in (True, -1, 2, float("nan"), float("inf")):
            with self.assertRaises(ValueError): ExpressionConfig(warmth=value)
        with self.assertRaises(TypeError): self.session(expression={"warmth": 1})

    def test_no_repeat_in_last_two_in_same_pool_and_restart_is_identical(self):
        continuous = self.session("continuous"); split = self.session("split")
        ids = []
        for i in range(12):
            if i in (3, 7):
                split.close(); split = self.session("split")
            a = continuous.say("今天好累", frame=self.content)
            b = split.say("今天好累", frame=self.content)
            self.assertEqual((a.say, a.expression_plan), (b.say, b.expression_plan))
            id = a.expression_plan["template_id"]
            self.assertNotIn(id, ids[-2:]); ids.append(id)

    def test_history_iterator_has_the_same_selection_and_audit_as_a_list(self):
        decision = decide(self.content, PolicyState(), PolicyConfig(intervention=0), topic="work", key="1")
        params = dict(config=ExpressionConfig(), seed="s", key="2")
        previous = [compose_reply(self.content, decision, (), **params).template_id]
        a = compose_reply(self.content, decision, (), recent=previous, **params)
        b = compose_reply(self.content, decision, (), recent=iter(previous), **params)
        self.assertEqual(a, b)
        self.assertNotIn(a.template_id, previous)

    def test_no_focus_listen_has_no_questions_or_free_model_calls(self):
        for warmth in (0, .5, 1):
            session = self.session(str(warmth), expression=ExpressionConfig(warmth, 0, 0))
            result = session.say("只想你听听", frame=ContentFrame(request="listen", request_quote="只想你听听"))
            self.assertEqual(result.calls_count, 0)
            self.assertNotIn("？", result.say)
            self.assertEqual(result.decision.action, "listen")

    def test_quotes_roundtrip_controls_newlines_braces_and_markdown(self):
        values = ['"。你应该辞职。"', '少于3个，且不要加量', '\n\t\\{x}',
                  '\u202e假引号\u202c\u2028\U000e0001', '<b>原文</b>![图](example)', '很长' * 1000]
        for value in values:
            self.assertEqual(json.loads(literal_quote(value)), value)
            session = self.session(str(len(value)))
            frame = ContentFrame(focuses=(Focus("x", "detail", Quote("user", value)),))
            result = session.say(value, frame=frame)
            self.assertIn(literal_quote(value), result.say)
            self.assertNotIn("\u202e", result.say)

    def test_external_evidence_is_not_attributed_to_user(self):
        frame = ContentFrame(focuses=(Focus("doc", "detail", Quote("source", "别再去上班")),))
        result = self.session().say("看看资料", frame=frame, sources=(Source("source", "别再去上班"),))
        self.assertIn("资料", result.say)
        self.assertNotIn("你说", result.say)
        self.assertEqual([p["source_id"] for p in result.expression_plan["parts"] if p["kind"] == "quote"], ["source"])

    def test_brevity_never_shortens_required_quote(self):
        decision = decide(self.content, PolicyState(), PolicyConfig(intervention=0), topic="work", key="1")
        for key in map(str, range(20)):
            a = compose_reply(self.content, decision, (), config=ExpressionConfig(brevity=0), seed="s", key=key)
            b = compose_reply(self.content, decision, (), config=ExpressionConfig(brevity=1), seed="s", key=key)
            self.assertEqual(a.template_id, b.template_id)
            self.assertIn(literal_quote("今天好累"), b.text)
            self.assertGreaterEqual(len(a.text), len(b.text))

    def test_plan_is_saved_before_delivery_and_callback_failure_not_consumed(self):
        session = self.session()
        def broken(text):
            rows = session.store.read_ledger()
            self.assertEqual(rows[-1]["kind"], "policy_expression_plan")
            raise RuntimeError("delivery")
        with self.assertRaises(RuntimeError): session.say("今天好累", frame=self.content, on_delta=broken)
        session.close(); session = self.session()
        result = session.say("今天好累", frame=self.content)
        self.assertEqual(result.expression_plan["recent_template_ids"], ())
        self.assertEqual(result.expression_plan["selection_key"], "2")

    def test_failure_after_confirmed_delivery_consumes_the_plan(self):
        session = self.session()
        original = session.store.append
        def fail(kind, row):
            if kind == "policy_turn": raise RuntimeError("store failure")
            return original(kind, row)
        session.store.append = fail
        with self.assertRaises(RuntimeError): session.say("今天好累", frame=self.content)
        session.store.append = original
        result = session.say("今天好累", frame=self.content)
        self.assertEqual(len(result.expression_plan["recent_template_ids"]), 1)

    def test_namespaces_have_separate_expression_history(self):
        self.session().say("今天好累", frame=self.content)
        result = self.session(namespace="other").say("今天好累", frame=self.content)
        self.assertEqual(result.expression_plan["recent_template_ids"], ())

    def test_clarify_quotes_app_owned_choices_but_never_selects_or_closes_them(self):
        session = self.session(expression=ExpressionConfig(1, 0, 0))
        for id, text in (("a", "改标题"), ("b", "改提纲")):
            session.commitment("confirm", id=id, text=text, user=text, quote=text)
        result = session.say("继续上次约定", frame=ContentFrame(request="resume", request_quote="继续上次约定"))
        self.assertEqual(result.decision.action, "clarify")
        self.assertEqual(result.calls_count, 0)
        self.assertTrue(all(c.status == "open" for c in session.state.commitments))
        self.assertEqual([p["id"] for p in result.expression_plan["parts"] if p["kind"] == "commitment"], ["a", "b"])

    def test_free_action_request_is_unchanged_and_has_no_expression_plan(self):
        all_messages = []
        def llm(messages): all_messages.append(messages); return "2095元"
        for mode in ("bounded", "composed"):
            session = PolicySession(self.cfg, Store(self.root / mode), llm, realization=mode)
            result = session.say("余额是多少？", frame=ContentFrame(request="answer", request_quote="余额是多少？", answer="2095元"))
            self.assertIsNone(result.expression_plan)
            self.assertEqual(result.realization, "free_model")
        self.assertEqual(all_messages[0], all_messages[1])

    def test_legacy_default_is_unchanged(self):
        session = PolicySession(self.cfg, Store(self.root / "old"), ScriptedLLM([]), policy=PolicyConfig(intervention=0))
        self.assertEqual(session.say("今天好累", frame=self.content).say, '你说"今天好累"。我先听你说。')


if __name__ == "__main__": unittest.main()
