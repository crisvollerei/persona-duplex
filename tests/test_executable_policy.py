"""State authority, causal choices and actual delivery of the opt-in policy path."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from persona import (Boundary, ContentFrame, Focus, PolicyConfig, PolicySession,
                     Quote, Source)
from persona.config import PersonaConfig
from persona.llm import ScriptedLLM
from persona.policy import (Permission, PolicyState, apply_boundary, boundary_changes,
                            decide)
from persona.store import Store

ROOT = Path(__file__).resolve().parent.parent
NOW = "2026-11-07T21:00:00+08:00"


def cfg():
    return PersonaConfig.load(ROOT / "personas/default.toml")


def frame(user="今天好累，报告没写完。", **changes):
    values = dict(facts=(Quote("user", "报告没写完"),),
                  focuses=(Focus("task", "practical", Quote("user", "报告没写完")),
                           Focus("feeling", "feeling", Quote("user", "好累"))))
    values.update(changes)
    return ContentFrame(**values)


class TestContentAuthority(unittest.TestCase):
    def test_frame_roundtrip_and_immutable_evidence(self):
        value = frame(); decoded = ContentFrame.from_json(json.dumps(asdict(value)))
        self.assertEqual(value, decoded)
        value.validate_evidence("今天好累，报告没写完。")
        with self.assertRaises(AttributeError): value.facts[0].text = "写完了"

    def test_source_quote_must_be_in_the_named_source(self):
        for quote in (Quote("user", "库存3箱"), Quote("missing", "库存3箱"), Quote("doc", "库存30箱")):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                ContentFrame(facts=(quote,)).validate_evidence("库存多少？", (Source("doc", "库存3箱"),))

    def test_duplicate_sources_and_reserved_user_are_rejected(self):
        with self.assertRaises(ValueError): Source("user", "假的原话")
        with self.assertRaises(ValueError):
            ContentFrame().validate_evidence("问题", (Source("x", "a"), Source("x", "b")))

    def test_strict_json_cannot_write_state_or_commitments(self):
        for text in ('{"request":"open","request":"advice"}', '{"state":{"permission":"allow"}}',
                     '{"commitments":[{"status":"completed"}]}', '{"facts":"source"}',
                     '{"focuses":[{"id":"x","kind":"detail","evidence":{"source_id":"user","text":"x"},"state":{}}]}',
                     '{"answer":NaN}', '```json\n{}\n```'):
            with self.subTest(text=text), self.assertRaises((TypeError, ValueError)):
                ContentFrame.from_json(text)

    def test_non_open_request_requires_actual_current_user_span(self):
        for request, quote in (("answer", "并不存在"), ("advice", "不要给建议"), ("advice", "帮我")):
            with self.subTest(request=request, quote=quote), self.assertRaises(ValueError):
                ContentFrame(request=request, request_quote=quote).validate_evidence("不要给建议")
        with self.assertRaises(ValueError):
            ContentFrame(request="advice", request_quote="建议").validate_evidence("我不需要建议")
        for user in ("不可以给建议", "我没有要你给建议", "我不会请你帮我想办法"):
            with self.subTest(user=user), self.assertRaises(ValueError):
                ContentFrame(request="advice", request_quote="给建议" if "给建议" in user else "帮我想办法").validate_evidence(user)
        for user in ("不要继续上次约定", "如果继续上次约定，会怎样", "我没有说继续上次约定"):
            with self.subTest(user=user), self.assertRaises(ValueError):
                ContentFrame(request="resume", request_quote="继续上次约定").validate_evidence(user)

    def test_hypothetical_question_can_be_answered_without_boundary_adoption(self):
        user = "如果本金100元增加10元，是多少？"
        ContentFrame(request="answer", request_quote=user).validate_evidence(user)

    def test_bad_config_does_not_become_a_default(self):
        for value in (float('nan'), float('inf'), True, -.1, 1.1, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError): PolicyConfig(intervention=value)


class TestCausalPolicy(unittest.TestCase):
    def test_parameter_changes_action_before_rendering_without_changing_content(self):
        content = frame(); before = asdict(content)
        low = decide(content, PolicyState(), PolicyConfig(intervention=0), topic="work", key="1")
        high = decide(content, PolicyState(), PolicyConfig(intervention=1), topic="work", key="1")
        self.assertEqual((low.action, high.action), ("respond", "advise"))
        self.assertEqual(low.draw, high.draw)
        self.assertEqual(asdict(content), before)

    def test_attention_changes_focus_not_action_or_evidence(self):
        a = decide(frame(), PolicyState(), PolicyConfig(intervention=0, practical=1, feeling=0), topic="w", key="k")
        b = decide(frame(), PolicyState(), PolicyConfig(intervention=0, practical=0, feeling=1), topic="w", key="k")
        self.assertEqual((a.focus_id, b.focus_id), ("task", "feeling"))
        self.assertEqual(a.action, b.action)

    def test_boundary_masks_all_parameter_draws_but_not_other_topics(self):
        state = apply_boundary(PolicyState(), dict(scope="work", advice="deny"))
        for i in range(32):
            self.assertEqual(decide(frame(), state, PolicyConfig(intervention=1), topic="work", key=str(i)).action, "respond")
        self.assertEqual(decide(frame(), state, PolicyConfig(intervention=1), topic="food", key="1").action, "advise")

    def test_explicit_current_request_does_not_permanently_revoke_a_boundary(self):
        state = apply_boundary(PolicyState(), dict(scope="work", advice="deny"))
        result = decide(ContentFrame(request="advice", request_quote="现在帮我想办法"), state,
                        PolicyConfig(intervention=0), topic="work", key="1")
        self.assertEqual(result.action, "advise")
        self.assertEqual(state.advice_permission("work"), "deny")

    def test_newest_applicable_control_and_expiry_restore_earlier_boundary(self):
        state = apply_boundary(PolicyState(), dict(scope="work", advice="deny"))
        state = apply_boundary(state, dict(scope="*", advice="allow", expires_at="2026-11-08T00:00:00+08:00"))
        self.assertEqual(state.advice_permission("work", at=NOW), "allow")
        self.assertEqual(state.advice_permission("work", at="2026-11-08T00:00:00+08:00"), "deny")

    def test_boundary_scope_is_derived_from_actual_command_not_model_label(self):
        f = ContentFrame(boundaries=(Boundary("deny", "不要给建议", "session"),))
        accepted, rejected = boundary_changes(f, "不要给建议", "work", at=NOW)
        self.assertEqual(accepted[0]["scope"], "work"); self.assertEqual(rejected, [])
        g = ContentFrame(boundaries=(Boundary("deny", "以后所有话题都不要给建议", "topic"),))
        accepted, _ = boundary_changes(g, "以后所有话题都不要给建议", "work", at=NOW)
        self.assertEqual(accepted[0]["scope"], "*")
        self.assertEqual(accepted[0]["expires_at"], None)

    def test_reported_hypothetical_negated_and_future_proposals_stay_unaccepted(self):
        cases = [("朋友说不要给建议", "不要给建议", "deny"),
                 ("如果我说不要给建议呢", "不要给建议", "deny"),
                 ('我引用“不要给建议”，只是举例', "不要给建议", "deny"),
                 ("不是可以给建议", "不是可以给建议", "allow"),
                 ("不可以给建议", "可以给建议", "allow"),
                 ("我没有说不要给建议", "不要给建议", "deny"),
                 ("我不是不要给建议", "不要给建议", "deny"),
                 ("下周不要给建议", "下周不要给建议", "deny")]
        for user, quote, mode in cases:
            with self.subTest(user=user):
                accepted, rejected = boundary_changes(ContentFrame(boundaries=(Boundary(mode, quote),)), user, "work", at=NOW)
                self.assertEqual(accepted, []); self.assertEqual(len(rejected), 1)

    def test_noncanonical_time_and_scope_are_not_invented(self):
        for user in ("明年不要给建议", "下个月不要给建议", "昨天不要给建议", "这学期不要给建议",
                     "从今晚十点开始不要给建议", "今晚这轮不要给建议", "以后在工作上不要给建议",
                     "以后我会去看电影，今晚不要给建议"):
            with self.subTest(user=user):
                for scope in ("topic", "session"):
                    accepted, _ = boundary_changes(ContentFrame(boundaries=(Boundary("deny", "不要给建议", scope),)), user, "work", at=NOW)
                    self.assertEqual(accepted, [])

    def test_short_model_quote_cannot_remove_a_user_supplied_expiry(self):
        f = ContentFrame(boundaries=(Boundary("deny", "不要给建议"),))
        accepted, _ = boundary_changes(f, "今晚不要给建议", "work", at=NOW)
        self.assertEqual(accepted[0]["expires_at"], "2026-11-08T00:00:00+08:00")
        accepted, _ = boundary_changes(f, "今晚不要给建议，以后再说", "work", at=NOW)
        self.assertEqual(accepted, [])
        accepted, _ = boundary_changes(ContentFrame(boundaries=(Boundary("deny", "今晚不要给建议", "turn"),)), "今晚不要给建议", "work", at=NOW)
        self.assertFalse(accepted[0]["transient"])
        self.assertEqual(accepted[0]["expires_at"], "2026-11-08T00:00:00+08:00")


class TestPolicySession(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.store = Store(temporary.name)

    def session(self, replies, **kwargs):
        self.llm = ScriptedLLM(replies)
        return PolicySession(cfg(), self.store, self.llm, clock=lambda: NOW, realization="free", **kwargs)

    def rows(self, kind):
        return [r for r in self.store.read_ledger() if r["kind"] == kind]

    def test_supplied_content_needs_one_call_and_decision_precedes_it(self):
        observed = []
        def render(messages):
            observed.append(self.rows("policy_input")[-1]["decision"]["action"])
            return "先把报告的一小段写下来？"
        session = PolicySession(cfg(), self.store, render, policy=PolicyConfig(intervention=1))
        result = session.say("今天好累，报告没写完。", frame=frame())
        self.assertEqual(observed, ["advise"]); self.assertEqual(result.calls_count, 1)

    def test_auto_content_is_persona_free_and_never_receives_persona_say(self):
        config = cfg(); config.role.identity = "IDENTITY_SECRET"; config.role.voice = "VOICE_SECRET"
        self.llm = ScriptedLLM([json.dumps(asdict(frame())), "RENDER_SECRET", json.dumps(asdict(frame())), "第二次"])
        session = PolicySession(config, self.store, self.llm, realization="free")
        session.say("今天好累，报告没写完。")
        result = session.say("今天好累，报告没写完。")
        for idx in (0, 2):
            request = json.dumps(self.llm.calls[idx], ensure_ascii=False)
            for secret in ("IDENTITY_SECRET", "VOICE_SECRET", "RENDER_SECRET"):
                self.assertNotIn(secret, request)
        self.assertEqual(result.calls_count, 2)

    def test_explicit_reply_reference_is_available_as_quoted_source_not_state_authority(self):
        first = frame()
        second = ContentFrame(request="answer", request_quote="你刚才第二条具体怎么做？",
                              facts=(Quote("prior_reply_1", "第二条：改标题"),), answer="解释标题")
        session = self.session(["第一条：保存。第二条：改标题。", json.dumps(asdict(second)), "解释"])
        session.say("今天好累，报告没写完。", frame=first)
        session.say("你刚才第二条具体怎么做？")
        request = json.loads(self.llm.calls[1][-1]["content"])
        self.assertEqual(request["sources"][0]["id"], "prior_reply_1")
        self.assertIn("第二条：改标题", request["sources"][0]["text"])
        self.assertEqual(session.state.revision, 0)

    def test_application_request_confirmation_is_separate_from_generated_json(self):
        session = self.session(["帮你理清"])
        user = "我现在需要一些具体的应对思路"
        supplied = ContentFrame(request="advice", request_quote=user)
        with self.assertRaises(ValueError): session.say(user, frame=supplied)
        result = session.say(user, frame=supplied, confirmed_request=True)
        self.assertEqual(result.decision.action, "advise")
        with self.assertRaises(ValueError): session.say(user, confirmed_request=True)

    def test_renderer_receives_selected_action_without_personality_weights(self):
        session = self.session(["收到"], policy=PolicyConfig(intervention=0, practical=1, feeling=0))
        session.say("今天好累，报告没写完。", frame=frame())
        request = json.dumps(self.llm.calls[0], ensure_ascii=False)
        self.assertNotIn('"intervention"', request)
        self.assertNotIn("HEAD", request); self.assertNotIn("baseline", request)
        self.assertEqual(json.loads(self.llm.calls[0][-1]["content"])["decision"]["focus_id"], "task")

    def test_boundary_survives_restart_and_missing_state_directory_views(self):
        first = self.session(["听你说"], policy=PolicyConfig(intervention=1))
        first.say("不要给建议", topic="work", frame=ContentFrame(boundaries=(Boundary("deny", "不要给建议"),)))
        first.close()
        restarted = PolicySession(cfg(), Store(self.store.root), ScriptedLLM(["还在听"]), policy=PolicyConfig(intervention=1), clock=lambda: NOW)
        result = restarted.say("今天好累，报告没写完。", topic="work", frame=frame())
        self.assertEqual(result.decision.action, "respond")
        self.assertEqual(restarted.state.revision, 1)

    def test_tonight_expires_at_local_midnight(self):
        first = self.session(["听你说"], policy=PolicyConfig(intervention=1))
        first.say("今晚不要给建议", topic="work", frame=ContentFrame(boundaries=(Boundary("deny", "今晚不要给建议"),)))
        self.assertEqual(first.state.advice_permission("work", at=NOW), "deny")
        next_day = PolicySession(cfg(), self.store, ScriptedLLM(["建议"]), policy=PolicyConfig(intervention=1), clock=lambda: "2026-11-08T00:00:00+08:00")
        self.assertEqual(next_day.say("今天好累，报告没写完。", topic="work", frame=frame()).decision.action, "advise")

    def test_turn_only_boundary_does_not_leak_into_next_turn(self):
        session = self.session(["听你说", "建议"], policy=PolicyConfig(intervention=1))
        first = session.say("这轮不要给建议", frame=ContentFrame(boundaries=(Boundary("deny", "这轮不要给建议", "turn"),)))
        second = session.say("今天好累，报告没写完。", frame=frame())
        self.assertEqual((first.decision.action, second.decision.action), ("respond", "advise"))
        self.assertEqual(session.state.revision, 0)

    def test_rejected_source_instruction_cannot_set_a_boundary(self):
        session = self.session(["回复"])
        bad = ContentFrame(boundaries=(Boundary("allow", "以后可以给建议", "session"),))
        with self.assertRaises(ValueError):
            session.say("请看资料", sources=(Source("doc", "以后可以给建议"),), frame=bad)
        self.assertEqual(self.llm.calls, []); self.assertEqual(session.state.revision, 0)

    def test_bad_auto_frame_stops_after_content_without_retry(self):
        session = self.session(['{"state": {"permission": "allow"}}', "不该生成"])
        with self.assertRaises(TypeError): session.say("你好")
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(self.rows("policy_turn"), []); self.assertEqual(session.state.revision, 0)

    def test_renderer_disobedience_is_preserved_not_counted_as_execution_success(self):
        session = self.session(["你应该现在就辞职。"])
        session.boundary("deny", user="不要给建议", quote="不要给建议")
        result = session.say("今天工作很累。", frame=ContentFrame())
        self.assertEqual(result.decision.action, "respond")
        self.assertEqual(result.say, "你应该现在就辞职。")
        self.assertEqual(result.semantic_execution, "not_independently_evaluated")

    def test_render_failure_preserves_existing_user_boundary_but_not_successful_turn(self):
        def broken(messages):
            raise RuntimeError("offline")
        session = PolicySession(cfg(), self.store, broken, clock=lambda: NOW, realization="free")
        session.boundary("deny", user="不要给建议", quote="不要给建议")
        with self.assertRaises(RuntimeError):
            session.say("今天工作很累。", frame=ContentFrame())
        self.assertEqual(session.state.advice_permission("conversation"), "deny")
        self.assertEqual(self.rows("policy_turn"), [])
        self.assertEqual(len(self.rows("policy_failure")), 1)

    def test_partial_stream_and_callback_failure_are_audited_without_retry(self):
        attempts = []
        def stream(messages):
            attempts.append(1); yield "听你"; yield "说。"; raise RuntimeError("disconnect")
        session = PolicySession(cfg(), self.store, ScriptedLLM([]), stream_llm=stream, realization="free")
        seen = []
        with self.assertRaises(RuntimeError): session.say("你好", frame=ContentFrame(), on_delta=seen.append)
        self.assertEqual(seen, ["听你", "说。"]); self.assertEqual(attempts, [1])
        self.assertEqual(self.rows("policy_failure")[-1]["delivered"], "听你说。")
        self.assertEqual(self.rows("policy_turn"), [])

    def test_failed_but_delivered_fragment_can_be_referenced_as_incomplete(self):
        def stream(messages):
            yield "第一条：保存。第二条：改标题。"
            raise RuntimeError("disconnect")
        first = PolicySession(cfg(), self.store, ScriptedLLM([]), stream_llm=stream, realization="free")
        with self.assertRaises(RuntimeError): first.say("你好", frame=ContentFrame())
        second_frame = ContentFrame(request="answer", request_quote="你刚才第二条怎么做？")
        second = self.session([json.dumps(asdict(second_frame)), "说明标题"])
        second.say("你刚才第二条怎么做？")
        source = json.loads(self.llm.calls[0][-1]["content"])["sources"][0]
        self.assertIn("后文未知", source["text"])
        self.assertIn("第二条：改标题", source["text"])

    def test_callback_reentrancy_is_rejected_and_no_successful_turn_is_saved(self):
        session = self.session(["回复"])
        with self.assertRaises(RuntimeError):
            session.say("你好", frame=ContentFrame(), on_delta=lambda text: session.say("重入", frame=ContentFrame()))
        self.assertEqual(self.rows("policy_turn"), [])
        self.assertEqual(self.rows("policy_failure")[-1]["attempted"], "回复")

    def test_contract_checks_actual_reply_without_rewriting(self):
        from persona import ReplyContract
        session = self.session(["还剩300元。"])
        result = session.say("余额？", frame=ContentFrame(request="answer", request_quote="余额？", answer="300元"),
                             contract=ReplyContract.conclusion(unit="元", answer_type="number"))
        self.assertFalse(result.contract_validation["valid"])
        self.assertEqual(result.say, "还剩300元。")

    def test_application_commitment_can_resume_but_say_cannot_complete_it(self):
        session = self.session(["已经完成！", "接着做"])
        session.commitment("confirm", id="resume", user="好，先改简历标题", quote="先改简历标题", topic="work", text="先改简历标题")
        result = session.say("继续上次约定", topic="work", frame=ContentFrame(request="resume", request_quote="继续上次约定"))
        self.assertEqual((result.decision.action, result.decision.commitment_id), ("continue", "resume"))
        self.assertEqual(session.state.commitments[0].status, "open")
        session.close()
        restarted = PolicySession(cfg(), Store(self.store.root), ScriptedLLM(["回复"]))
        self.assertEqual(restarted.state.commitments[0].status, "open")
        restarted.commitment("complete", id="resume", user="已经改好了", quote="已经改好了", topic="work")
        self.assertEqual(restarted.state.commitments[0].status, "completed")

    def test_commitment_event_rejects_quotes_duplicate_ids_wrong_topics_and_closed_session(self):
        session = self.session([])
        with self.assertRaises(ValueError):
            session.commitment("confirm", id="x", user='朋友说“做简历”', quote="做简历", text="做简历")
        session.commitment("confirm", id="x", user="做简历", quote="做简历", topic="work", text="做简历")
        with self.assertRaises(ValueError):
            session.commitment("confirm", id="x", user="做简历", quote="做简历", topic="work", text="做简历")
        with self.assertRaises(ValueError):
            session.commitment("complete", id="x", user="完成了", quote="完成了", topic="food")
        session.close()
        with self.assertRaises(RuntimeError): session.say("你好", frame=ContentFrame())

    def test_multiple_resume_targets_are_clarified_or_explicitly_selected(self):
        session = self.session(["继续哪项？", "接着做标题", "继续哪项？"])
        for id, text in (("outline", "改提纲"), ("title", "改标题")):
            session.commitment("confirm", id=id, text=text, user=text, quote=text, topic="work")
        first = session.say("继续上次的", topic="work", frame=ContentFrame(request="resume", request_quote="继续上次的"))
        second = session.say("继续改标题", topic="work", frame=ContentFrame(request="resume", request_quote="继续改标题", resume_id="title"))
        self.assertEqual((first.decision.action, first.decision.commitment_id), ("clarify", None))
        self.assertEqual((second.decision.action, second.decision.commitment_id), ("continue", "title"))
        third = session.say("继续上次的", topic="work", frame=ContentFrame(request="resume", request_quote="继续上次的", resume_id="title"))
        self.assertEqual(third.decision.action, "clarify")

    def test_namespaces_do_not_inherit_another_policy_state(self):
        session = self.session(["听你说"])
        session.say("不要给建议", frame=ContentFrame(boundaries=(Boundary("deny", "不要给建议"),)))
        other = PolicySession(cfg(), self.store, ScriptedLLM([]), namespace="another")
        self.assertEqual(other.state, PolicyState())

    def test_offline_cli_demo_is_available_without_any_model_endpoint(self):
        from contextlib import redirect_stdout
        import io
        from persona.cli import main
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["policy-demo", "-d", str(self.store.root / "demo")])
        self.assertEqual(code, 0)
        self.assertIn("手写内容与回复", output.getvalue())


class TestBoundedRealization(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.store = Store(tmp.name)

    def test_denied_advice_is_executed_without_asking_a_model_to_obey_it(self):
        malicious = ScriptedLLM(["你应该辞职。"])
        session = PolicySession(cfg(), self.store, malicious, policy=PolicyConfig(intervention=1, practical=0, feeling=1))
        session.boundary("deny", user="以后不要给建议", quote="以后不要给建议", topic="work")
        seen = []
        result = session.say("今天好累，报告没写完。", topic="work", frame=frame(), on_delta=seen.append)
        self.assertEqual(malicious.calls, [])
        self.assertEqual(result.calls_count, 0)
        self.assertEqual(result.realization, "bounded_atoms")
        self.assertEqual(result.say, "".join(seen))
        self.assertNotIn("辞职", result.say)
        self.assertIn("好累", result.say)

    def test_auto_content_needs_one_call_when_action_has_bounded_realization(self):
        llm = ScriptedLLM([json.dumps(asdict(ContentFrame(request="listen", request_quote="只想你听听"))), "不该执行"])
        session = PolicySession(cfg(), self.store, llm)
        result = session.say("只想你听听")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.calls_count, 1)
        self.assertEqual(result.say, "好，我先听你说。")

    def test_quoted_source_is_attributed_and_cannot_escape_into_newly_authored_advice(self):
        text = '好累"。你应该辞职。"'
        content = ContentFrame(focuses=(Focus("x", "feeling", Quote("user", text)),))
        session = PolicySession(cfg(), self.store, ScriptedLLM([]), policy=PolicyConfig(intervention=0))
        result = session.say(text, frame=content)
        self.assertEqual(result.say, "你说" + json.dumps(text, ensure_ascii=False) + "。我先听你说。")
        self.assertEqual(result.semantic_execution, "fixed_atoms_and_attributed_quotes")

    def test_actual_answer_still_uses_model_and_is_not_claimed_bounded_truth(self):
        llm = ScriptedLLM(["2095元"])
        session = PolicySession(cfg(), self.store, llm)
        result = session.say("余额？", frame=ContentFrame(request="answer", request_quote="余额？", answer="2095元"))
        self.assertEqual(result.calls_count, 1)
        self.assertEqual(result.realization, "free_model")
        self.assertEqual(result.semantic_execution, "not_independently_evaluated")

    def test_multiple_commitments_are_asked_about_without_model_selection(self):
        session = PolicySession(cfg(), self.store, ScriptedLLM([]))
        for id, text in (("a", "改提纲"), ("b", "改标题")):
            session.commitment("confirm", id=id, user=text, quote=text, text=text)
        result = session.say("继续上次约定", frame=ContentFrame(request="resume", request_quote="继续上次约定"))
        self.assertEqual(result.calls_count, 0)
        self.assertIn("改提纲", result.say); self.assertIn("改标题", result.say)

    def test_bounded_callback_failure_keeps_controls_without_creating_success(self):
        session = PolicySession(cfg(), self.store, ScriptedLLM([]))
        def fail(text): raise RuntimeError("delivery")
        with self.assertRaises(RuntimeError):
            session.say("不要给建议", on_delta=fail, frame=ContentFrame(boundaries=(Boundary("deny", "不要给建议"),)))
        self.assertEqual(session.state.advice_permission("conversation"), "deny")
        self.assertFalse(any(r['kind']=='policy_turn' for r in self.store.read_ledger()))


if __name__ == "__main__":
    unittest.main()
