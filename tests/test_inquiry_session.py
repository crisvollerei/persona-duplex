"""Question authority, actual delivery, recovery and shared proposal validation."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from persona import (ContentFrame, InquiryConfig, PolicyConfig, PolicySession,
                     QuestionChoice, QuestionSlot, Quote, Source)
from persona.config import PersonaConfig
from persona.llm import ScriptedLLM
from persona.store import Store

ROOT = Path(__file__).resolve().parent.parent


class TestInquirySession(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.cfg = PersonaConfig.load(ROOT / "personas/default.toml")
        self.sources = (Source("task", "报告没写完，截止时间和格式还没确认"),)
        self.slots = (
            QuestionSlot("due", "work", "报告最晚什么时候交？", "practical", Quote("task", "截止时间"), "确认交付期限"),
            QuestionSlot("format", "work", "报告需要什么文件格式？", "detail", Quote("task", "格式"), "确定导出格式"),
        )

    def session(self, **kwargs):
        self.llm = ScriptedLLM([])
        return PolicySession(self.cfg, Store(self.root), self.llm,
            policy=PolicyConfig(intervention=0, practical=1, detail=0),
            inquiry=kwargs.pop("inquiry", InquiryConfig(initiative=1)), clock=lambda:"2026-11-07T21:00:00+08:00", **kwargs)

    def say(self, session, **kwargs):
        return session.say("我还在整理", topic="work", sources=self.sources, frame=ContentFrame(), questions=self.slots, **kwargs)

    def test_slots_require_explicit_opt_in(self):
        with self.assertRaises(ValueError): self.say(self.session(inquiry=None))

    def test_exact_reviewed_question_is_delivered_without_free_generation(self):
        session = self.session(realization="free"); seen = []
        result = self.say(session, on_delta=seen.append)
        self.assertEqual(result.say, self.slots[0].question)
        self.assertEqual(result.say, "".join(seen))
        self.assertEqual(result.realization, "application_question")
        self.assertEqual(result.calls_count, 0)
        self.assertEqual(session.inquiry_state.records[0].status, "pending")

    def test_direct_control_needs_no_extraction_and_survives_restart(self):
        session = self.session()
        result = session.say("以后不要追问", topic="work")
        self.assertEqual(result.calls_count, 0); self.assertEqual(self.llm.calls, [])
        session.close(); session = self.session()
        result = self.say(session)
        self.assertEqual(result.question_selection["reason"], "inquiry_denied")
        self.assertEqual(session.state.advice_permission("work"), "unspecified")

    def test_advice_denial_does_not_forbid_information_question(self):
        session = self.session()
        session.boundary("deny", user="不要给建议", quote="不要给建议", topic="work")
        self.assertEqual(self.say(session).decision.action, "ask_question")

    def test_listen_and_existing_clarify_cannot_bypass_question_boundary(self):
        session = self.session()
        result = session.say("只想你听听", topic="work", sources=self.sources, questions=self.slots,
                             frame=ContentFrame(request="listen", request_quote="只想你听听"))
        self.assertEqual(result.decision.action, "listen")
        session.question_boundary(user="不要追问", topic="work")
        result = session.say("继续上次约定", topic="work", frame=ContentFrame(request="resume", request_quote="继续上次约定"))
        self.assertEqual(result.decision.action, "pause")
        self.assertNotIn("？", result.say)
        self.assertEqual(result.calls_count, 0)

    def test_pending_cannot_be_resolved_by_short_reply_silence_or_more_turns(self):
        session = self.session(); self.say(session)
        for _ in range(4):
            result = self.say(session)
            self.assertEqual(result.question_selection["reason"], "pending_question")
        self.assertEqual(session.inquiry_state.records[0].status, "pending")

    def test_confirmed_answer_keeps_quote_and_cooldown_then_selects_next_slot(self):
        session = self.session(); self.say(session)
        session.resolve_question("due", operation="answer", user="明天下午五点", quote="明天下午五点", topic="work")
        self.assertEqual(session.inquiry_state.records[0].answer, "明天下午五点")
        session.close(); session = self.session()
        self.assertEqual(self.say(session).question_selection["reason"], "question_cooldown")
        self.assertEqual(self.say(session).question_selection["selected_id"], "format")

    def test_decline_and_permission_allow_do_not_refund_budget_or_reask_slot(self):
        session = self.session(inquiry=InquiryConfig(initiative=1, max_per_topic=1))
        self.say(session)
        session.resolve_question("due", operation="decline", user="这个先不说", quote="这个先不说", topic="work")
        session.question_boundary(user="可以追问", topic="work")
        self.assertEqual(self.say(session).question_selection["reason"], "topic_question_limit")
        self.assertEqual(session.inquiry_state.records[0].answer, "")

    def test_unconfirmed_callback_does_not_consume_question_or_budget(self):
        session = self.session()
        def fail(text): raise RuntimeError("not delivered")
        with self.assertRaises(RuntimeError): self.say(session, on_delta=fail)
        self.assertEqual(session.inquiry_state.records, ())
        session.close(); session = self.session()
        self.assertEqual(self.say(session).question_selection["selected_id"], "due")

    def test_confirmed_delivery_survives_completion_write_failure(self):
        session = self.session(); append = session.store.append
        def fail(kind, row):
            if kind == "policy_turn": raise RuntimeError("completion write")
            return append(kind, row)
        session.store.append = fail
        with self.assertRaises(RuntimeError): self.say(session)
        session.store.append = append; session.close(); session = self.session()
        self.assertEqual(session.inquiry_state.records[0].slot.id, "due")
        self.assertEqual(self.say(session).question_selection["reason"], "pending_question")

    def test_external_proposals_can_choose_a_legal_slot_but_cannot_change_permissions(self):
        session = self.session()
        result = self.say(session, question_choice=QuestionChoice("format"))
        self.assertEqual(result.question_selection["program_selected_id"], "due")
        self.assertEqual(result.say, self.slots[1].question)
        before = session.inquiry_state
        with self.assertRaises(ValueError): self.say(session, question_choice=QuestionChoice("due"))
        self.assertEqual(session.inquiry_state, before)
        failures = [r for r in session.store.read_ledger() if r["kind"] == "policy_failure"]
        self.assertEqual(failures[-1]["question_choice"], {"slot_id":"due"})
        self.assertEqual(failures[-1]["delivered"], "")

    def test_external_abstention_does_not_silently_ask_program_preference(self):
        session = self.session()
        result = self.say(session, question_choice=QuestionChoice(None))
        self.assertEqual(result.decision.action, "respond")
        self.assertEqual(session.inquiry_state.records, ())

    def test_delivered_slot_id_cannot_be_redefined(self):
        session = self.session(); self.say(session)
        with self.assertRaises(ValueError):
            session.say("我还在整理", topic="work", sources=self.sources, frame=ContentFrame(),
                        questions=(replace(self.slots[0], question="要不要辞职？"),))

    def test_turn_only_control_does_not_create_long_term_permission(self):
        session = self.session()
        result = session.say("这轮不要追问", topic="work")
        self.assertEqual(result.decision.action, "listen")
        self.assertEqual(session.inquiry_state.permissions, ())
        self.assertEqual(self.say(session).decision.action, "ask_question")

    def test_saved_question_boundary_survives_reopening_without_active_inquiry(self):
        session = self.session(); session.question_boundary(user="不要追问", topic="work"); session.close()
        session = self.session(inquiry=None)
        result = session.say("继续上次约定", topic="work", frame=ContentFrame(request="resume", request_quote="继续上次约定"))
        self.assertEqual(result.decision.action, "pause")
        session.say("可以追问", topic="work")
        result = session.say("继续上次约定", topic="work", frame=ContentFrame(request="resume", request_quote="继续上次约定"))
        self.assertEqual(result.decision.action, "clarify")

    def test_explicit_continue_control_can_ask_an_available_reviewed_slot_now(self):
        session = self.session(); session.policy = replace(session.policy, intervention=1)
        session.question_boundary(user="不要追问", topic="work")
        result = session.say("继续问吧", topic="work", sources=self.sources, questions=self.slots)
        self.assertEqual(result.decision.action, "ask_question")
        self.assertEqual(result.calls_count, 0)

    def test_auto_content_receives_question_and_confirmed_answer_as_data_not_authority(self):
        session = self.session(); self.say(session)
        session.resolve_question("due", operation="answer", user="周五", quote="周五", topic="work")
        session.close()
        llm = ScriptedLLM(['{}'])
        session = PolicySession(self.cfg, Store(self.root), llm, inquiry=InquiryConfig(initiative=0),
                                policy=PolicyConfig(intervention=0))
        session.say("还是照这个时间", topic="work")
        payload = json.loads(llm.calls[0][-1]["content"])
        sources = payload["sources"]
        self.assertEqual(len(sources), 1)
        self.assertIn("报告最晚什么时候交", sources[0]["text"])
        self.assertIn("周五", sources[0]["text"])
        self.assertIn("未经外部事实核验", sources[0]["text"])
        self.assertEqual(session.inquiry_state.records[0].status, "answered")


if __name__ == "__main__": unittest.main()
