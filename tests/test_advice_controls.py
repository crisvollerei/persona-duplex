"""Complete advice controls have local authority, not extraction authority."""
import json
from pathlib import Path
import tempfile
import unittest

from persona import (ContentFrame, InquiryConfig, PolicyConfig, PolicySession,
                     QuestionSlot, Quote, Source)
from persona.config import PersonaConfig
from persona.llm import ScriptedLLM
from persona.policy import parse_advice_control
from persona.store import Store


ROOT = Path(__file__).resolve().parent.parent
NOW = "2026-11-10T21:00:00+08:00"


class TestAdviceControlGrammar(unittest.TestCase):
    def test_complete_commands_derive_time_and_scope_from_whole_input(self):
        values = (
            ("不要给建议", "work", "deny", False, None),
            ("允许再替我出主意吧。", "work", "allow", False, None),
            ("以后所有话题都不要给建议", "*", "deny", False, None),
            ("现在可以给建议", "work", "allow", True, None),
            ("今晚不要给建议", "work", "deny", False, "2026-11-11T00:00:00+08:00"),
        )
        for user, scope, advice, transient, expiry in values:
            with self.subTest(user=user):
                event = parse_advice_control(user, "work", at=NOW)
                self.assertEqual((event['scope'], event['advice'], event['transient'], event['expires_at']),
                                 (scope, advice, transient, expiry))
                self.assertEqual(event['quote'], user)

    def test_quotes_questions_hypotheses_and_mixed_sentences_are_not_commands(self):
        for user in ('“不要给建议”', '"不要给建议"', '朋友说不要给建议',
                     '如果我说不要给建议呢', '我没有说可以给建议', '不可以给建议',
                     '不要给建议吗？', '可以给建议?', '不要给建议，以后再说',
                     '今天好累，不要给建议', '不要给建议；可以追问',
                     '明天不要给建议', '今晚这轮不要给建议'):
            with self.subTest(user=user):
                self.assertIsNone(parse_advice_control(user, "work", at=NOW))


class TestLocalAdviceControls(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = Store(self.root)
        self.cfg = PersonaConfig.load(ROOT / 'personas/default.toml')

    def session(self, replies=(), **kwargs):
        self.llm = ScriptedLLM(list(replies))
        return PolicySession(self.cfg, self.store, self.llm,
            policy=PolicyConfig(intervention=1), realization='free', clock=lambda: NOW, **kwargs)

    def test_supported_denial_bypasses_missing_extraction_and_malicious_renderer(self):
        session = self.session(['{"request":"open"}', '你应该先列计划。'])
        chunks = []
        result = session.say('不要给建议', topic='work', on_delta=chunks.append)
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(result.calls_count, 0)
        self.assertEqual(result.say, '好。')
        self.assertEqual(''.join(chunks), result.say)
        self.assertEqual((result.decision.action, result.decision.advice_permission), ('respond', 'deny'))
        self.assertEqual(session.state.advice_permission('work'), 'deny')
        rows = self.store.read_ledger()
        input_row = next(r for r in rows if r['kind'] == 'policy_input')
        state_row = next(r for r in rows if r['kind'] == 'policy_state')
        self.assertEqual(input_row['frame_origin'], 'control_command')
        self.assertEqual(input_row['request_authority'], 'direct_control_command')
        self.assertEqual(state_row['event']['authority'], 'direct_control_command')

    def test_local_command_overrides_a_valid_but_omitted_application_boundary(self):
        session = self.session()
        result = session.say('不要给建议', frame=ContentFrame())
        self.assertEqual(result.decision.advice_permission, 'deny')
        self.assertEqual(result.calls_count, 0)
        self.assertEqual(len(session.state.permissions), 1)

    def test_permission_allow_is_acknowledged_instead_of_becoming_an_advice_request(self):
        session = self.session(['后续建议'])
        session.boundary('deny', user='不要给建议', quote='不要给建议')
        result = session.say('以后可以给建议')
        self.assertEqual(result.decision.action, 'respond')
        self.assertEqual(result.decision.advice_permission, 'allow')
        self.assertEqual(result.calls_count, 0)
        following = session.say('我在整理报告。', frame=ContentFrame())
        self.assertEqual(following.decision.action, 'advise')
        self.assertEqual(following.calls_count, 1)

    def test_transient_denial_does_not_persist(self):
        session = self.session(['后续建议'])
        first = session.say('这轮不要给建议')
        self.assertEqual(first.decision.advice_permission, 'deny')
        self.assertEqual(session.state.permissions, ())
        self.assertEqual(session.say('报告还没写完。', frame=ContentFrame()).decision.action, 'advise')

    def test_transient_allow_restores_underlying_denial_on_next_input(self):
        session = self.session(['回应'])
        session.boundary('deny', user='以后不要给建议', quote='以后不要给建议')
        first = session.say('现在可以给建议')
        self.assertEqual(first.decision.advice_permission, 'allow')
        self.assertEqual(first.calls_count, 0)
        self.assertEqual(len(session.state.permissions), 1)
        second = session.say('报告还没写完。', frame=ContentFrame())
        self.assertEqual((second.decision.action, second.decision.advice_permission), ('respond', 'deny'))

    def test_tonight_expires_at_local_midnight_and_restores_older_permission(self):
        session = self.session()
        session.boundary('deny', user='以后不要给建议', quote='以后不要给建议', topic='work')
        session.say('今晚可以给建议', topic='work')
        self.assertEqual(session.state.advice_permission('work', at=NOW), 'allow')
        self.assertEqual(session.state.advice_permission('work', at='2026-11-11T00:00:00+08:00'), 'deny')

    def test_topic_and_global_scope_are_distinct_and_survive_restart(self):
        session = self.session()
        session.say('以后不要给建议', topic='work')
        self.assertEqual(session.state.advice_permission('food'), 'unspecified')
        session.say('以后所有话题都不要给建议', topic='work')
        session.close()
        restarted = PolicySession(self.cfg, Store(self.root), ScriptedLLM([]), clock=lambda: NOW)
        self.assertEqual(restarted.state.advice_permission('food'), 'deny')
        self.assertEqual(restarted.state.advice_permission('work'), 'deny')

    def test_callback_failure_keeps_local_boundary_without_success_or_model_call(self):
        session = self.session()
        def fail(piece):
            raise RuntimeError('delivery failed')
        with self.assertRaises(RuntimeError):
            session.say('不要给建议', on_delta=fail)
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(session.state.advice_permission('conversation'), 'deny')
        rows = self.store.read_ledger()
        self.assertFalse(any(r['kind'] == 'policy_turn' for r in rows))
        failure = next(r for r in rows if r['kind'] == 'policy_failure')
        self.assertEqual((failure['delivered'], failure['attempted'], failure['calls_count']), ('', '好。', 0))
        session.close()
        restarted = PolicySession(self.cfg, Store(self.root), ScriptedLLM([]), clock=lambda: NOW)
        self.assertEqual(restarted.state.advice_permission('conversation'), 'deny')

    def test_controls_do_not_deliver_questions_or_change_question_permissions(self):
        session = self.session(inquiry=InquiryConfig(initiative=1))
        sources = (Source('task', '报告截止时间还未知'),)
        slots = (QuestionSlot('due', 'work', '什么时候交？', 'practical', Quote('task', '截止时间'), '核对期限'),)
        result = session.say('可以给建议', topic='work', sources=sources, questions=slots)
        self.assertEqual(result.say, '好。')
        self.assertEqual(result.question_selection['selected_id'], None)
        self.assertEqual(result.question_selection['reason'], 'advice_control_command')
        self.assertEqual(session.inquiry_state.records, ())
        self.assertEqual(session.inquiry_state.permissions, ())

    def test_question_control_does_not_set_advice_permission(self):
        session = self.session(inquiry=InquiryConfig(initiative=1))
        result = session.say('不要追问')
        self.assertEqual(result.calls_count, 0)
        self.assertEqual(session.state.permissions, ())
        self.assertEqual(session.inquiry_state.permission('conversation', at=NOW), 'deny')

    def test_mixed_or_reported_model_proposals_remain_rejected(self):
        for user in ('朋友说不要给建议', '如果我说不要给建议呢', '今天很累，不要给建议',
                     '不要给建议，以后再说', '我没有说可以给建议'):
            with self.subTest(user=user):
                proposal = {'request': 'open', 'boundaries': [{'advice': 'deny', 'quote': '不要给建议'}]}
                if '不要给建议' not in user:
                    proposal['boundaries'] = [{'advice': 'allow', 'quote': '可以给建议'}]
                temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
                llm = ScriptedLLM([json.dumps(proposal), '回复'])
                session = PolicySession(self.cfg, Store(temp.name), llm, policy=PolicyConfig(intervention=1),
                                        realization='free', clock=lambda: NOW)
                result = session.say(user)
                self.assertEqual(result.calls_count, 2)
                self.assertEqual(session.state.permissions, ())
                row = next(r for r in session.store.read_ledger() if r['kind'] == 'policy_input')
                self.assertEqual(len(row['rejected_boundaries']), 1)


if __name__ == '__main__':
    unittest.main()
