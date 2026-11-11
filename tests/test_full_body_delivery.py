"""Complete authorized SAY code survives bad metadata without granting state."""
from pathlib import Path
import tempfile
import unittest

from persona import ReplyContract
from persona.body import prepare_code_body
from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.parse import say_wire
from persona.store import Store


ROOT = Path(__file__).resolve().parent.parent
HEAD = ('need: 用户请求代码\nconfidence: .9\ndirect: PRIVATE_REFERENCE\n'
        'baseline: 按格式提供代码\ndisplace: none')
TAIL = 'settle: -\ndelta: -\npending: -'
CODE = '```python\ndirect: str = "west"\npending: int = 6\nready: bool = False\n```'
CONTRACT = ReplyContract.code_block(language='python')


class FullBodyDeliveryTests(unittest.TestCase):
    def cfg(self):
        return PersonaConfig.load(ROOT/'personas/default.toml')

    def run_reply(self, raw, *, streaming=False, old_pending=False, contract=CONTRACT):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        cfg = self.cfg()
        store = Store(folder.name)
        pending = store.add_pending('old reading', .6, 'old evidence', 't0', cfg.limits) if old_pending else None
        if pending is not None:
            raw = raw.replace('OLD_PENDING', pending['id'])
        llm = ScriptedLLM([raw])
        calls, deltas = [], []
        def stream(messages):
            calls.append(messages)
            yield from raw
        engine = TurnEngine(cfg, store, llm, generation_path='full',
                            stream_llm=stream if streaming else None)
        before = store.relation()
        result = engine.turn('请提供指定Python示例。', contract=contract,
                             on_delta=deltas.append if streaming else None)
        self.assertEqual(len(llm.calls)+len(calls), 1)
        if streaming:
            self.assertEqual(''.join(deltas), result.say)
        return engine, result, before, pending

    def assert_metadata_rejected(self, engine, result, before):
        self.assertEqual(result.say, CODE)
        self.assertTrue(result.contract_validation['valid'])
        self.assertFalse(result.validity['valid'])
        self.assertEqual(result.relation_changed, {})
        self.assertIsNone(result.pending_added)
        self.assertEqual(result.settled, [])
        self.assertEqual(engine.store.relation(), before)
        self.assertEqual(list(engine.store.buffer_dir.glob('*.json')), [])
        self.assertEqual(engine.history[-1]['content'], say_wire(result.say))
        self.assertNotIn('PRIVATE_REFERENCE', str(result.say))

    def test_recorded_failure_shape_preserves_body_and_rejects_footer(self):
        # The original regression is a correct SAY block plus an orphan fence
        # after TAIL; the raw return must not stand in for actual delivery.
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                raw = make_reply(HEAD, CODE, TAIL+'\n```')
                engine, result, before, _ = self.run_reply(raw, streaming=streaming)
                self.assert_metadata_rejected(engine, result, before)
                self.assertIn('fence_outside_say', result.validity['reasons'])
                self.assertIn('fence_outside_say', engine._probe['body_framing']['outer_protocol_errors'])

    def test_bad_outer_footer_never_settles_or_adds_pending_or_delta(self):
        tail = ('settle: OLD_PENDING | confirmed | invented\ndelta:\n  closeness: .05\n'
                'pending:\n  claim: new reading\n  confidence: .7\n  predicts: new evidence\n```')
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                engine, result, before, pending = self.run_reply(make_reply(HEAD, CODE, tail),
                    streaming=streaming, old_pending=True)
                self.assert_metadata_rejected(engine, result, before)
                self.assertEqual([p['id'] for p in engine.store.pendings()], [pending['id']])

    def test_missing_empty_and_malformed_tail_are_independent_of_complete_say(self):
        outputs = (make_reply(HEAD, CODE), make_reply(HEAD, CODE)+'\n<<<TAIL>>>\n',
                   make_reply(HEAD, CODE, 'not metadata'),
                   make_reply(HEAD, CODE, 'delta:\n  closeness: .05\npending: (a new prediction)'),
                   make_reply(HEAD, CODE, 'delta:\n  closeness: .05\npending:\n  claim: c\n  predicts: p'),
                   make_reply(HEAD, CODE, TAIL+'\nextra prose'),
                   make_reply(HEAD, CODE, TAIL+'\npending:\n  claim: x\n  confidence: .8\n  predicts: y'))
        for raw in outputs:
            with self.subTest(raw=raw):
                engine, result, before, _ = self.run_reply(raw)
                self.assert_metadata_rejected(engine, result, before)

    def test_duplicate_or_out_of_order_outer_markers_cannot_promote_state(self):
        for marker in ('<<<TAIL>>>', '<<<SAY>>>', '<<<HEAD>>>', '<<<tail>>>'):
            with self.subTest(marker=marker):
                raw = make_reply(HEAD, CODE, TAIL+'\n'+marker+'\ndelta:\n  closeness: .05')
                engine, result, before, _ = self.run_reply(raw)
                self.assert_metadata_rejected(engine, result, before)
                self.assertTrue(result.parsed.structural_errors)

    def test_valid_full_tail_still_has_its_normal_authority(self):
        tail = ('settle: -\ndelta:\n  closeness: .05\npending:\n'
                '  claim: new reading\n  confidence: .7\n  predicts: new evidence')
        engine, result, _, _ = self.run_reply(make_reply(HEAD, CODE, tail))
        self.assertEqual(result.say, CODE)
        self.assertTrue(result.validity['valid'], result.validity)
        self.assertEqual(result.relation_changed, {'closeness': .05})
        self.assertIsNotNone(result.pending_added)
        self.assertEqual(len(list(engine.store.buffer_dir.glob('*.json'))), 1)

    def test_invalid_top_level_values_cannot_authorize_other_state_fields(self):
        valid_delta = 'delta:\n  closeness: .05'
        valid_pending = 'pending:\n  claim: new reading\n  confidence: .7\n  predicts: new evidence'
        tails = [f'settle: {value}\n{valid_delta}\n{valid_pending}'
                 for value in ('garbage', 'watch', 'p_old | maybe | invented', '| confirmed | invented')]
        tails += [f'settle: -\ndelta: closeness: .05\n{valid_pending}',
                  f'settle: -\n{valid_delta}\npending: a new prediction']
        for tail in tails:
            with self.subTest(tail=tail):
                engine, result, before, _ = self.run_reply(make_reply(HEAD, CODE, tail))
                self.assert_metadata_rejected(engine, result, before)
                self.assertTrue(any(r.startswith('invalid_code_tail_') for r in result.validity['reasons']))

    def test_legal_empty_markers_and_repeated_settle_remain_valid(self):
        for marker in ('-', 'none', '无', '（无）', '(none)'):
            with self.subTest(marker=marker):
                tail = f'settle: -\nsettle: -\ndelta: {marker}\npending: {marker}'
                _, result, _, _ = self.run_reply(make_reply(HEAD, CODE, tail))
                self.assertEqual(result.say, CODE)
                self.assertTrue(result.validity['valid'], result.validity)
        tail = 'settle: OLD_PENDING | watch\nsettle: other | refuted | explicit\ndelta: -\npending: -'
        engine, result, _, pending = self.run_reply(make_reply(HEAD, CODE, tail), old_pending=True)
        self.assertTrue(result.validity['valid'], result.validity)
        self.assertEqual([p['id'] for p in engine.store.pendings()], [pending['id']])

    def test_untrusted_or_incomplete_say_is_not_recovered(self):
        cases = (make_reply(HEAD, CODE, TAIL), make_reply(HEAD, CODE[:-3], TAIL),
                 make_reply(HEAD, CODE+'\nexplanation', TAIL),
                 make_reply(HEAD, 'explanation\n'+CODE, TAIL),
                 make_reply(HEAD, CODE+'\n'+CODE, TAIL))
        for index, raw in enumerate(cases):
            with self.subTest(index=index):
                engine, result, _, _ = self.run_reply(raw, contract=None if index==0 else CONTRACT)
                self.assertEqual(result.say, '')
                self.assertFalse(result.validity['valid'])
                self.assertIsNone(result.pending_added)

    def test_completed_code_before_interrupted_stream_is_not_delivered(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.cfg()
            store = Store(folder)
            calls, deltas = [], []
            llm = ScriptedLLM([])
            def stream(messages):
                calls.append(messages)
                yield make_reply(HEAD, CODE)
                raise RuntimeError('offline disconnection before completion')
            engine = TurnEngine(cfg, store, llm, stream_llm=stream)
            with self.assertRaises(RuntimeError):
                engine.turn('代码示例', contract=CONTRACT, on_delta=deltas.append)
            self.assertEqual(deltas, [])
            self.assertEqual(len(calls), 1)
            self.assertEqual(llm.calls, [])
            self.assertEqual(list(store.buffer_dir.glob('*.json')), [])

    def test_preparation_retains_all_outer_bytes_for_audit(self):
        raw = make_reply(HEAD, CODE, TAIL+'\n```\nPRIVATE_FOOTER')
        framed, errors = prepare_code_body(raw, 'full', CONTRACT)
        self.assertTrue(framed.endswith(TAIL+'\n```\nPRIVATE_FOOTER'))
        self.assertIn('fence_outside_say', errors)
        self.assertIn('unrecognized_code_tail_content', errors)


if __name__ == '__main__':
    unittest.main()
