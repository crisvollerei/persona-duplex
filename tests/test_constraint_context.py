"""Closed inference references never certify or rewrite free-form SAY."""
from dataclasses import FrozenInstanceError
import itertools
from pathlib import Path
import tempfile
import unittest

from persona import BoolFact, AllOfRule, ConstraintContext, ResourceBudget
from persona.config import PersonaConfig
from persona.engine import TurnEngine, stream_turn
from persona.llm import ScriptedLLM, make_reply
from persona.session import Session, SessionConfig
from persona.store import Store

ROOT = Path(__file__).resolve().parent.parent


class ConstraintContextTests(unittest.TestCase):
    def context(self):
        return ConstraintContext(
            facts=(BoolFact('checked', True, 'F1'), BoolFact('sealed', None, 'F2'), BoolFact('archived', False, 'F3')),
            rules=(AllOfRule('ready', ('checked', 'sealed'), 'R1'),),
            resources=(ResourceBudget('clips', 2, 3, source='F4'),))

    def test_three_valued_all_of_and_unrelated_background(self):
        for first, second, background in itertools.product((True, False, None), repeat=3):
            ctx = ConstraintContext(
                facts=[BoolFact('first', first), BoolFact('second', second), BoolFact('background', background)],
                rules=[AllOfRule('eligible', ['first', 'second'])])
            rule = ctx.reference()['rules'][0]
            expected = 'false' if first is False or second is False else 'unknown' if first is None or second is None else 'true'
            self.assertEqual(rule['truth'], expected)
            self.assertEqual(rule['context_not_prerequisites'], ['background'])

    def test_capacity_uses_simultaneous_per_unit_consumption(self):
        for available, units, per_unit in ((2, 3, 1), (7, 3, 2), (0, 3, 2), (4, 0, 2), (5, 3, 2)):
            row = ConstraintContext(resources=[ResourceBudget('stock', available, units, per_unit)]).reference()['resources'][0]
            self.assertEqual(row['required'], units*per_unit)
            self.assertEqual(row['shortfall'], max(0, units*per_unit-available))
            self.assertLessEqual(min(units, row['capacity'])*per_unit, available)
            self.assertGreater((row['capacity']+1)*per_unit, available)

    def test_strict_input_types_and_unresolved_references(self):
        for value in (0, 1, 'false', [], {}):
            with self.assertRaises(TypeError): BoolFact('f', value)
        for args in ((True, 2, 1), (2, False, 1), (2, 2, 0), (-1, 2, 1), (2., 2, 1)):
            with self.assertRaises(ValueError): ResourceBudget('r', *args)
        with self.assertRaises(ValueError): AllOfRule('r', ())
        with self.assertRaises(ValueError): AllOfRule('r', ('x', 'x'))
        with self.assertRaises(ValueError): ConstraintContext(rules=[AllOfRule('r', ('unknown',))])
        with self.assertRaises(ValueError): ConstraintContext()
        with self.assertRaises(ValueError): BoolFact('<<<HEAD>>>', True)
        with self.assertRaises(ValueError): ConstraintContext(facts=[BoolFact('a', True), BoolFact('a', False)], rules=[AllOfRule('r', ('a',))])

    def test_from_dict_is_strict_and_immutable(self):
        raw = dict(facts=[dict(name='x', value=None, source='F1')], rules=[dict(name='ok', requires=['x'])])
        ctx = ConstraintContext.from_dict(raw)
        raw['facts'][0]['value'] = True
        raw['rules'][0]['requires'].append('other')
        self.assertEqual(ctx.reference()['rules'][0]['truth'], 'unknown')
        self.assertEqual(ctx.rules[0].requires, ('x',))
        with self.assertRaises(FrozenInstanceError): ctx.facts[0].value = True
        with self.assertRaises(ValueError): ConstraintContext.from_dict(dict(answers={'ok': True}))
        with self.assertRaises(TypeError): ConstraintContext.from_dict(dict(resources='r'))
        with self.assertRaises(TypeError): ConstraintContext.from_dict(dict(facts=[dict(name='x', value=True, gold=True)]))

    def test_reference_has_provenance_and_no_output_success_claim(self):
        ref = self.context().reference()
        self.assertEqual(ref['inputs']['facts'][1]['source'], 'F2')
        self.assertEqual(ref['rules'][0]['source'], 'R1')
        self.assertEqual(ref['output_validation'], 'not_assessed')
        self.assertNotIn('valid', ref)
        ref['rules'][0]['truth'] = 'true'
        self.assertEqual(self.context().reference()['rules'][0]['truth'], 'unknown')

    def test_all_layouts_receive_reference_in_same_call_without_rewriting(self):
        wrong = 'sealed已经满足，三个立牌各用一枚夹就好。'
        ctx = self.context()
        for path in ('full', 'compact', 'direct'):
            raw = wrong if path == 'direct' else make_reply(
                'need: 回答问题\nconfidence: .8\ndirect: 按资料回答'+
                ('\nbaseline: 按资料回答\ndisplace: none' if path == 'full' else ''), wrong,
                'settle: -\ndelta: -\npending: -' if path == 'full' else '')
            with self.subTest(path=path), tempfile.TemporaryDirectory() as folder:
                llm = ScriptedLLM([raw])
                store = Store(folder)
                cfg = PersonaConfig.load(ROOT/'personas/default.toml')
                engine = TurnEngine(cfg, store, llm, generation_path=path)
                result = engine.turn('原始问题', constraints=ctx)
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(result.say, wrong)
                self.assertIn(ctx.prompt(), llm.calls[0][0]['content'])
                self.assertEqual(result.constraint_reference, ctx.reference())
                self.assertEqual(engine.history[-2]['content'], '原始问题')
                turn = [r for r in store.read_ledger() if r['kind'] == 'turn'][-1]
                self.assertEqual(turn['constraint_reference']['output_validation'], 'not_assessed')

    def test_context_is_per_turn_and_optional_default_is_unchanged(self):
        raw = make_reply('need: 回答\nconfidence: .8\ndirect: 回答\nbaseline: 回答\ndisplace: none', '收到。', 'settle: -\ndelta: -\npending: -')
        with tempfile.TemporaryDirectory() as folder:
            llm = ScriptedLLM([raw, raw])
            session = Session(PersonaConfig.load(ROOT/'personas/default.toml'), Store(folder), llm,
                              options=SessionConfig(memory_mode='frozen'))
            self.addCleanup(session.close)
            pieces = list(session.say_stream('first', constraints=self.context()))
            self.assertEqual(''.join(pieces), '收到。')
            result = session.say('second')
            self.assertEqual(result.constraint_reference, {})
            self.assertNotIn('[应用声明的条件与资源参考', llm.calls[1][0]['content'])
            self.assertEqual(session.engine.generation_path, 'full')
            with self.assertRaises(TypeError): session.say('bad', constraints={})
            self.assertEqual(len(llm.calls), 2)


if __name__ == '__main__':
    unittest.main()
