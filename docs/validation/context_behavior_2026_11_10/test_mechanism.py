import copy
import unittest
import mechanism as m


class Choices(unittest.TestCase):
    def data(self):
        return dict(allowed_actions=['a', 'b'], requested=False, request_scope=['propose'], draw=.5,
                    persona=dict(intervention=.2, detail=.8, feeling=.2, practical=.5), candidates=[
                        dict(id='a', kind='respond', focus='detail'),
                        dict(id='b', kind='propose', focus='practical')])

    def test_requested_proposal_is_fulfillment_for_both_preferences(self):
        d = self.data(); d['requested'] = True
        for intervention in (0, 1):
            d['persona']['intervention'] = intervention
            self.assertEqual(m.choose(d), 'b')

    def test_permission_blocks_high_intervention(self):
        d = self.data(); d['persona']['intervention'] = 1; d['allowed_actions'] = ['a']
        self.assertEqual(m.choose(d), 'a')

    def test_context_ablation_keeps_distinction(self):
        d = self.data(); d['requested'] = True
        self.assertEqual(m.choose(d), 'b')
        self.assertEqual(m.choose(d, contextual=False), 'a')

    def test_local_intervention_does_not_mutate_evidence_or_permission(self):
        d = self.data(); original = copy.deepcopy(d)
        self.assertEqual(m.choose(d), 'a')
        d['persona']['intervention'] = 1
        self.assertEqual(m.choose(d), 'b')
        self.assertEqual(d['candidates'], original['candidates'])
        self.assertEqual(d['allowed_actions'], original['allowed_actions'])

    def test_unsatisfiable_request_abstains(self):
        d = self.data(); d.update(requested=True, request_scope=['ask'])
        self.assertIsNone(m.choose(d))

    def test_strict_parser_rejects_silent_duplicate_override(self):
        with self.assertRaises(ValueError):
            m.parse_reply('{"action_id":"a","action_id":"b","say":"x"}', self.data())

    def test_parser_does_not_launder_disallowed_but_known_action(self):
        d = self.data(); d['allowed_actions'] = ['a']
        self.assertEqual(m.parse_reply('{"action_id":"b","say":"x"}', d)['action_id'], 'b')

    def test_same_payload_and_rule_for_both_methods(self):
        d = self.data()
        program, prompt = m.messages(d, 'program'), m.messages(d, 'prompt')
        obj = m.strict_json(program[1]['content']); obj.pop('selected_action_id')
        self.assertEqual(obj, m.strict_json(prompt[1]['content']))
        self.assertTrue(all(m.RULE in x[0]['content'] for x in (program, prompt)))


if __name__ == '__main__':
    unittest.main()
