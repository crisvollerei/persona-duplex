from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import benchmark as b
import mechanism as m


class SituatedChoices(unittest.TestCase):
    def data(self):
        return dict(candidates=[
            dict(id='a',kind='respond',focus='feeling',effects={'draft':None}),
            dict(id='b',kind='propose',focus='practical',effects={'draft':'optional plan'})],
            allowed_actions=['a','b'], requested=False, request_scope=['propose'], draw=.5,
            persona=dict(intervention=.2,practical=.7,feeling=.9,detail=.5),state={'draft':None})

    def test_explicit_request_overrides_inclination(self):
        d=self.data(); d['requested']=True
        for value in (0,1):
            d['persona']['intervention']=value
            self.assertEqual(m.choose(d),'b')

    def test_boundary_overrides_high_intervention(self):
        d=self.data(); d['persona']['intervention']=1; d['allowed_actions']=['a']
        self.assertEqual(m.choose(d),'a')

    def test_static_ablation_can_miss_explicit_request(self):
        d=self.data(); d['requested']=True
        self.assertEqual(m.choose(d),'b')
        self.assertEqual(m.choose(d,contextual=False),'a')

    def test_payloads_share_all_evidence(self):
        d=self.data()
        a,b=m.messages(d,'program'),m.messages(d,'prompt')
        obj=m.strict_json(a[1]['content']); obj.pop('selected_action_id')
        self.assertEqual(obj,m.strict_json(b[1]['content']))
        self.assertIn(m.RULE,a[0]['content']);self.assertIn(m.RULE,b[0]['content'])

    def test_executor_does_not_replace_model_proposal_with_program_choice(self):
        d=self.data()
        self.assertEqual(m.choose(d),'a')
        result=m.execute_effects(d,'b',d['state'])
        self.assertEqual(result['executed_id'],'b')
        self.assertEqual(result['state']['draft'],'optional plan')

    def test_executor_rejects_permission_without_mutating_state(self):
        d=self.data(); d['allowed_actions']=['a']; before=deepcopy(d['state'])
        result=m.execute_effects(d,'b',d['state'])
        self.assertFalse(result['accepted']);self.assertEqual(result['state'],before)
        self.assertEqual(d['state'],before)

    def test_executor_rejects_request_scope(self):
        d=self.data();d['requested']=True
        result=m.execute_effects(d,'a',d['state'])
        self.assertFalse(result['accepted']);self.assertEqual(result['reason'],'request_scope_rejected')

    def test_executor_rejects_unknown_author_effect_key(self):
        d=self.data();d['candidates'][1]['effects']={'real_booking':True}
        result=m.execute_effects(d,'b',d['state'])
        self.assertFalse(result['accepted']);self.assertNotIn('real_booking',result['state'])

    def test_confirmed_update_does_not_alias_mutable_values(self):
        original={'draft':[]};update={'draft':['approved']}
        updated=m.apply_confirmed_updates(original,update)
        updated['draft'].append('later')
        self.assertEqual(original,{'draft':[]});self.assertEqual(update,{'draft':['approved']})

    def test_parser_retains_known_but_forbidden_choice_for_audit(self):
        d=self.data();d['allowed_actions']=['a']
        self.assertEqual(m.parse_reply('{"action_id":"b","say":"x"}',d)['action_id'],'b')

    def test_duplicate_keys_cannot_override_proposal(self):
        with self.assertRaises(ValueError):
            m.parse_reply('{"action_id":"a","action_id":"b","say":"x"}',self.data())

    def test_rejects_nonfinite_state(self):
        with self.assertRaises(ValueError):
            m.apply_confirmed_updates({'draft':None},{'draft':float('nan')})

    def test_no_matching_request_abstains(self):
        d=self.data();d.update(requested=True,request_scope=['ask'])
        self.assertIsNone(m.choose(d))

    def test_four_turn_episode_separates_proposal_execution_and_user_revision(self):
        data = self.data()
        case = dict(id='case',context='fixed evidence',facts={},initial_state={'draft':None},turns=[])
        for i in range(1,5):
            case['turns'].append(dict(id=f'T{i}',user=f'user {i}',requested=i==2,
                request_scope=['propose'] if i==2 else [],
                allowed_actions=['a'] if i==3 else ['a','b'],candidates=deepcopy(data['candidates']),
                user_update={'draft':'cancelled by user'} if i==3 else {}))
        profile=dict(id='p',**data['persona'])
        class FakeClient:
            def __init__(self):self.count=0;self.requests=[]
            def call(self,id,messages,max_tokens):
                self.requests.append(messages);self.count+=1
                action=('b','a','b','a')[self.count-1]
                return dict(status='ok',output=json.dumps(dict(action_id=action,say='actual words')))
        with tempfile.TemporaryDirectory(prefix='persona-situated-harness-') as directory:
            root=Path(directory).resolve()
            self.assertTrue(root.is_absolute() and root.name.startswith('persona-situated-harness-'))
            b.save(root/'cases.json',[case]);b.save(root/'profiles.json',[profile]);b.save(root/'protocol.json',{})
            client=FakeClient()
            with patch.object(b,'HERE',root):
                result=b.episode(dict(id='job',case='case',profile='p',method='program',rep=1),client,root/'output')
            rows=result['turns']
            self.assertEqual(client.count,4)
            self.assertEqual(rows[0]['model_reported_id'],'b')
            self.assertEqual(rows[0]['executed_id'],'b')
            self.assertEqual(rows[1]['execution_error'],'request_scope_rejected')
            self.assertIsNone(rows[1]['executed_id'])
            self.assertEqual(rows[1]['state_after_action']['draft'],'optional plan')
            self.assertEqual(rows[2]['execution_error'],'permission_rejected')
            self.assertEqual(rows[2]['state_after_user']['draft'],'cancelled by user')
            self.assertEqual(rows[2]['state_after_action']['draft'],'cancelled by user')
            self.assertEqual(rows[3]['state_before']['draft'],'cancelled by user')
            self.assertEqual(rows[3]['state_after_action']['draft'],None)
            self.assertEqual(len(rows[3]['payload']['history']),6)
            self.assertTrue(all(r['say']=='actual words' for r in rows))


if __name__=='__main__':
    unittest.main()
