"""Offline integration contracts for the archived research implementation."""
import tempfile
import unittest
import benchmark as b


class MockClient:
    def __init__(self,raw):
        self.raw=raw;self.messages=[]

    def call(self,id,messages):
        self.messages.append(messages)
        return {'status':'ok','output':self.raw}


class PilotContracts(unittest.TestCase):
    def setUp(self):
        self.m=b.modules()
        self.profiles=b.load(b.HERE/'profiles.json')
        self.materials=b.load(b.HERE/'materials.json')

    def test_all_materials_initialize_and_recall_without_model(self):
        self.assertEqual(len(self.materials['development']),6)
        self.assertEqual(len(self.materials['test']),24)
        self.assertEqual(len({c['family'] for c in self.materials['test']}),8)
        for c in self.materials['development']+self.materials['test']:
            self.assertLessEqual(len(c['history']),12)
            for p in self.profiles:
                with tempfile.TemporaryDirectory() as td:
                    cfg=b.make_config(self.m,p);store=b.seed_store(self.m,c,td)
                    matches,context=b.shared_context(self.m,cfg,store,c)
                    self.assertLessEqual(len(matches),5)
                    self.assertIsInstance(context,str)

    def test_real_session_and_direct_prompt_share_context_and_one_call(self):
        c=self.materials['test'][3];p=self.profiles[0]
        text='100-3×18-27=19\n结余：19元'
        raw=self.m['llm'].make_reply('need: 只是想说说话\nconfidence: 0.9\ndirect: 19元\nbaseline: 19元\ndisplace: none',
                                   text,'settle: -\ndelta: -\npending: -')
        fclient=MockClient(raw);bclient=MockClient(text)
        f=b.one(self.m,fclient,c,p,'framework',None)
        r=b.one(self.m,bclient,c,p,'strong_prompt',b.load(b.HERE/'prompts.json')['candidates'][0])
        self.assertEqual((f['status'],r['status']),('ok','ok'))
        self.assertEqual(f['evidence_sha256'],r['evidence_sha256'])
        self.assertEqual(f['say'],r['say'])
        self.assertEqual(f['mechanical']['status'],'passed')
        self.assertEqual(len(fclient.messages),1);self.assertEqual(len(bclient.messages),1)
        self.assertIn(f['memory_context'],fclient.messages[0][0]['content'])

    def test_recall_retains_relevant_old_values_and_excludes_invalid_sources(self):
        cases={c['id']:c for c in self.materials['test']}
        expected={'T16':['17','1.4'],'T17':['MT-','041'],'T20':['A5'],'T21':['PNG']}
        for id,fragments in expected.items():
            with tempfile.TemporaryDirectory() as td:
                c=cases[id];cfg=b.make_config(self.m,self.profiles[0]);store=b.seed_store(self.m,c,td)
                _,context=b.shared_context(self.m,cfg,store,c)
                for fragment in fragments:self.assertIn(fragment,context)
        with tempfile.TemporaryDirectory() as td:
            c=cases['T18'];cfg=b.make_config(self.m,self.profiles[0]);store=b.seed_store(self.m,c,td)
            rows,_=b.shared_context(self.m,cfg,store,c)
            self.assertFalse(any(j.get('source')=='llm' for j in rows))

    def test_structural_failure_is_not_retried_or_silently_repaired(self):
        c=self.materials['test'][3];client=MockClient('direct: leaked reference')
        result=b.one(self.m,client,c,self.profiles[0],'framework',None)
        self.assertEqual(len(client.messages),1)
        self.assertEqual(result['status'],'empty_reply')
        self.assertEqual(result['say'],'')
        self.assertFalse(result['validity']['valid'])


if __name__=='__main__':unittest.main()
