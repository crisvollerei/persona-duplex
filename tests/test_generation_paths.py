"""Single-call layouts share state without inventing absent observations."""
from pathlib import Path
import tempfile
import unittest

from persona import ReplyContract, ResponseScope
from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.session import Session, SessionConfig
from persona.store import Store

ROOT=Path(__file__).resolve().parent.parent
HEAD='need: 只是想说说话\nconfidence: 0.8\ndirect: 数据未确认'
SAY='草稿：这一页，终于和你相见。'


def reply(path,say=SAY):
    if path=='direct':return say
    return make_reply(HEAD+('\nbaseline: '+SAY+'\ndisplace: none' if path=='full' else ''),
                      say,'settle: -\ndelta: -\npending: -' if path=='full' else '')


class GenerationPathsTests(unittest.TestCase):
    def cfg(self):return PersonaConfig.load(ROOT/'personas/shorekeeper.toml')

    def test_all_layouts_one_call_and_same_declared_scope(self):
        scope=ResponseScope(max_sentences=1,prefix='草稿：',allow_questions=False)
        for path in ('full','compact','direct'):
            for streaming in (False,True):
                with self.subTest(path=path,streaming=streaming),tempfile.TemporaryDirectory() as td:
                    llm=ScriptedLLM([reply(path)]);streams=[];deltas=[]
                    def stream(messages):
                        streams.append(messages);yield from reply(path)
                    engine=TurnEngine(self.cfg(),Store(td),llm,generation_path=path,
                                      stream_llm=stream if streaming else None)
                    result=engine.turn('写一句未来文案。',scope=scope,on_delta=deltas.append if streaming else None)
                    self.assertEqual(result.say,SAY)
                    self.assertTrue(result.validity['valid'],result.validity)
                    self.assertEqual(len(llm.calls)+len(streams),1)
                    messages=(llm.calls or streams)[0]
                    self.assertIn(scope.prompt(),messages[0]['content'])
                    if streaming:self.assertEqual(''.join(deltas),SAY)
                    self.assertEqual(len(list(engine.store.buffer_dir.glob('*.json'))),1)
                    event=[r for r in engine.store.read_ledger() if r['kind']=='turn'][-1]
                    self.assertEqual(event['generation_path'],path)
                    self.assertEqual(event['response_scope']['prefix'],'草稿：')
                    if path=='direct':
                        self.assertIsNone(result.parsed.head)
                        self.assertEqual(event['observations'],'not_collected')
                        self.assertIsNone(event['confidence'])
                    if path!='full':
                        self.assertIsNone(result.parsed.tail)
                        self.assertEqual(result.pending_added,None)
                        self.assertNotIn('<<<TAIL>>>',engine.history[-1]['content'])

    def test_explicit_layouts_do_not_silently_mix_multicall_protocols(self):
        for path,protocol in [('bad','text'),('direct','persona'),('compact','auto'),('direct','tool')]:
            with tempfile.TemporaryDirectory() as td,self.assertRaises(ValueError):
                TurnEngine(self.cfg(),Store(td),ScriptedLLM([]),generation_path=path,protocol=protocol)

    def test_compact_tail_never_changes_state(self):
        raw=reply('compact')+'\n<<<TAIL>>>\ndelta: closeness: 1\npending: claim: 新预测'
        with tempfile.TemporaryDirectory() as td:
            store=Store(td);engine=TurnEngine(self.cfg(),store,ScriptedLLM([raw]),generation_path='compact')
            result=engine.turn('你好')
            self.assertFalse(result.validity['valid'])
            self.assertIsNone(result.parsed.tail)
            self.assertEqual(result.relation_changed,{})
            self.assertEqual(store.pendings(),[])
            self.assertEqual(list(store.buffer_dir.glob('*.json')),[])

    def test_direct_rejects_internal_format_without_recovery_call(self):
        for streaming in (False,True):
            with self.subTest(streaming=streaming),tempfile.TemporaryDirectory() as td:
                raw=reply('full');llm=ScriptedLLM([raw]);streams=[];deltas=[]
                def stream(messages):streams.append(messages);yield from raw
                engine=TurnEngine(self.cfg(),Store(td),llm,generation_path='direct',
                    stream_llm=stream if streaming else None)
                result=engine.turn('你好',on_delta=deltas.append if streaming else None)
                self.assertEqual(result.say,'')
                self.assertFalse(result.validity['valid'])
                self.assertEqual(len(llm.calls)+len(streams),1)
                self.assertEqual(''.join(deltas),'')

    def test_scope_failure_records_original_delivery_without_truncation(self):
        text='草稿：第一句话。第二句话。'
        with tempfile.TemporaryDirectory() as td:
            llm=ScriptedLLM([text]);store=Store(td)
            with Session(self.cfg(),store,llm,options=SessionConfig(memory_mode='frozen',generation_path='direct')) as session:
                delivered=[]
                result=session.say('只写一句。',scope=ResponseScope(max_sentences=1,prefix='草稿：'),on_delta=delivered.append)
                self.assertEqual(''.join(delivered),text)
                self.assertEqual(result.say,text)
                self.assertFalse(result.scope_validation['valid'])
                self.assertTrue(result.validity['valid'])
                self.assertEqual(len(llm.calls),1)

    def test_light_history_does_not_fabricate_metadata(self):
        for path in ('compact','direct'):
            with tempfile.TemporaryDirectory() as td:
                llm=ScriptedLLM([reply(path),reply(path)])
                with Session(self.cfg(),Store(td),llm,options=SessionConfig(memory_mode='frozen',generation_path=path)) as session:
                    session.say('第一轮');session.say('第二轮')
                self.assertEqual(llm.calls[1][2],{'role':'assistant','content':SAY})
                self.assertNotIn('baseline:',llm.calls[1][2]['content'])

    def test_scope_validates_before_generation_and_state_changes(self):
        with tempfile.TemporaryDirectory() as td:
            llm=ScriptedLLM([]);engine=TurnEngine(self.cfg(),Store(td),llm)
            before=len(engine.store.read_ledger())
            with self.assertRaises(TypeError):engine.turn('你好',scope={'prefix':'x'})
            self.assertEqual(len(engine.store.read_ledger()),before)
            self.assertEqual(llm.calls,[])

    def test_session_stream_wrapper_forwards_scope(self):
        with tempfile.TemporaryDirectory() as td:
            with Session(self.cfg(),Store(td),ScriptedLLM([SAY]),
                options=SessionConfig(memory_mode='frozen',generation_path='direct')) as session:
                turn=session.say_stream('文案',scope=ResponseScope(prefix='草稿：'))
                self.assertEqual(''.join(turn),SAY)
                self.assertTrue(turn.result.scope_validation['valid'])

    def test_explicit_code_contract_frames_all_layouts_without_changing_code(self):
        body='```yaml\ndirect: false\npending: 2\n# 示例标记 <<<HEAD>>> <<<TAIL>>>\n```'
        for path in ('full','compact','direct'):
            for streaming in (False,True):
                with self.subTest(path=path,streaming=streaming),tempfile.TemporaryDirectory() as td:
                    raw=reply(path,body);llm=ScriptedLLM([raw]);streams=[];delivered=[]
                    def stream(messages):streams.append(messages);yield from raw
                    engine=TurnEngine(self.cfg(),Store(td),llm,generation_path=path,
                        stream_llm=stream if streaming else None)
                    result=engine.turn('教程代码',contract=ReplyContract.code_block(language='yaml'),
                        on_delta=delivered.append if streaming else None)
                    self.assertEqual(result.say,body)
                    self.assertTrue(result.contract_validation['valid'])
                    self.assertTrue(result.validity['valid'],result.validity)
                    self.assertEqual(len(llm.calls)+len(streams),1)
                    if streaming:self.assertEqual(''.join(delivered),body)
                    self.assertTrue(engine._probe['body_framing']['applied'])

    def test_code_contract_stream_interrupts_without_partial_delivery_or_retry(self):
        with tempfile.TemporaryDirectory() as td:
            llm=ScriptedLLM([]);calls=[];delivered=[];store=Store(td)
            def stream(messages):
                calls.append(messages);yield '```yaml\ndirect: false'
                raise RuntimeError('offline disconnect')
            engine=TurnEngine(self.cfg(),store,llm,generation_path='direct',stream_llm=stream)
            with self.assertRaises(RuntimeError):
                engine.turn('教程代码',contract=ReplyContract.code_block(language='yaml'),on_delta=delivered.append)
            self.assertEqual(delivered,[])
            self.assertEqual(len(calls),1)
            self.assertEqual(llm.calls,[])
            self.assertEqual(list(store.buffer_dir.glob('*.json')),[])

    def test_code_authority_is_not_inferred_from_a_model_fence(self):
        with tempfile.TemporaryDirectory() as td:
            engine=TurnEngine(self.cfg(),Store(td),ScriptedLLM(['```yaml\ndirect: false\n```']),generation_path='direct')
            result=engine.turn('普通输入')
            self.assertEqual(result.say,'')
            self.assertFalse(result.validity['valid'])

    def test_code_framing_rejects_unsupported_protocol_before_call(self):
        for protocol in ('auto','persona','tool'):
            with tempfile.TemporaryDirectory() as td:
                llm=ScriptedLLM([]);engine=TurnEngine(self.cfg(),Store(td),llm,protocol=protocol)
                with self.assertRaises(ValueError):
                    engine.turn('代码',contract=ReplyContract.code_block(language='yaml'))
                self.assertEqual(llm.calls,[])


if __name__=='__main__':unittest.main()
