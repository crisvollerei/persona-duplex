"""Literal SAY provenance survives delivery, interruption, and restart."""
from pathlib import Path
import tempfile
import unittest

from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.parse import encode_literal_say, safe_say, say_wire

ROOT = Path(__file__).resolve().parent.parent
from persona.store import Store

HEAD = 'need: 只是想说说话\nconfidence: 0.8\ndirect: PRIVATE_HEAD_VALUE\nbaseline: 正常回答\ndisplace: none'
TAIL = 'settle: -\ndelta: -\npending: -'
BODY = '（判断题）这个说法正确。\ndirect: English adjective meaning immediate\n```yaml\ndirect: true\n```\n故事里她写下：（内心：终于回来了。）'


class LiteralDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def cfg(self):
        return PersonaConfig.load(ROOT/'personas/default.toml')

    def test_text_all_chunk_sizes_preserve_body_and_auditable_wire(self):
        raw = make_reply(HEAD, encode_literal_say(BODY), TAIL)
        for size in (1, 2, 7, len(raw)):
            with self.subTest(size=size):
                store = Store(Path(self.temp.name)/str(size))
                seen = []
                engine = TurnEngine(self.cfg(), store, ScriptedLLM([]),
                    stream_llm=lambda messages: (raw[i:i+size] for i in range(0,len(raw),size)))
                result = engine.turn('请给出字面示例', on_delta=seen.append)
                self.assertEqual(result.say, BODY)
                self.assertEqual(''.join(seen), BODY)
                self.assertNotIn('PRIVATE_HEAD_VALUE', result.say)
                self.assertTrue(result.validity['valid'], result.validity)
                saved = [r for r in store.read_ledger() if r['kind']=='turn'][-1]
                self.assertEqual(saved['say'], BODY)
                self.assertEqual(safe_say(saved['say_wire']), BODY)
                reopened = TurnEngine(self.cfg(), Store(store.root), ScriptedLLM([]))
                history = reopened.history[-1]['content']
                self.assertIn('<<<LITERAL>>>', history)
                self.assertIn('English adjective', history)

    def test_text_interruption_preserves_only_delivered_complete_literals(self):
        raw = make_reply(HEAD, encode_literal_say(BODY), TAIL).split('<<<TAIL>>>')[0]
        def interrupted(messages):
            yield raw
            yield '<<<LITERAL>>>"unfinished'
            raise RuntimeError('disconnect')
        store = Store(Path(self.temp.name)/'text-interrupted')
        seen = []
        result = TurnEngine(self.cfg(), store, ScriptedLLM([]), stream_llm=interrupted).turn('例子', on_delta=seen.append)
        self.assertEqual(result.say, BODY)
        self.assertEqual(''.join(seen), BODY)
        self.assertEqual(safe_say(result.say), BODY)
        self.assertIsNone(result.parsed.tail)
        self.assertEqual(list(store.buffer_dir.glob('*.json')), [])

    def test_tool_and_persona_interruption_do_not_strip_literal_provenance(self):
        for mode in ('tool', 'persona'):
            with self.subTest(mode=mode):
                calls = []
                def tool(messages, *, tool, **kwargs):
                    name = tool['function']['name']; calls.append(name)
                    if name == 'raw_reply':
                        return dict(need='引用示例', confidence=.8, raw='这是用户要求的字面示例', constraints=[])
                    if name == 'inner':
                        return dict(need='只是想说说话', confidence=.8, direct='PRIVATE_HEAD_VALUE', baseline='正常回答', displace='none')
                    raise AssertionError('No observation call after interruption')
                wire = encode_literal_say(BODY)
                expression = make_reply('baseline: 正常回答\ndisplace: none', wire) if mode=='persona' else wire
                def broken(messages):
                    yield expression
                    raise RuntimeError('disconnect')
                store = Store(Path(self.temp.name)/mode)
                seen = []
                result = TurnEngine(self.cfg(), store, ScriptedLLM([]), tool_llm=tool,
                                    stream_llm=broken, protocol=mode).turn('给出字面示例', on_delta=seen.append)
                self.assertEqual(result.say, BODY)
                self.assertEqual(''.join(seen), BODY)
                self.assertEqual(safe_say(result.say), BODY)
                self.assertIsNone(result.parsed.tail)
                self.assertEqual(len(calls), 1)
                self.assertEqual(list(store.buffer_dir.glob('*.json')), [])

    def test_marker_examples_do_not_become_state(self):
        body = '<<<HEAD>>>\ndirect: 字段示例\n<<<TAIL>>>\npending: 这只是代码数据'
        store = Store(Path(self.temp.name)/'markers')
        result = TurnEngine(self.cfg(), store, ScriptedLLM([make_reply(HEAD, encode_literal_say(body), TAIL)])).turn('解释协议')
        self.assertEqual(result.say, body)
        self.assertEqual(store.pendings(), [])
        self.assertTrue(result.validity['valid'], result.validity)


if __name__ == '__main__':
    unittest.main()
