"""Strict literal JSON containers survive parsing and actual delivery."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.parse import (SayText, parse, read_literal, safe_say, say_wire,
                           section_errors)
from persona.prompt import LITERAL_OPEN, LITERAL_CLOSE, build_history_entry
from persona.protocol import clean_phase2, StreamCleanPhase2
from persona.store import Store
from persona.stream import StreamParser


ROOT = Path(__file__).resolve().parent.parent
HEAD = ('need: 只是想说说话\nconfidence: .8\ndirect: PRIVATE_CONTENT\n'
        'baseline: 正常回应\ndisplace: none')
TAIL = 'settle: -\ndelta: -\npending: -'


def literal(body):
    return LITERAL_OPEN + body + LITERAL_CLOSE


def output(body):
    return make_reply(HEAD, literal(body), TAIL)


class LiteralContainerTests(unittest.TestCase):
    def streamed(self, raw, cut):
        parser = StreamParser()
        pieces = parser.feed(raw[:cut]) + parser.feed(raw[cut:])
        result = parser.finish()
        visible = ''.join(pieces)
        self.assertTrue(result.say.startswith(visible))
        return result, visible + result.say[len(visible):]

    def test_object_and_array_are_delivered_at_every_split(self):
        for body in ('{"箱号":"B2","数量":7,"状态":"待核对"}',
                     '[{"name":"样品","done":false},null,1.25,"空"]'):
            raw = output(body)
            self.assertEqual(parse(raw).say, body)
            for cut in range(len(raw)+1):
                with self.subTest(body=body, cut=cut):
                    parsed, delivered = self.streamed(raw, cut)
                    self.assertEqual(parsed.say, body)
                    self.assertEqual(delivered, body)
                    self.assertEqual(parsed.structural_errors, [])

    def test_container_layout_and_escape_spelling_are_not_reserialized(self):
        body = '{\n  "label" : "\\u7bb1",\n  "value": 1.00, "nested": [ 1, 2 ]\n}'
        self.assertEqual(read_literal(literal(body)), ('complete', len(literal(body)), body))
        self.assertEqual(parse(output(body)).say, body)
        for cut in range(len(output(body))+1):
            self.assertEqual(self.streamed(output(body), cut)[1], body)

    def test_nested_markers_and_reserved_fields_are_only_literal_data(self):
        body = '{\n "direct":"dictionary",\n "tail":"<<<TAIL>>>",\n "pending":{"claim":"a"},\n "close":"<<<END_LITERAL>>>"\n}'
        raw = output(body)
        self.assertEqual(section_errors(raw), [])
        self.assertEqual(parse(raw).say, body)
        self.assertIsNone(parse(raw).tail.pending)
        for cut in range(len(raw)+1):
            parsed, delivered = self.streamed(raw, cut)
            self.assertEqual(delivered, body)
            self.assertEqual(parsed.structural_errors, [])

    def test_duplicate_keys_and_nonfinite_numbers_never_deliver(self):
        invalid = ('{"k":1,"k":2}', '{"k":1,"\\u006b":2}',
                   '{"a":{"k":1,"k":2}}', '[{"k":1,"k":2}]',
                   '{"k":NaN}', '[Infinity]', '{"k":-Infinity}',
                   '{"k":1e999}', '[[-1e999]]')
        for body in invalid:
            raw = make_reply(HEAD, '安全前文。'+literal(body), TAIL)
            self.assertEqual(parse(raw).say, '安全前文。')
            self.assertFalse(parse(raw).tail_usable)
            for cut in range(len(raw)+1):
                with self.subTest(body=body, cut=cut):
                    parsed, delivered = self.streamed(raw, cut)
                    self.assertEqual(delivered, '安全前文。')
                    self.assertEqual(parsed.say, '安全前文。')

    def test_incomplete_bad_or_multiple_documents_do_not_fall_back(self):
        invalid = ('{"value":"SECRET"', '[1,"SECRET"', '{"value":"SECRET",}',
                   '{"value":"SECRET"} []', '[1] [2]',
                   '{"value":"SECRET"} garbage')
        for body in invalid:
            self.assertEqual(parse(output(body)).say, '')
            for cut in range(len(output(body))+1):
                self.assertEqual(self.streamed(output(body), cut)[1], '')
        for body in ('{"value":"SECRET"}', '[1,2]'):
            self.assertEqual(safe_say(LITERAL_OPEN+body), '')

    def test_lone_surrogates_are_rejected_in_nested_values_and_keys(self):
        for body in ('{"bad":"\\ud800"}', '{"\\udfff":"x"}', '[{"a":["\\ud800"]}]'):
            self.assertEqual(read_literal(literal(body))[0], 'invalid')
            self.assertEqual(parse(output(body)).say, '')
        body = '{"emoji":"\\ud83d\\ude00"}'
        self.assertEqual(parse(output(body)).say, body)

    def test_noncontainer_scalars_remain_invalid(self):
        for body in ('123', '1.5', 'true', 'false', 'null'):
            self.assertEqual(read_literal(literal(body))[0], 'invalid')
            self.assertEqual(parse(output(body)).say, '')
        self.assertEqual(parse(output('"字符串仍按原协议解码"')).say, '字符串仍按原协议解码')

    def test_framed_container_waits_until_complete_closing_marker(self):
        parser = StreamParser()
        self.assertEqual(parser.feed('<<<HEAD>>>\n'+HEAD+'\n<<<SAY>>>'), [])
        body = '{"箱号":"B2","数量":7}'
        self.assertEqual(parser.feed(LITERAL_OPEN+body), [])
        self.assertEqual(parser.feed(LITERAL_CLOSE[:-1]), [])
        self.assertEqual(''.join(parser.feed(LITERAL_CLOSE[-1:])), body)
        self.assertEqual(parser.finish().say, body)

    def test_interruption_keeps_complete_container_and_drops_partial_next_one(self):
        body = '{"direct":"词典释义"}'
        parser = StreamParser()
        pieces = parser.feed('<<<HEAD>>>\n'+HEAD+'\n<<<SAY>>>'+literal(body))
        pieces += parser.feed(LITERAL_OPEN+'{"a":"SECRET"')
        result = parser.finish(interrupted=True)
        self.assertEqual(result.say, ''.join(pieces))
        self.assertEqual(result.say, body)
        self.assertEqual(safe_say(result.say), body)
        self.assertIsNone(result.tail)

    def test_special_tokens_and_unframed_residue_remain_protected(self):
        body = '{"value":"左<|endoftext|>右"}'
        self.assertEqual(parse(output(body)).say, '{"value":"左右"}')
        for cut in range(len(output(body))+1):
            self.assertEqual(self.streamed(output(body), cut)[1], '{"value":"左右"}')
        broken = '{"value":"SECRET<|unfinished"}'
        self.assertEqual(parse(output(broken)).say, '')
        raw = make_reply(HEAD, literal('{"x":1}')+'\ndirect: SECRET', TAIL)
        self.assertEqual(parse(raw).say, '')

    def test_provenance_recheck_history_and_phase_two_share_contract(self):
        body = '{\n "direct": "字典", "example": "<<<HEAD>>>"\n}'
        parsed = parse(output(body))
        self.assertIsInstance(parsed.say, SayText)
        for value in (parsed.say, deepcopy(parsed.say), pickle.loads(pickle.dumps(parsed.say))):
            self.assertEqual(safe_say(value), body)
            self.assertEqual(safe_say(say_wire(value)), body)
        self.assertEqual(safe_say(SayText('tampered', say_wire(parsed.say))), '')
        history = build_history_entry(parsed.say, parsed.head, parsed.tail)
        self.assertEqual(parse(history).say, body)
        self.assertEqual(clean_phase2(literal(body)), body)
        for cut in range(len(literal(body))+1):
            cleaner = StreamCleanPhase2()
            pieces = cleaner.feed(literal(body)[:cut])+cleaner.feed(literal(body)[cut:])+cleaner.finish()
            self.assertEqual(''.join(pieces), body)

    def test_bare_literal_marker_payload_is_opaque_before_closing(self):
        bodies = ('{"example":"<<<HEAD>>>","direct":"word"}',
                  json.dumps('<<<HEAD>>>\ndirect: 只是示例', ensure_ascii=False))
        for body in bodies:
            expected = body if body.startswith('{') else json.loads(body)
            for cut in range(len(literal(body))+1):
                with self.subTest(body=body, cut=cut):
                    cleaner = StreamCleanPhase2()
                    pieces = cleaner.feed(literal(body)[:cut])+cleaner.feed(literal(body)[cut:])+cleaner.finish()
                    self.assertEqual(''.join(pieces), expected)
        for body in ('<<<literal>>>"SECRET <<<HEAD>>>"<<<END_LITERAL>>>',
                     '<<<LIT malformed <<<HEAD>>>SECRET'):
            for cut in range(len(body)+1):
                cleaner = StreamCleanPhase2()
                pieces = cleaner.feed(body[:cut])+cleaner.feed(body[cut:])+cleaner.finish()
                self.assertEqual(''.join(pieces), '')

    def test_single_call_engine_delivers_recorded_object_instead_of_empty_reply(self):
        body = '{"箱号":"B2","数量":7,"状态":"待核对"}'
        raw = output(body)
        for chunk_size in (None,1,5,len(raw)):
            with tempfile.TemporaryDirectory() as td:
                cfg = PersonaConfig.load(ROOT/'personas/default.toml')
                llm = ScriptedLLM([raw] if chunk_size is None else [])
                stream = None if chunk_size is None else lambda messages: iter(
                    raw[i:i+chunk_size] for i in range(0,len(raw),chunk_size))
                delivered=[]
                engine=TurnEngine(cfg,Store(td),llm,stream_llm=stream)
                result=engine.turn('只返回箱单JSON。',on_delta=delivered.append)
                self.assertEqual(result.say,body)
                self.assertEqual(''.join(delivered),body)
                self.assertTrue(result.validity['valid'])
                self.assertEqual(len(llm.calls),1 if chunk_size is None else 0)
                row=[r for r in engine.store.read_ledger() if r['kind']=='turn'][-1]
                self.assertEqual(row['say'],body)
                self.assertEqual(safe_say(row['say_wire']),body)

    def test_explicit_plain_mode_keeps_normal_and_literal_content(self):
        body='普通回复。'+literal('{"example":"<<<HEAD>>>","direct":"词典"}')
        expected='普通回复。{"example":"<<<HEAD>>>","direct":"词典"}'
        for cut in range(len(body)+1):
            parser=StreamParser(plain=True)
            pieces=parser.feed(body[:cut])+parser.feed(body[cut:])
            result=parser.finish()
            visible=''.join(pieces)
            self.assertEqual(visible+result.say[len(visible):],expected)
            self.assertEqual(result.say,expected)
            self.assertIsNone(result.head)
            self.assertIsNone(result.tail)

    def test_explicit_plain_mode_never_extracts_unexpected_protocol(self):
        for body in ('direct: PRIVATE\n<<<SAY>>>仍不能交付',
                     'need: listen\n<<<SAY>>>仍不能交付',
                     '<<<HEAD>>>\ndirect: PRIVATE\n<<<SAY>>>仍不能交付',
                     '<<<SAY>>>仍不能交付', '<<<TAIL>>>\npending: PRIVATE'):
            for cut in range(len(body)+1):
                parser=StreamParser(plain=True)
                pieces=parser.feed(body[:cut])+parser.feed(body[cut:])
                result=parser.finish()
                self.assertEqual(''.join(pieces),'')
                self.assertEqual(result.say,'')
                self.assertTrue(parser.output_blocked)
                self.assertIsNone(result.head)

    def test_plain_mode_keeps_delivered_prefix_and_truncates_residue(self):
        for suffix in ('\ndirect: PRIVATE', '<<<HEAD>>>PRIVATE<<<SAY>>>不恢复',
                       '<<<TAIL>>>PRIVATE'):
            body='已经说出的正文。'+suffix
            for cut in range(len(body)+1):
                parser=StreamParser(plain=True)
                pieces=parser.feed(body[:cut])+parser.feed(body[cut:])
                result=parser.finish()
                self.assertEqual(''.join(pieces),'已经说出的正文。')
                self.assertEqual(result.say,'已经说出的正文。')
                self.assertTrue(parser.output_blocked)
        with self.assertRaises(TypeError):
            StreamParser(plain='yes')


if __name__=='__main__':unittest.main()
