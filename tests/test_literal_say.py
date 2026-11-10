"""Explicit literal framing preserves content without trusting raw SAY blocks."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.parse import (Head, SayText, encode_literal_say, parse, plain_text,
                           safe_say, say_wire, section_errors)
from persona.prompt import build_history_entry, LITERAL_OPEN, LITERAL_CLOSE
from persona.protocol import clean_phase2, StreamCleanPhase2, to_parsed
from persona.store import Store
from persona.stream import StreamParser


ROOT = Path(__file__).resolve().parent.parent
HEAD = ('need: listen\nconfidence: .6\ndirect: PRIVATE_HEAD_ONLY\n'
        'baseline: 常态回应\ndisplace: none')
TAIL = 'settle: -\npending: -'


def wrapped(body):
    return make_reply(HEAD, body, TAIL)


class TestLiteralSay(unittest.TestCase):
    def stream(self, raw, cut):
        parser = StreamParser()
        pieces = parser.feed(raw[:cut]) + parser.feed(raw[cut:])
        result = parser.finish()
        visible = ''.join(pieces)
        self.assertTrue(result.say.startswith(visible), (cut, visible, result.say))
        pieces.append(result.say[len(visible):])
        return result, ''.join(pieces)

    def test_reserved_content_is_expressible_at_every_two_chunk_boundary(self):
        examples = ('（判断题）这个说法正确。',
                    'direct: English adjective meaning immediate',
                    '```yaml\ndirect: true\npending: false\n```',
                    '故事里她写下：（内心：终于回来了。）',
                    '（判断依据）见下文。')
        for text in examples:
            raw = wrapped(encode_literal_say(text))
            expected = parse(raw)
            self.assertEqual(expected.say, text)
            self.assertEqual(expected.structural_errors, [])
            for cut in range(len(raw) + 1):
                with self.subTest(text=text, cut=cut):
                    result, delivered = self.stream(raw, cut)
                    self.assertEqual(delivered, text)
                    self.assertEqual(result.say, text)
                    self.assertNotIn('PRIVATE_HEAD_ONLY', delivered)

    def test_raw_marker_characters_inside_json_do_not_end_literal(self):
        text = 'Example: <<<HEAD>>>\ndirect: literal field\n<<<TAIL>>>'
        wire = LITERAL_OPEN + json.dumps(text, ensure_ascii=False) + LITERAL_CLOSE
        raw = wrapped(wire)
        self.assertEqual(section_errors(raw), [])
        self.assertEqual(parse(raw).say, text)
        for cut in range(len(raw) + 1):
            result, delivered = self.stream(raw, cut)
            self.assertEqual(delivered, text)
            self.assertEqual(result.structural_errors, [])

    def test_json_quoted_closing_delimiter_is_data_not_termination(self):
        text = '字面 <<<END_LITERAL>>> 仍在字符串里，后面才结束。'
        raw = wrapped(LITERAL_OPEN + json.dumps(text, ensure_ascii=False) + LITERAL_CLOSE)
        for cut in range(len(raw) + 1):
            self.assertEqual(self.stream(raw, cut)[1], text)
        self.assertEqual(parse(raw).say, text)

    def test_mixed_normal_and_multiple_literal_segments(self):
        wire = '开头。' + encode_literal_say('direct: 字典解释') + '\n接着：' + encode_literal_say('（内心：故事原文）') + '结尾。'
        expected = '开头。direct: 字典解释\n接着：（内心：故事原文）结尾。'
        raw = wrapped(wire)
        for size in (1, 2, 3, 11, len(raw)):
            sp, pieces = StreamParser(), []
            for index in range(0, len(raw), size):
                pieces.extend(sp.feed(raw[index:index+size]))
            result = sp.finish()
            visible = ''.join(pieces)
            pieces.append(result.say[len(visible):])
            self.assertEqual(''.join(pieces), expected)
        self.assertEqual(parse(raw).say, expected)

    def test_literal_whitespace_and_escapes_are_preserved(self):
        text = '  \n（判断题）"引号"与\\反斜杠\t\n尾部空白  \n'
        raw = wrapped(encode_literal_say(text))
        for cut in range(len(raw) + 1):
            result, delivered = self.stream(raw, cut)
            self.assertEqual(delivered, text)
            self.assertEqual(result.say, text)
        self.assertEqual(parse(raw).say, text)

    def test_normal_prefix_is_incremental_but_literal_waits_for_close(self):
        sp = StreamParser()
        prefix = f'<<<HEAD>>>\n{HEAD}\n<<<SAY>>>普通正文。'
        self.assertEqual(''.join(sp.feed(prefix)), '普通正文。')
        self.assertEqual(sp.feed(LITERAL_OPEN + '"direct: 字典解释"'), [])
        self.assertEqual(''.join(sp.feed(LITERAL_CLOSE)), 'direct: 字典解释')
        self.assertEqual(sp.finish().say, '普通正文。direct: 字典解释')

    def test_unfinished_or_invalid_literal_does_not_fall_back_to_content(self):
        invalid = (LITERAL_OPEN + '"SENSITIVE_BODY',
                   LITERAL_OPEN + '"SENSITIVE_BODY"',
                   LITERAL_OPEN + '{"text":"SENSITIVE_BODY"}' + LITERAL_CLOSE,
                   LITERAL_OPEN + '123' + LITERAL_CLOSE,
                   LITERAL_OPEN + '"bad\\qSENSITIVE_BODY"' + LITERAL_CLOSE,
                   LITERAL_OPEN + '"\\ud800"' + LITERAL_CLOSE,
                   '<<<literal>>>"SENSITIVE_BODY"<<<END_LITERAL>>>',
                   '<<LITERAL>>"SENSITIVE_BODY"<<END_LITERAL>>',
                   '<<<LIT', '<<<LIT malformed', LITERAL_CLOSE)
        for body in invalid:
            raw = wrapped('安全前文。' + body)
            expected = parse(raw)
            self.assertEqual(expected.say, '安全前文。')
            self.assertFalse(expected.tail_usable)
            for cut in range(len(raw) + 1):
                with self.subTest(body=body, cut=cut):
                    result, delivered = self.stream(raw, cut)
                    self.assertEqual(delivered, '安全前文。')
                    self.assertEqual(result.say, '安全前文。')
                    self.assertNotIn('SENSITIVE_BODY', delivered)

    def test_interruption_drops_only_unclosed_literal(self):
        sp = StreamParser()
        prefix = f'<<<HEAD>>>\n{HEAD}\n<<<SAY>>>先说：' + encode_literal_say('direct: 字典解释')
        delivered = sp.feed(prefix)
        delivered += sp.feed('然后' + LITERAL_OPEN + '"SENSITIVE_BODY')
        result = sp.finish(interrupted=True)
        self.assertEqual(''.join(delivered), '先说：direct: 字典解释然后')
        self.assertEqual(result.say, ''.join(delivered))
        self.assertEqual(safe_say(result.say), result.say)
        self.assertIsNone(result.tail)

    def test_ordinary_residue_protection_is_unchanged(self):
        for text in ('direct: PRIVATE_HEAD_ONLY', '（内心：PRIVATE_HEAD_ONLY）',
                     '（判断题）PRIVATE_HEAD_ONLY', '<<<head>>PRIVATE_HEAD_ONLY'):
            self.assertEqual(safe_say(text), '')
            self.assertEqual(parse(wrapped(text)).say, '')

    def test_model_special_tokens_cannot_be_smuggled_through_literal(self):
        text = '左边<|endoftext|>右边<｜end▁of▁sentence｜>'
        raw = wrapped(encode_literal_say(text))
        self.assertEqual(parse(raw).say, '左边右边')
        for cut in range(len(raw) + 1):
            self.assertEqual(self.stream(raw, cut)[1], '左边右边')
        broken = wrapped(encode_literal_say('SENSITIVE_BODY<|unfinished'))
        self.assertEqual(parse(broken).say, '')
        for cut in range(len(broken) + 1):
            self.assertEqual(self.stream(broken, cut)[1], '')

    def test_literal_does_not_exempt_subsequent_normal_segments_from_guards(self):
        prefix = encode_literal_say('direct: 字典解释')
        invalid = ('<<<literal>>>SENSITIVE_BODY', '<<<LIT malformed',
                   '<<LITERAL>>SENSITIVE_BODY', LITERAL_CLOSE,
                   '<|unfinished', '<|endoftext|>SENSITIVE_BODY',
                   '<end_of_turn>SENSITIVE_BODY')
        for suffix in invalid:
            for body, expected in ((prefix + suffix, 'direct: 字典解释'),
                                   ('安全前文。' + suffix + prefix, '安全前文。')):
                raw = wrapped(body)
                self.assertEqual(parse(raw).say, expected)
                for cut in range(len(raw) + 1):
                    with self.subTest(suffix=suffix, cut=cut):
                        result, delivered = self.stream(raw, cut)
                        self.assertEqual(delivered, expected)
                        self.assertEqual(result.say, expected)
                        self.assertNotIn('SENSITIVE_BODY', delivered)

    def test_carrier_requires_compatible_recheckable_wire(self):
        text = 'direct: 字典解释'
        valid = safe_say(encode_literal_say(text))
        self.assertIsInstance(valid, SayText)
        self.assertEqual(safe_say(valid), text)
        self.assertEqual(safe_say(SayText(text, text)), '')
        self.assertEqual(safe_say(SayText('tampered', say_wire(valid))), '')
        self.assertEqual(safe_say(SayText(text, None)), '')
        self.assertEqual(safe_say(SayText(text, 123)), '')
        cyclic = SayText(text)
        cyclic.wire = cyclic
        self.assertEqual(safe_say(cyclic), '')

    def test_deepcopy_pickle_and_json_wire_roundtrip(self):
        text = 'direct: 字典解释\n（内心：小说）'
        value = safe_say(encode_literal_say(text))
        for restored in (deepcopy(value), pickle.loads(pickle.dumps(value))):
            self.assertEqual(say_wire(restored), say_wire(value))
            self.assertEqual(safe_say(restored), text)
        record = json.loads(json.dumps(dict(say=value, say_wire=say_wire(value)), ensure_ascii=False))
        self.assertIs(type(record['say']), str)
        self.assertEqual(safe_say(record['say_wire']), record['say'])

    def test_history_keeps_wire_framing(self):
        text = '（判断题）\n<<<TAIL>>>\ndirect: 示例'
        first = parse(wrapped(encode_literal_say(text)))
        history = build_history_entry(first.say, first.head, first.tail)
        self.assertIn(LITERAL_OPEN, history)
        self.assertEqual(parse(history).say, text)
        self.assertEqual(section_errors(history), [])

    def test_phase2_cleaning_and_conversion_preserve_provenance(self):
        text = '```yaml\ndirect: true\n```'
        value = clean_phase2(encode_literal_say(text))
        self.assertEqual(value, text)
        self.assertEqual(to_parsed(Head(), value).say, text)
        self.assertEqual(clean_phase2(value), text)
        self.assertEqual(clean_phase2(wrapped(encode_literal_say(text))), text)
        self.assertEqual(plain_text(encode_literal_say(text)), text)

    def test_legacy_stream_cleaner_uses_same_literal_contract(self):
        text = 'direct: 字典解释\n（判断题）'
        raw = encode_literal_say(text)
        for cut in range(len(raw) + 1):
            cleaner = StreamCleanPhase2()
            pieces = cleaner.feed(raw[:cut]) + cleaner.feed(raw[cut:]) + cleaner.finish()
            self.assertEqual(''.join(pieces), text)
            self.assertEqual(safe_say(cleaner.text), text)

    def test_single_call_engine_and_callback_record_actual_decoded_text(self):
        text = '故事里她写下：（内心：终于回来了。）\ndirect: word'
        raw = wrapped(encode_literal_say(text))
        with tempfile.TemporaryDirectory() as folder:
            llm = ScriptedLLM([raw])
            store = Store(folder)
            cfg = PersonaConfig.load(ROOT / 'personas/default.toml')
            result = TurnEngine(cfg, store, llm).turn('请引用这段故事和词典字段。')
            self.assertEqual(result.say, text)
            self.assertEqual(len(llm.calls), 1)
        for size in (1, 5, len(raw)):
            with tempfile.TemporaryDirectory() as folder:
                llm = ScriptedLLM([])
                store = Store(folder)
                seen = []
                cfg = PersonaConfig.load(ROOT / 'personas/default.toml')
                engine = TurnEngine(cfg, store, llm, stream_llm=lambda messages: iter(
                    raw[i:i+size] for i in range(0, len(raw), size)))
                result = engine.turn('请引用这段故事和词典字段。', on_delta=seen.append)
                self.assertEqual(result.say, text)
                self.assertEqual(''.join(seen), text)
                self.assertEqual(llm.calls, [])
                turn = [r for r in store.read_ledger() if r['kind'] == 'turn'][-1]
                self.assertEqual(turn['say'], text)


if __name__ == '__main__':
    unittest.main()
