"""Reference-field marker data and unrequested END have separate authority."""
from pathlib import Path
import tempfile
import unittest

from persona import ReplyContract
from persona.body import prepare_code_body
from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.generation import adapt, parse_output, validity
from persona.llm import ScriptedLLM, make_reply
from persona.parse import encode_literal_say, parse_head, safe_say, section_errors
from persona.store import Store
from persona.stream import StreamParser


ROOT = Path(__file__).resolve().parent.parent
HEAD = ('need: 用户请求教程\nconfidence: .9\n'
        'direct: markers: ["<<<HEAD>>>", "<<<TAIL>>>", "<<<SAY>>>"]\n'
        'baseline: 展示 <<<HEAD>>> 和 <<<TAIL>>> 字面数据\ndisplace: none')
COMPACT = ('need: 用户请求教程\nconfidence: .9\n'
           'direct: 展示 ["<<<HEAD>>>", "<<<TAIL>>>", "<<<SAY>>>"]')
TAIL = 'settle: -\ndelta: -\npending: -'
CODE = '```yaml\nscene:\n  direct: north\n  pending: []\nmarkers: ["<<<HEAD>>>", "<<<TAIL>>>"]\n```'


def envelope(body, path='full', head=None, tail=TAIL):
    if path == 'direct':
        return body
    return make_reply(head or (HEAD if path == 'full' else COMPACT), body,
                      tail if path == 'full' else '')


def stream(raw, path, chunks):
    parser = StreamParser(expression_only=path == 'compact', plain=path == 'direct')
    pieces = []
    for part in chunks:
        pieces.extend(parser.feed(part))
    result = adapt(parser.finish(), path)
    return ''.join(pieces), result


class MarkerBoundaryTests(unittest.TestCase):
    def test_head_reference_values_are_preserved_without_structural_markers(self):
        for path in ('full', 'compact'):
            raw = envelope('教程在这里。', path)
            parsed = parse_output(raw, path)
            self.assertEqual(parsed.say, '教程在这里。')
            self.assertIn('"<<<SAY>>>"', parsed.head.direct)
            self.assertIn('<<<HEAD>>>', parsed.head.direct)
            self.assertIn('<<<TAIL>>>', parsed.head.direct)
            self.assertTrue(validity(parsed, path)['valid'], parsed)
        self.assertIn('<<<TAIL>>>', parse_head(HEAD).direct)

    def test_reference_markers_match_streaming_at_every_split(self):
        for path in ('full', 'compact'):
            raw = envelope('正文只有这一句。', path)
            expected = parse_output(raw, path)
            for split in range(len(raw)+1):
                with self.subTest(path=path, split=split):
                    shown, parsed = stream(raw, path, [raw[:split], raw[split:]])
                    self.assertEqual(shown, expected.say)
                    self.assertEqual(parsed.say, expected.say)
                    self.assertEqual(parsed.head.direct, expected.head.direct)
                    self.assertTrue(validity(parsed, path)['valid'], parsed)
            shown, parsed = stream(raw, path, list(raw))
            self.assertEqual(shown, expected.say)
            self.assertEqual(parsed.head.direct, expected.head.direct)

    def test_legacy_unquoted_inline_say_retains_its_boundary(self):
        # A bare inline SAY is ambiguous; it keeps its historical separator
        # meaning. Quoted SAY in the preceding tests is reference-field data.
        for path, head in (('full', HEAD), ('compact', 'need: n\nconfidence: .8\ndirect: d'),
                           ('compact', "need: n\nconfidence: .8\ndirect: don't claim more")):
            raw = '<<<HEAD>>>'+head+'<<<SAY>>>正文。'+('<<<TAIL>>>'+TAIL if path == 'full' else '')
            parsed = parse_output(raw, path)
            self.assertEqual(parsed.say, '正文。')
            self.assertTrue(validity(parsed, path)['valid'])
            self.assertEqual(stream(raw, path, list(raw))[0], '正文。')

    def test_real_independent_duplicate_and_wrong_order_remain_invalid(self):
        for raw in (envelope('正文。').replace('<<<SAY>>>', '<<<HEAD>>>\n'+HEAD+'\n<<<SAY>>>'),
                    '<<<SAY>>>正文。\n<<<HEAD>>>\n'+HEAD+'\n<<<TAIL>>>\n'+TAIL):
            self.assertTrue(section_errors(raw))
            self.assertFalse(parse_output(raw, 'full').tail_usable)
        raw = envelope('第一句。\n<<<SAY>>>第二句。')
        self.assertEqual(parse_output(raw, 'full').say, '第一句。')
        self.assertEqual(stream(raw, 'full', list(raw))[0], '第一句。')

    def test_field_interruption_cannot_reauthorize_a_second_say(self):
        raw = '<<<SAY>>>第一句。\ndirect: PRIVATE\n<<<SAY>>>第二份。'
        for split in range(len(raw)+1):
            shown, parsed = stream(raw, 'compact', [raw[:split], raw[split:]])
            self.assertEqual(shown, '第一句。')
            self.assertEqual(parsed.say, '第一句。')
            self.assertFalse(validity(parsed, 'compact')['valid'])
        self.assertEqual(stream(raw, 'compact', list(raw))[0], '第一句。')

    def test_complete_code_with_quoted_head_markers_has_normal_state_authority(self):
        with tempfile.TemporaryDirectory() as folder:
            llm = ScriptedLLM([envelope(CODE, tail='settle: -\ndelta:\n  closeness: .03\npending: -')])
            engine = TurnEngine(PersonaConfig.load(ROOT/'personas/default.toml'), Store(folder), llm)
            result = engine.turn('只给教程代码。', contract=ReplyContract.code_block(language='yaml'))
            self.assertEqual(result.say, CODE)
            self.assertTrue(result.validity['valid'], result.validity)
            self.assertEqual(result.relation_changed, {'closeness': .03})
            self.assertEqual(len(llm.calls), 1)

    def test_unrequested_end_preserves_prefix_and_is_invalid_all_paths(self):
        for path in ('full', 'compact', 'direct'):
            raw = envelope('完整正文。\n<<<END>>>\n不应继续交付。', path)
            parsed = parse_output(raw, path)
            self.assertEqual(parsed.say, '完整正文。')
            self.assertIn('unexpected_control_marker', validity(parsed, path)['reasons'])
            self.assertFalse(validity(parsed, path)['valid'])

    def test_end_cannot_escape_at_any_stream_split_or_reopen_say(self):
        for path in ('full', 'compact', 'direct'):
            raw = envelope('第一句。\n<<<END>>>\n<<<SAY>>>第二份。', path)
            for split in range(len(raw)+1):
                with self.subTest(path=path, split=split):
                    shown, parsed = stream(raw, path, [raw[:split], raw[split:]])
                    self.assertEqual(shown, '第一句。')
                    self.assertEqual(parsed.say, shown)
                    self.assertFalse(validity(parsed, path)['valid'])
            self.assertEqual(stream(raw, path, list(raw))[0], '第一句。')

    def test_malformed_or_unfinished_end_is_not_washed_as_valid(self):
        for marker in ('<<<end>>>', '<< END >>', '<<<END', '<<<end>\n'):
            for path in ('full', 'compact', 'direct'):
                raw = envelope('正文。'+marker, path)
                # A partial control followed by a real TAIL is still not
                # visible, even though the surrounding envelope is complete.
                parsed = parse_output(raw, path)
                self.assertEqual(parsed.say, '正文。')
                shown, streamed = stream(raw, path, list(raw))
                self.assertEqual(shown, parsed.say)
                if path == 'direct' or marker.endswith('>>>') or marker == '<< END >>':
                    self.assertFalse(validity(parsed, path)['valid'])
                    self.assertFalse(validity(streamed, path)['valid'])

    def test_requested_literal_and_code_end_are_preserved(self):
        literal = '字面标记：<<<END>>>；<<<HEAD>>>；<<<SAY>>>；<<<TAIL>>>'
        for path in ('full', 'compact', 'direct'):
            raw = envelope(encode_literal_say(literal), path)
            self.assertEqual(parse_output(raw, path).say, literal)
            self.assertTrue(validity(parse_output(raw, path), path)['valid'])
            shown, parsed = stream(raw, path, list(raw))
            self.assertEqual(shown, literal)
            self.assertTrue(validity(parsed, path)['valid'])
            code = '```yaml\nmarker: "<<<END>>>"\n```'
            framed, errors = prepare_code_body(envelope(code, path), path, ReplyContract.code_block(language='yaml'))
            parsed = parse_output(framed, path)
            parsed.structural_errors.extend(errors)
            self.assertEqual(parsed.say, code)
            self.assertTrue(validity(parsed, path)['valid'])

    def test_end_rejects_same_tail_state_proposals(self):
        for streaming in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                raw = envelope('可见正文。<<<END>>>', tail='settle: -\ndelta:\n  closeness: .05\npending: -')
                llm = ScriptedLLM([raw])
                calls = []
                def chunks(messages):
                    calls.append(messages)
                    yield from raw
                engine = TurnEngine(PersonaConfig.load(ROOT/'personas/default.toml'), Store(folder), llm,
                                    stream_llm=chunks if streaming else None)
                shown = []
                result = engine.turn('正常回复。', on_delta=shown.append if streaming else None)
                self.assertEqual(result.say, '可见正文。')
                self.assertEqual(result.relation_changed, {})
                self.assertIsNone(result.pending_added)
                self.assertFalse(result.validity['valid'])
                self.assertEqual(len(llm.calls)+len(calls), 1)
                if streaming:
                    self.assertEqual(''.join(shown), result.say)

    def test_unrelated_angles_and_longer_words_are_not_protocol(self):
        for text in ('<END>', '<<<ENDLESS>>>', '<300> 和 <苏州> 都是正文。'):
            self.assertEqual(safe_say(text), text)
            self.assertEqual(parse_output(text, 'direct').say, text)
            self.assertEqual(stream(text, 'direct', list(text))[0], text)

    def test_bare_incomplete_end_cannot_reenter_through_plain_text_fallback(self):
        for raw in ('<<<E', '<<<END', '<<<end>\n'):
            for path in ('full', 'compact', 'direct'):
                with tempfile.TemporaryDirectory() as folder:
                    engine = TurnEngine(PersonaConfig.load(ROOT/'personas/default.toml'),
                                        Store(folder), ScriptedLLM([raw]), generation_path=path)
                    shown = []
                    result = engine.turn('请正常回复。', on_delta=shown.append)
                    self.assertEqual(result.say, '')
                    self.assertEqual(shown, [])
                    self.assertFalse(result.validity['valid'])


if __name__ == '__main__':
    unittest.main()
