"""Caller-authorized code fences remain data without relaxing other replies."""
from dataclasses import asdict
import unittest

from persona.body import code_block_span, frame_code_body, prepare_code_body
from persona.contracts import ReplyContract
from persona.generation import parse_output
from persona.llm import make_reply
from persona.parse import safe_say, say_wire, section_errors


HEAD = ('need: 用户请求代码\nconfidence: .9\ndirect: PRIVATE_REFERENCE\n'
        'baseline: 按格式提供代码\ndisplace: none')
COMPACT = 'need: 用户请求代码\nconfidence: .9\ndirect: PRIVATE_REFERENCE'
TAIL = 'settle: -\ndelta: -\npending: -'
CODE = '```yaml\ndirect: false\npending: 2\n```'


def envelope(code, path):
    if path == 'direct':return code
    return make_reply(HEAD if path=='full' else COMPACT, code, TAIL if path=='full' else '')


class CodeBodyContractTests(unittest.TestCase):
    def setUp(self):
        self.contract=ReplyContract.code_block(language='yaml')

    def test_contract_roundtrip_and_explicit_prompt(self):
        self.assertEqual(ReplyContract(**asdict(self.contract)),self.contract)
        self.assertEqual(self.contract.kind,'code_block')
        self.assertEqual(self.contract.code_language,'yaml')
        self.assertIn('yaml',self.contract.prompt())
        self.assertTrue(self.contract.validate(CODE)['valid'])
        self.assertEqual(self.contract.validate(CODE)['scope'],'mechanical_format_not_semantic_truth')

    def test_only_supported_explicit_language_is_accepted(self):
        for language in ('yaml','json','python','javascript','js','typescript','ts','toml','text','sh','sql'):
            self.assertEqual(ReplyContract.code_block(language=language).code_language,language)
        for language in ('',None,True,'unknown','YAML','yaml\n<<<HEAD>>>','python extra'):
            with self.subTest(language=language), self.assertRaises(ValueError):
                ReplyContract.code_block(language=language)
        with self.assertRaises(ValueError):ReplyContract(code_language='yaml')
        with self.assertRaises(ValueError):ReplyContract(kind='code_block')

    def test_actual_code_payload_preserved_for_all_three_paths(self):
        for path in ('full','compact','direct'):
            raw=envelope(CODE,path)
            framed=frame_code_body(raw,path,self.contract)
            self.assertNotEqual(framed,raw)
            result=parse_output(framed,path)
            self.assertEqual(result.say,CODE)
            self.assertEqual(safe_say(result.say),CODE)
            self.assertNotIn('PRIVATE_REFERENCE',result.say)
            self.assertEqual(result.structural_errors,[])
            self.assertTrue(self.contract.validate(result.say)['valid'])
            self.assertEqual(frame_code_body(framed,path,self.contract),framed)

    def test_payload_section_markers_cannot_become_outer_sections(self):
        code='```yaml\nexample: |\n  <<<HEAD>>>\n  direct: literal\n  <<<SAY>>>\n  <<<TAIL>>>\npending: 2\n```'
        for path in ('full','compact','direct'):
            raw=envelope(code,path);framed=frame_code_body(raw,path,self.contract)
            result=parse_output(framed,path)
            self.assertEqual(result.say,code)
            self.assertEqual(result.structural_errors,[])
            if path=='full':
                self.assertEqual(section_errors(framed),[])
                self.assertIsNone(result.tail.pending)
            else:self.assertIsNone(result.tail)

    def test_no_code_authority_without_application_contract(self):
        for contract in (None,ReplyContract.json_object(),ReplyContract.conclusion()):
            for path in ('full','compact','direct'):
                raw=envelope(CODE,path)
                self.assertEqual(frame_code_body(raw,path,contract),raw)
                self.assertEqual(parse_output(raw,path).say,'')

    def test_wrong_language_incomplete_multiple_or_explained_blocks_are_not_rewritten(self):
        invalid=('```python\nx=1\n```','```\ndirect: false\n```',
                 '```unknown\ndirect: false\n```','```yaml\ndirect: false',
                 CODE+'\n'+CODE,'说明：\n'+CODE,CODE+'\n后续说明',
                 '~~~yaml\ndirect: false\n~~~')
        for code in invalid:
            self.assertFalse(self.contract.validate(code)['valid'])
            for path in ('full','compact','direct'):
                raw=envelope(code,path)
                self.assertEqual(frame_code_body(raw,path,self.contract),raw)

    def test_longer_fence_and_internal_short_fences_preserve_original_bytes(self):
        code='````yaml\ndirect: |\n  ```python\n  x = 1\n  ```\npending: 2\n````'
        self.assertTrue(self.contract.validate(code)['valid'])
        for path in ('full','compact','direct'):
            self.assertEqual(parse_output(frame_code_body(envelope(code,path),path,self.contract),path).say,code)

    def test_outer_head_order_and_fields_are_required(self):
        bad=('<<<SAY>>>\n'+CODE+'\n<<<TAIL>>>\n'+TAIL,
             '前言\n'+envelope(CODE,'full'),
             '<<<HEAD>>>\n'+HEAD+'\n<<<HEAD>>>\n'+HEAD+'\n<<<SAY>>>\n'+CODE+'\n<<<TAIL>>>\n'+TAIL,
             envelope(CODE,'full').replace('<<<HEAD>>>','<<<head>>>'),
             envelope(CODE,'full').replace('confidence: .9','confidence: invalid'),
             envelope(CODE,'full').replace('baseline: 按格式提供代码\n',''),
             '<<<HEAD>>>\n'+HEAD+'\n<<<TAIL>>>\n'+TAIL+'\n<<<SAY>>>\n'+CODE)
        for raw in bad:
            self.assertEqual(frame_code_body(raw,'full',self.contract),raw)
        raw=envelope(CODE,'compact').replace('confidence: .9','confidence: 3')
        self.assertEqual(frame_code_body(raw,'compact',self.contract),raw)

    def test_full_body_does_not_depend_on_tail_authority_and_compact_forbids_it(self):
        for raw in (make_reply(HEAD,CODE),make_reply(HEAD,CODE,'unrecognized metadata'),
                    make_reply(HEAD,CODE)+'\n<<<TAIL>>>\n',
                    make_reply(HEAD,CODE,TAIL+'\n<<<HEAD>>>PRIVATE'),
                    make_reply(HEAD,CODE,TAIL+'\n```yaml\nx: 1\n```')):
            framed, errors=prepare_code_body(raw,'full',self.contract)
            parsed=parse_output(framed,'full')
            parsed.structural_errors.extend(errors)
            self.assertNotEqual(framed,raw)
            self.assertEqual(parsed.say,CODE)
            self.assertFalse(parsed.validity()['valid'])
        raw=make_reply(COMPACT,CODE,TAIL)
        self.assertEqual(frame_code_body(raw,'compact',self.contract),raw)

    def test_head_or_tail_code_examples_do_not_establish_say_authority(self):
        raw=make_reply(HEAD+'\n```yaml\ndirect: secret\n```',CODE,TAIL)
        self.assertEqual(frame_code_body(raw,'full',self.contract),raw)
        raw=envelope(CODE+'\ndirect: PRIVATE','full')
        self.assertEqual(frame_code_body(raw,'full',self.contract),raw)

    def test_whitespace_line_endings_and_empty_block_are_format_only(self):
        code='  ```yaml\r\n  direct: false\r\n  pending: 2\r\n  ```  '
        span=code_block_span('\r\n'+code+'\r\n','yaml')
        self.assertEqual(('\r\n'+code+'\r\n')[slice(*span)],code)
        self.assertTrue(self.contract.validate('\r\n'+code+'\r\n')['valid'])
        framed=frame_code_body('\r\n'+code+'\r\n','direct',self.contract)
        self.assertEqual(parse_output(framed,'direct').say,code)
        self.assertTrue(self.contract.validate('```yaml\n```')['valid'])
        # The contract validates transport shape, not language syntax or truth.
        self.assertTrue(self.contract.validate('```yaml\nnot valid: [\n```')['valid'])

    def test_helper_rejects_invalid_api_arguments(self):
        with self.assertRaises(TypeError):frame_code_body(123,'direct',self.contract)
        with self.assertRaises(ValueError):frame_code_body(CODE,'unknown',self.contract)
        with self.assertRaises(TypeError):frame_code_body(CODE,'direct',object())
        with self.assertRaises(TypeError):code_block_span(123,'yaml')


if __name__=='__main__':unittest.main()
