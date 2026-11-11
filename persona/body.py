"""Caller-declared code bodies share the verified literal SAY boundary.

Framing is a deterministic transport operation: it neither invents code nor
repairs malformed output, and it never grants state authority to code fields.
"""
from __future__ import annotations

import re

from .contracts import ReplyContract
from .parse import (encode_literal_say, mask_literal_blocks, parse_head,
                    parse_tail, section_errors)
from .prompt import SECTION_HEAD, SECTION_SAY, SECTION_TAIL

_OPEN_FENCE = re.compile(
    r"\A(?P<padding>(?:[ \t]*(?:\r\n|\n))*)(?P<indent> {0,3})"
    r"(?P<fence>`{3,})(?P<language>[A-Za-z][A-Za-z0-9_+-]*)[ \t]*(?:\r\n|\n)")
_FENCE_LINE = re.compile(r"^ {0,3}(?:`{3,}|~{3,})[^\r\n]*$", re.MULTILINE)


def _code_prefix(text: str, language: str) -> tuple[int, int] | None:
    """Find one exact-language opening block and its first Markdown close."""
    opening = _OPEN_FENCE.match(text)
    if opening is None or opening['language'] != language:
        return None
    width = len(opening['fence'])
    closing = re.compile(r"^ {0,3}`{"+str(width)+r",}[ \t]*$")
    cursor = opening.end()
    for line in text[cursor:].splitlines(keepends=True):
        content = line[:-2] if line.endswith('\r\n') else line[:-1] if line.endswith('\n') else line
        if closing.fullmatch(content):
            return opening.start('indent'), cursor+len(content)
        cursor += len(line)
    return None


def code_block_span(text: str, language: str) -> tuple[int, int] | None:
    """Return the original single fenced document, with only outer whitespace."""
    if not isinstance(text, str):
        raise TypeError('Code reply must be text')
    span = _code_prefix(text, language)
    if span is None or text[span[1]:].strip():
        return None
    return span


def _tail_shape_errors(content: str) -> list[str]:
    """Do not grant state authority to ignored text around a framed code body."""
    def is_empty(value: str) -> bool:
        value = value.strip()
        wrapped = re.fullmatch(r'（([^（）]*)）|\(([^()]*)\)', value)
        if wrapped:
            value = next(part for part in wrapped.groups() if part is not None).strip()
        return value.lower() in ('', '-', 'none', '无', 'null', 'nil', 'n/a', '暂无', '不挂')

    def valid_settle(value: str) -> bool:
        if is_empty(value):
            return True
        for piece in re.split(r'[;；][ \t]*(?=p_)', value):
            parts = [part.strip() for part in piece.split('|')]
            if len(parts) < 2 or parts[1].lower() not in ('confirmed', 'refuted', 'watch'):
                return False
            if not re.fullmatch(r'[^\s|]+', parts[0].strip(',， ')):
                return False
        return True

    parsed = parse_tail(content)
    errors = [*parsed.errors, *(parsed.pending.errors if parsed.pending else [])]
    section = ''
    empty = False
    seen = set()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        field = re.fullmatch(r'(settle|delta|pending)[ \t]*[:：](.*)', line, re.I)
        if field:
            section, value = field[1].lower(), field[2].strip()
            if section != 'settle' and section in seen:
                errors.append('duplicate_code_tail_field')
            seen.add(section)
            if section == 'settle':
                if not valid_settle(value):
                    errors.append('invalid_code_tail_settle')
            elif not is_empty(value):
                # The generic parser reads delta and pending subfields on
                # following lines; inline prose is not a state proposal.
                errors.append('invalid_code_tail_'+section)
            empty = bool(value) and is_empty(value)
            continue
        if is_empty(line):
            continue
        nested = re.fullmatch(r'[-*]?[ \t]*([A-Za-z_.]+)[ \t]*[:：].+', line)
        if nested and not empty:
            if section == 'delta' or section == 'pending' and nested[1].lower() in ('claim','confidence','predicts'):
                continue
        errors.append('unrecognized_code_tail_content')
    if _FENCE_LINE.search(content):
        errors.append('fence_outside_say')
    return list(dict.fromkeys(errors))


def prepare_code_body(raw: str, generation_path: str,
                      contract: ReplyContract | None) -> tuple[str, list[str]]:
    """Return the original framed body and separate outer-protocol errors.

    Full/compact envelopes must have their own valid HEAD before the first SAY.
    Complete full SAY code does not acquire state authority from a damaged or
    absent TAIL. The caller must carry these errors into protocol validation.
    Nothing after the closing SAY fence is moved into the visible body.
    """
    if not isinstance(raw, str):
        raise TypeError('Raw model output must be text')
    if generation_path not in ('full', 'compact', 'direct'):
        raise ValueError('Unknown generation path')
    if contract is not None and not isinstance(contract, ReplyContract):
        raise TypeError('contract must be ReplyContract or None')
    if contract is None or contract.kind != 'code_block':
        return raw, []
    language = contract.code_language
    if generation_path == 'direct':
        span = code_block_span(raw, language)
        if span is None:
            return raw, []
        start, end = span
        return raw[:start]+encode_literal_say(raw[start:end])+raw[end:], []

    # Only the real, ordered outer envelope establishes where SAY starts.
    prefix = re.match(r"\A\s*"+re.escape(SECTION_HEAD), raw)
    if prefix is None:
        return raw, []
    masked = mask_literal_blocks(raw, sections=True)
    say_at = masked.find(SECTION_SAY, prefix.end())
    if say_at < 0:
        return raw, []
    header = raw[:say_at+len(SECTION_SAY)]
    masked_header = masked[:say_at+len(SECTION_SAY)]
    if (masked_header.count(SECTION_HEAD) != 1 or masked_header.count(SECTION_SAY) != 1
            or SECTION_TAIL in masked_header):
        return raw, []
    head = parse_head(raw[prefix.end():say_at])
    allowed_missing = {'missing_baseline','missing_displace'} if generation_path == 'compact' else set()
    if any(error not in allowed_missing for error in head.errors):
        return raw, []
    if generation_path == 'compact' and (head.baseline or head.displace not in ('','none')):
        return raw, []
    if _FENCE_LINE.search(header):
        return raw, []
    say_start = say_at+len(SECTION_SAY)
    body = raw[say_start:]
    span = _code_prefix(body, language)
    if span is None:
        return raw, []
    start, end = span
    suffix = body[end:]
    outer_errors = []
    if generation_path == 'compact':
        if suffix.strip():
            return raw, []
    else:
        tail = re.fullmatch(r"\s*"+re.escape(SECTION_TAIL)+r"(?P<content>[\s\S]*)", suffix)
        if tail is None:
            # No TAIL is an invalid full protocol, but a complete standalone
            # SAY body still has an unambiguous end. Extra SAY prose does not.
            if suffix.strip():
                return raw, []
        elif tail['content'].strip():
            outer_errors.extend(_tail_shape_errors(tail['content']))
    framed = raw[:say_start+start]+encode_literal_say(body[start:end])+suffix
    outer_errors.extend(section_errors(framed, expression_only=generation_path=='compact'))
    return framed, list(dict.fromkeys(outer_errors))


def frame_code_body(raw: str, generation_path: str, contract: ReplyContract | None) -> str:
    """Compatibility helper for framing; engines also retain protocol errors."""
    return prepare_code_body(raw, generation_path, contract)[0]
