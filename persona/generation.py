"""Explicit single-call text layouts; missing observations stay unobserved."""
from .parse import (ParsedTurn, parse, safe_say, literal_errors, control_errors,
                    detect_declarations, detect_read_aloud)

PATHS = ('full', 'compact', 'direct')


def adapt(parsed: ParsedTurn, path: str) -> ParsedTurn:
    if path == 'full':
        return parsed
    if path not in PATHS:
        raise ValueError('Unknown generation path')
    expected = {'缺少 TAIL'} | ({'缺少 HEAD'} if path == 'direct' else set())
    parsed.errors = [e for e in parsed.errors if e not in expected]
    missing_markers = {'missing_tail_marker'} | (
        {'missing_head_marker','missing_say_marker'} if path == 'direct' else set())
    parsed.structural_errors = [e for e in parsed.structural_errors if e not in missing_markers]
    if parsed.tail is not None:
        parsed.structural_errors.append('unexpected_tail_in_'+path)
    parsed.tail = None  # These layouts never authorize model state proposals.
    if path == 'direct' and parsed.head is not None:
        parsed.structural_errors.append('unexpected_head_in_direct')
        parsed.head = None
    if path == 'compact' and parsed.head and (
            parsed.head.baseline or parsed.head.displace not in ('', 'none')):
        parsed.structural_errors.append('unexpected_compact_observation')
    if path == 'compact' and parsed.head:
        parsed.head.errors = [e for e in parsed.head.errors
                              if e not in ('missing_baseline','missing_displace')]
    return parsed


def parse_output(raw: str, path: str) -> ParsedTurn:
    if path == 'direct':
        say = safe_say(raw)
        return ParsedTurn(None, say, None,
            declared=detect_declarations(say),read_aloud=detect_read_aloud(say),
            errors=[] if say.strip() else ['缺少 SAY'],
            structural_errors=[*literal_errors(raw), *control_errors(raw)])
    return adapt(parse(raw, expression_only=path=='compact'),path)


def validity(parsed, path, *, protocol='text', probe=None, allowed=True):
    if path == 'full':
        return parsed.validity(protocol=protocol,probe=probe,allowed=allowed)
    reasons = [*parsed.errors,*parsed.structural_errors]
    if path == 'compact':
        if parsed.head is None:
            reasons.append('missing_head')
        else:
            reasons.extend(parsed.head.errors)
            for field in ('need','direct'):
                if not getattr(parsed.head,field).strip():reasons.append('missing_'+field)
    if not parsed.say.strip():reasons.append('missing_say')
    if parsed.declared:reasons.append('declared_displacement')
    for field in ('stream_interrupted','output_blocked','observation_failed'):
        if (probe or {}).get(field):reasons.append(field)
    reasons=list(dict.fromkeys(reasons))
    return dict(valid=not reasons,status='invalid' if reasons else 'valid',reasons=reasons,
                scope='local_protocol_not_semantic_truth',generation_path=path,
                observations='content_reference_only' if path=='compact' else 'not_collected')
