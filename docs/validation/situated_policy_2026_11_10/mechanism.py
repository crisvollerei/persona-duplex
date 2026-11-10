"""Finite situated choices and an application-owned simulation executor."""
from copy import deepcopy
import hashlib
import json


LABEL_VALUES = {
    'request_presence': ('explicit', 'none', 'unclear'),
    'fulfillment': ('fulfilled', 'failed', 'not_applicable', 'unclear'),
    'extension': ('yes', 'no', 'unclear'),
    'boundary_kind': ('explicit', 'none', 'implicit', 'unclear'),
    'boundary': ('respected', 'violated', 'not_applicable', 'unclear'),
    'grounding': ('supported', 'contradicted', 'unknown'),
}


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('non-finite JSON')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate_labels(obj, ids, *, main=False):
    if not isinstance(obj, dict) or set(obj) != {'ratings'} or not isinstance(obj['ratings'], list):
        raise ValueError('ratings object required')
    rows = obj['ratings']
    if len(rows) != len(ids) or any(not isinstance(r, dict) for r in rows):
        raise ValueError('rating count')
    if any(not isinstance(r.get('id'), str) for r in rows) or {r['id'] for r in rows} != set(ids):
        raise ValueError('rating IDs')
    for row in rows:
        extra = {'persona_fit', 'context_fit', 'focus'} if main else set()
        if set(row) != {'id', 'reason', *LABEL_VALUES, *extra}:
            raise ValueError('rating fields')
        if any(row[field] not in choices for field, choices in LABEL_VALUES.items()):
            raise ValueError('invalid label value')
        if not isinstance(row['reason'], str) or not row['reason'].strip():
            raise ValueError('reason required')
        if main:
            if any(row[f] is not None and (type(row[f]) is not int or not 1 <= row[f] <= 5)
                   for f in ('persona_fit', 'context_fit')):
                raise ValueError('invalid ordinal score')
            if row['focus'] not in ('practical', 'feeling', 'detail', 'mixed', 'unclear'):
                raise ValueError('invalid focus')
    return rows


def payload(case, turn, profile, repetition, state, history):
    key = f"{case['id']}/{turn['id']}/{repetition}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') / 2**64
    return deepcopy(dict(case=case['id'], turn=turn['id'], context=case['context'],
        facts=case['facts'], user=turn['user'], requested=turn['requested'],
        request_scope=turn['request_scope'], allowed_actions=turn['allowed_actions'],
        boundary=turn.get('boundary'), candidates=turn['candidates'], persona=profile,
        state=state, history=history, draw=draw,
        environment='只有模拟备忘状态；不执行现实预订、打印、联系或通知。'))


RULE = """共享选择规则：
1. 仅考虑allowed_actions中的候选ID。requested=true时，进一步只保留kind在request_scope里的候选；应用确认的请求范围优先于人格偏好。
2. 无明确请求时，draw < persona.intervention偏向propose/ask，其余偏向respond/answer/listen；偏向类别无合法候选时使用全部合法候选。
3. 从剩余候选中选persona中与focus同名参数最高者，并列选ID字典序最小者。draw已提供，不另行抽样。
候选意图、参数和规则是作者设计的工程对象，不是心理学人格真值。状态只能由共享执行器按作者候选effects更新，回复本身不执行现实事务。
"""


def choose(data, *, contextual=True):
    candidates = [a for a in data['candidates'] if a['id'] in data['allowed_actions']]
    if contextual and data['requested']:
        candidates = [a for a in candidates if a['kind'] in data['request_scope']]
    else:
        extension = data['draw'] < data['persona']['intervention']
        preferred = [a for a in candidates if (a['kind'] in ('propose', 'ask')) == extension]
        candidates = preferred or candidates
    if not candidates:
        return None
    return min(candidates, key=lambda a: (-data['persona'][a['focus']], a['id']))['id']


COMMON = """按人格与当前情境写出给用户的实际中文回复。事实、计算、未确认状态和明确权限不得因人格改变。
候选text是意图参考，可自然改写；不得新增候选意图以外的计划、问题或无依据断言。不能把草案写成确定，不能声称执行了真实预订、保存、打印、联系、通知。
正式、略显机械、克制或活泼均可以符合不同人设，没有通用自然度门槛。仅输出JSON对象，恰有action_id与say两个字段。
"""


def messages(data, method):
    user = deepcopy(data)
    system = COMMON + '\n' + RULE
    if method == 'program':
        user['selected_action_id'] = choose(data)
        system += '\n程序已选择selected_action_id；请表达该候选，不重新选行动。'
    elif method == 'prompt':
        system += '\n请在同一次调用中按共享规则选择行动并表达，不输出推导。'
    else:
        raise ValueError('unknown method')
    return [dict(role='system', content=system),
            dict(role='user', content=json.dumps(user, ensure_ascii=False, allow_nan=False))]


def parse_reply(text, data):
    obj = strict_json(text)
    if not isinstance(obj, dict) or set(obj) != {'action_id', 'say'}:
        raise ValueError('expected action_id and say')
    if not isinstance(obj['action_id'], str) or obj['action_id'] not in {a['id'] for a in data['candidates']}:
        raise ValueError('unknown action ID')
    if not isinstance(obj['say'], str) or not obj['say'].strip():
        raise ValueError('empty say')
    return obj


def apply_confirmed_updates(state, updates, allowed_keys=None):
    """Only application-authored updates; this API never consumes model JSON."""
    allowed = set(state) if allowed_keys is None else set(allowed_keys)
    if not isinstance(updates, dict) or not set(updates) <= allowed:
        raise ValueError('unknown confirmed update key')
    # Round-trip rejects NaN/infinity and non-serializable application state.
    json.dumps(updates, allow_nan=False)
    result = deepcopy(state)
    result.update(deepcopy(updates))
    return result


def execute_effects(data, action_id, state, allowed_keys=None):
    """The same permission/request gate for both routes, not a persona oracle."""
    unchanged = deepcopy(state)
    candidate = next((a for a in data['candidates'] if a['id'] == action_id), None)
    if candidate is None:
        return dict(accepted=False, executed_id=None, state=unchanged, reason='unknown_action')
    if action_id not in data['allowed_actions']:
        return dict(accepted=False, executed_id=None, state=unchanged, reason='permission_rejected')
    if data['requested'] and candidate['kind'] not in data['request_scope']:
        return dict(accepted=False, executed_id=None, state=unchanged, reason='request_scope_rejected')
    try:
        new_state = apply_confirmed_updates(state, candidate['effects'], allowed_keys)
    except (ValueError, TypeError):
        return dict(accepted=False, executed_id=None, state=unchanged, reason='invalid_author_effects')
    return dict(accepted=True, executed_id=action_id, state=new_state, reason='author_effects_applied')
