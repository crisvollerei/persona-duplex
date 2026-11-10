"""Research-only context-aware choice. No changes to the default Session."""
import hashlib
import json


def strict_json(text):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj
    def constant(value):
        raise ValueError("non-finite JSON")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def payload(case, variant, profile, repetition):
    """Both methods receive exactly the same evidence, parameters, and draw."""
    key = f"{case['id']}/{repetition}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return dict(case=case['id'], context=case['context'], user=variant['user'],
                facts=case['facts'], candidates=case['actions'],
                allowed_actions=variant['allowed_actions'],
                requested=variant['requested'], request_scope=variant['request_scope'],
                persona=profile, draw=draw,
                environment="纯合成备忘场景；effects仅表示未确认草案，不执行现实事务。")


RULE = """选择规则（两种方法完全共享）：
1. 只考虑allowed_actions中的候选ID。requested=true时，再保留kind在request_scope中的候选；这是应用已确认的请求范围，不由人格覆盖。
2. requested=false时，draw < persona.intervention才偏向propose/ask（额外推进）；否则偏向respond/answer（接住已有议题）。对应类别没有合法候选时使用全部合法候选。
3. 在剩余候选中取persona中与候选focus同名的偏好最大者。并列时按候选ID字典序最小。不要用文风改写事实、边界、请求或已确认状态。
这些是本研究设计的选择规则，不是人格心理学定律。候选和请求范围由作者审核，不代表系统已能从任意自然语言可靠提取它们。
"""


def choose(data, *, contextual=True):
    available = [a for a in data['candidates'] if a['id'] in data['allowed_actions']]
    if contextual and data['requested']:
        available = [a for a in available if a['kind'] in data['request_scope']]
    else:
        extension = data['draw'] < data['persona']['intervention']
        preferred = [a for a in available if (a['kind'] in ('propose', 'ask')) == extension]
        available = preferred or available
    if not available:
        return None
    return min(available, key=lambda a: (-data['persona'][a['focus']], a['id']))['id']


def parse_reply(text, data):
    obj = strict_json(text)
    if not isinstance(obj, dict) or set(obj) != {'action_id', 'say'}:
        raise ValueError('expected action_id and say only')
    if not isinstance(obj['say'], str) or not obj['say'].strip():
        raise ValueError('empty say')
    if not isinstance(obj['action_id'], str) or obj['action_id'] not in {a['id'] for a in data['candidates']}:
        raise ValueError('unknown action')
    return obj


COMMON = """你参与一个合成活动规划研究。请按当前人格与情境写出实际给用户的中文回复。
完整保留事实、条件、未知和用户边界。人格允许正式、机械、活泼等不同语感，没有通用自然度门槛。
candidate.text是意图参考，可以自然改写，但不能把待确认变成确定、未执行变成已执行，也不能新增候选未涵盖的安排或问题。
候选effects是草案备忘，不能声称已在真实系统保存、通知、预订或执行。仅输出JSON对象，字段为action_id和say。
"""


def messages(data, method):
    text = COMMON + '\n' + RULE
    user = dict(data)
    if method == 'program':
        user['selected_action_id'] = choose(data)
        text += '\n程序已选择selected_action_id；只表达该候选，不重新选择。'
    elif method == 'prompt':
        text += '\n你在同一次调用中按共享规则选择候选并表达；无需输出解释或推导。'
    else:
        raise ValueError('unknown method')
    return [{'role': 'system', 'content': text},
            {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)}]


LABEL_SYSTEM = """你是行为编码者，只依据每项的context、user、reply，独立判断，不推测作者期望或生成方法。
fulfills_request: 完成当前明确请求为true，未完成为false；无明确请求或无法判断为null。
adds_agenda: 在当前请求范围之外新增行动安排、方案或需回答的信息目标为true；仅完成请求、照应已知内容为false；不清楚为null。建议句式/问号不自动等于新增议程。
fact_status: supported（有依据或无额外事实断言）、contradicted（明确冲突）、unknown（无法核实）。
boundary_status: respected（无违规或未设边界）、violated（明确违反）、unknown（不清楚）。
输出一个JSON对象{\"ratings\":[{\"id\":\"...\",\"fulfills_request\":true,\"adds_agenda\":false,\"fact_status\":\"supported\",\"boundary_status\":\"respected\",\"reason\":\"简短定位原话\"}]}。
每项恰好一条，允许null；不要把无明确请求编码成完成请求，也不要用善意豁免越界。
"""


def validate_labels(obj, ids):
    if not isinstance(obj, dict) or set(obj) != {'ratings'} or not isinstance(obj['ratings'], list):
        raise ValueError('ratings object required')
    rows = obj['ratings']
    if len(rows) != len(ids) or any(not isinstance(r, dict) for r in rows):
        raise ValueError('rating count')
    if any(not isinstance(r.get('id'), str) for r in rows) or {r['id'] for r in rows} != set(ids):
        raise ValueError('rating IDs')
    for r in rows:
        if set(r) != {'id', 'fulfills_request', 'adds_agenda', 'fact_status', 'boundary_status', 'reason'}:
            raise ValueError('rating fields')
        if any(r[k] is not None and type(r[k]) is not bool for k in ('fulfills_request', 'adds_agenda')):
            raise ValueError('boolean or null required')
        if r['fact_status'] not in ('supported', 'contradicted', 'unknown') or r['boundary_status'] not in ('respected', 'violated', 'unknown'):
            raise ValueError('invalid status')
        if not isinstance(r['reason'], str) or not r['reason'].strip():
            raise ValueError('reason required')
    return rows
