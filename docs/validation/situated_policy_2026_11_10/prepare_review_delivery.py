"""Public language-evaluation context including independently applied effects."""
import argparse
import json
import random
import benchmark as b


def register():
    path = b.HERE / 'review_context_protocol.json'
    if path.exists():
        raise ValueError('review context already registered')
    b.verify()
    b.save(path, dict(registered_at=b.now(), generation_protocol_sha256=b.sha(b.HERE / 'protocol.json'),
         effective_input_builder='prepare_review_delivery.py',
         hashes={f: b.sha(b.HERE / f) for f in ('prepare_review_delivery.py', 'rubric.md')},
         public_context='Facts, actual history, state before expression, and independently executed simulation result.',
         hidden='Method, action IDs, selection rules, labels, other ratings.',
         interpretation='The application result confirms only simulation effects, never real-world execution.'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--register', action='store_true'); a = p.parse_args()
    if a.register:
        register()
        return
    b.verify()
    contract = b.load(b.HERE / 'review_context_protocol.json')
    for name, digest in contract['hashes'].items():
        assert b.sha(b.HERE / name) == digest
    rows = b.load(b.HERE / 'run/results.json')['records']
    profiles = {p['id']: p for p in b.load(b.HERE / 'profiles.json')}
    units, mapping = {}, {}
    for row in rows:
        if not isinstance(row.get('say'), str) or not row['say'].strip():
            continue
        data = row['payload']
        context = dict(task=data['context'], facts=data['facts'], history=data['history'],
                       state_before_expression=row['state_after_user'],
                       application_result=dict(accepted=row['executed_id'] is not None,
                                               state_after_expression=row['state_after_action']),
                       note='应用结果仅确认模拟备忘变化，不证明现实执行、用户已认可方案或文字没有额外断言。')
        public = dict(context=context, user=data['user'], reply=row['say'], persona=profiles[row['profile']]['spec'])
        key = json.dumps(public, ensure_ascii=False, sort_keys=True)
        units.setdefault(key, public)
        mapping[row['id']] = key
    keys = list(units)
    random.Random(202611103).shuffle(keys)
    aliases = {key: f'R{i:03}' for i, key in enumerate(keys, 1)}
    public = [dict(id=aliases[key], **units[key]) for key in keys]
    b.save(b.HERE / 'reviews/inputs.json', public)
    b.save(b.HERE / 'review_key.json', {id: aliases[key] for id,key in mapping.items()})
    for start in range(0, len(public), 24):
        b.save(b.HERE / 'reviews/packets' / f'packet_{start//24+1:02}.json', public[start:start+24])
    print({'planned_generation_units': len(rows), 'scorable_texts': len(mapping), 'unique_review_units': len(public)})


if __name__ == '__main__':
    main()
