"""Anonymous exact-text units. Does not generate or merge any judgments."""
import json
import random
import benchmark as b


def main():
    b.verify()
    data = b.load(b.HERE / 'run/results.json')
    rows = data['records']
    profiles = {p['id']: p for p in b.load(b.HERE / 'profiles.json')}
    units, mapping = {}, {}
    for row in rows:
        if not isinstance(row.get('say'), str) or not row['say'].strip():
            continue
        payload = row['payload']
        context = dict(task=payload['context'], facts=payload['facts'],
                       history=payload.get('history', []), simulation_state=payload.get('state', {}),
                       note='状态是模拟备忘，不是现实执行或用户确认的自动证明；以当前原话和确认证据核对。')
        public = dict(context=context, user=payload['user'], reply=row['say'], persona=profiles[row['profile']]['spec'])
        key = json.dumps(public, ensure_ascii=False, sort_keys=True)
        units.setdefault(key, public)
        mapping[row['id']] = key
    keys = list(units)
    random.Random(202611103).shuffle(keys)
    aliases = {key: f'R{i:03}' for i, key in enumerate(keys, 1)}
    b.save(b.HERE / 'reviews/inputs.json', [dict(id=aliases[key], **units[key]) for key in keys])
    public_rows = [dict(id=aliases[key], **units[key]) for key in keys]
    for offset in range(0, len(public_rows), 24):
        b.save(b.HERE / 'reviews/packets' / f'packet_{offset//24+1:02}.json', public_rows[offset:offset+24])
    b.save(b.HERE / 'review_key.json', {id: aliases[key] for id, key in mapping.items()})
    print({'generated_units': len(rows), 'scorable_texts': len(mapping), 'unique_review_units': len(keys)})


if __name__ == '__main__':
    main()
