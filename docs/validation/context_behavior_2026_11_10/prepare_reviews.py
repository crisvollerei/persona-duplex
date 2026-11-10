"""Make anonymous, exact-text review units. Never generates ratings."""
import random
import benchmark as b


def main():
    b.verify()
    rows = b.load(b.HERE / 'run/results.json')['records']
    profiles = {p['id']: p for p in b.load(b.HERE / 'profiles.json')}
    unique, mapping = {}, {}
    for r in rows:
        if r['status'] != 'ok' or r['say'] is None:
            continue
        data = r['payload']
        key = (r['profile'], data['context'], data['user'], r['say'])
        if key not in unique:
            unique[key] = dict(context=data['context'], user=data['user'], reply=r['say'],
                               persona=profiles[r['profile']]['spec'])
        mapping[r['id']] = key
    keys = list(unique)
    random.Random(202611102).shuffle(keys)
    aliases = {k: f'R{i:03}' for i, k in enumerate(keys, 1)}
    b.save(b.HERE / 'reviews/inputs.json', [dict(id=aliases[k], **unique[k]) for k in keys])
    b.save(b.HERE / 'review_key.json', {id: aliases[k] for id, k in mapping.items()})
    print({'generated': len(rows), 'unique_review_units': len(keys), 'mapped_outputs': len(mapping)})


if __name__ == '__main__':
    main()
