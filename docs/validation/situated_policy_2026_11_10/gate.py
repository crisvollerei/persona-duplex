"""Frozen semantic-interpretation branch, separate from mechanical research."""
from collections import defaultdict
import benchmark as b
import mechanism as m

FIELDS = ('request_presence', 'fulfillment', 'extension', 'boundary_kind', 'boundary', 'grounding')


def ratings(path):
    obj = b.load(path)
    return obj['ratings'] if isinstance(obj, dict) else obj


def assess(rows, key):
    m.validate_labels({'ratings': rows}, list(key))
    by_id = {r['id']: r for r in rows}
    if set(by_id) != set(key) or len(rows) != len(key):
        raise ValueError('missing or duplicate calibration IDs')
    clear = [id for id, r in key.items() if not r.get('probe', False)]
    probes = [id for id in key if id not in clear]
    correct = {f: sum(by_id[id][f] == key[id]['expected'][f] for id in clear) for f in FIELDS}
    mismatches = [dict(id=id, original_id=key[id]['original_id'], field=f,
                       expected=key[id]['expected'][f], observed=by_id[id][f], reason=by_id[id]['reason'])
                  for id in clear for f in FIELDS if by_id[id][f] != key[id]['expected'][f]]
    faults = [(id, f) for id in clear for f in ('grounding', 'boundary')
              if (f == 'grounding' and key[id]['expected'][f] == 'contradicted')
              or (f == 'boundary' and key[id]['expected']['boundary_kind'] == 'explicit'
                  and key[id]['expected'][f] == 'violated')]
    found = sum(by_id[id][f] == key[id]['expected'][f] for id, f in faults)
    agreement = sum(correct.values()) / (len(clear) * len(FIELDS))
    pairs = defaultdict(list)
    for id, row in key.items():
        pairs[row['pair_id']].append(id)
    pair_results = []
    for pair, ids in sorted(pairs.items()):
        pair_results.append(dict(pair=pair, probe=any(key[id].get('probe', False) for id in ids),
             all_expected=all(by_id[id][f] == key[id]['expected'][f] for id in ids for f in FIELDS)))
    return dict(clear_n=len(clear), probe_n=len(probes), clear_field_correct=correct,
                clear_field_denominator=len(clear) * len(FIELDS), agreement=agreement,
                explicit_faults=len(faults), explicit_faults_found=found,
                passed=agreement >= .9 and found == len(faults), mismatches=mismatches,
                probe_ratings=[dict(id=id, **{f: by_id[id][f] for f in FIELDS}, reason=by_id[id]['reason']) for id in probes],
                pairs=pair_results)


def main():
    b.verify()
    key = b.load(b.HERE / 'calibration_key.json')
    assessments = {c: assess(ratings(b.HERE / 'calibration_reviews' / ('ratings_' + c + '.json')), key)
                   for c in ('1', '2')}
    repeats = []
    for row in b.load(b.HERE / 'calibration_run/results.json')['records']:
        repeats.append(dict(repetition=row['repetition'], status=row['status'],
             assessment=assess(row['ratings'], key) if row['status'] == 'ok' else None))
    result = dict(protocol_sha256=b.sha(b.HERE / 'protocol.json'),
                  semantic_eligible=all(r['passed'] for r in assessments.values()),
                  mechanical_study='proceed_as_preregistered', assessments=assessments, backend_repeats=repeats,
                  interpretation='Author-expectation calibration, not semantic truth or human validation.')
    b.save(b.HERE / 'calibration_gate.json', result)
    print({'semantic_eligible': result['semantic_eligible'],
           'assessments': {k: (v['agreement'], v['explicit_faults_found'], v['explicit_faults']) for k,v in assessments.items()}})


if __name__ == '__main__':
    main()
