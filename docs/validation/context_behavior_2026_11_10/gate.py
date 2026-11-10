"""Recompute the frozen calibration decision; no model calls."""
import argparse
from collections import defaultdict
from pathlib import Path
import benchmark as b
import mechanism as m


FIELDS = ('fulfills_request', 'adds_agenda', 'fact_status', 'boundary_status')


def assess(ratings, key):
    ratings = m.validate_labels({'ratings': ratings}, list(key))
    by_id = {r['id']: r for r in ratings}
    pairs = defaultdict(list)
    fields = {f: 0 for f in FIELDS}
    faults, detected = 0, 0
    mismatches = []
    for id, row in key.items():
        exp, got = row['expected'], by_id[id]
        pairs[row['original_id'][:-1]].append(id)
        for f in FIELDS:
            ok = got[f] == exp[f]
            fields[f] += ok
            if not ok:
                mismatches.append(dict(id=id, original_id=row['original_id'], field=f,
                                       expected=exp[f], observed=got[f], reason=got['reason']))
            if (f == 'fact_status' and exp[f] == 'contradicted') or (f == 'boundary_status' and exp[f] == 'violated'):
                faults += 1
                detected += ok
    changed, matched, controls, controls_matched = 0, 0, 0, 0
    for ids in pairs.values():
        assert len(ids) == 2
        changing = [f for f in FIELDS[:2] if key[ids[0]]['expected'][f] != key[ids[1]]['expected'][f]]
        if changing:
            changed += 1
            matched += all(by_id[id][f] == key[id]['expected'][f] for id in ids for f in changing)
        else:
            controls += 1
            controls_matched += all(by_id[id][f] == key[id]['expected'][f] for id in ids for f in FIELDS)
    score = sum(fields.values()) / (len(key) * len(FIELDS))
    return dict(field_correct=fields, denominator_per_field=len(key), overall_agreement=score,
                contrast_pairs=changed, correct_contrast_pairs=matched,
                control_pairs=controls, correct_control_pairs=controls_matched,
                faults=faults, detected_faults=detected, mismatches=mismatches,
                passed=score >= .9 and matched / changed >= .8 and faults == detected)


def main():
    b.verify()
    key = b.load(b.HERE / 'calibration_key.json')
    scores = {}
    for coder in ('1', '2'):
        path = b.HERE / 'calibration_reviews' / ('ratings_' + coder + '.json')
        rows = b.load(path)
        if isinstance(rows, dict):
            rows = rows['ratings']
        scores[coder] = assess(rows, key)
    online = []
    path = b.HERE / 'calibration_run/results.json'
    if path.exists():
        for record in b.load(path)['records']:
            online.append(dict(repetition=record['repetition'], status=record['status'],
                               assessment=assess(record['ratings'], key) if record['status'] == 'ok' else None))
    b.save(b.HERE / 'calibration_gate.json', dict(protocol_sha256=b.sha(b.HERE / 'protocol.json'),
           decision='proceed' if all(r['passed'] for r in scores.values()) else 'stop',
           codex=scores, backend_repeats=online,
           interpretation='Author-fixture construct check, not semantic ground truth or human validity.'))
    print({c: (r['overall_agreement'], r['correct_contrast_pairs'], r['detected_faults'], r['passed'])
           for c, r in scores.items()})


if __name__ == '__main__':
    main()
