"""Read-only offline provenance audit; --seal creates a delivery manifest once."""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
import random
import re
from urllib.parse import unquote

import benchmark as b
import gate
import mechanism as m
import replay as rp


SENSITIVE = re.compile(rb'(?i)(?:sk-[a-z0-9_-]{24,}|Bearer\s+[a-z0-9._-]{20,})')
HIDDEN = {'method', 'action_id', 'selected_action_id', 'program_selected_id', 'model_reported_id',
          'executed_id', 'draw', 'candidates', 'allowed_actions', 'request_scope', 'rules', 'labels'}


def object_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from object_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from object_keys(child)


def audit_reviews(checks, *, sealing):
    inputs = b.HERE / 'reviews/inputs.json'
    if not inputs.exists():
        checks.true(not sealing, 'review inputs required before sealing')
        return dict(available=False, final_ratings_available=[])
    rows = b.load(b.HERE / 'run/results.json')['records']
    profiles = {p['id']: p for p in b.load(b.HERE / 'profiles.json')}
    units, source_mapping = {}, {}
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
        source_mapping[row['id']] = key
    keys = list(units)
    random.Random(202611103).shuffle(keys)
    aliases = {key: f'R{i:03}' for i, key in enumerate(keys, 1)}
    expected = [dict(id=aliases[key], **units[key]) for key in keys]
    actual = b.load(inputs)
    checks.equal(actual, expected, 'anonymous inputs rebuilt from executed records')
    checks.equal(b.load(b.HERE / 'review_key.json'), {id: aliases[key] for id, key in source_mapping.items()},
                 'review mapping to actual generated text')
    for row in actual:
        checks.equal(set(row), {'id', 'context', 'user', 'reply', 'persona'}, 'public review fields ' + row['id'])
        checks.true(not (set(object_keys(row)) & HIDDEN), 'hidden treatment metadata ' + row['id'])
    packets = [b.load(p) for p in sorted((b.HERE / 'reviews/packets').glob('packet_*.json'))]
    checks.equal([r for packet in packets for r in packet], actual, 'review packet association')
    available = []
    for coder in ('1', '2'):
        path = b.HERE / 'reviews' / ('ratings_' + coder + '.json')
        if path.exists():
            ratings = gate.ratings(path)
            m.validate_labels({'ratings': ratings}, [r['id'] for r in actual], main=True)
            checks.equal(len(ratings), len(actual), 'final review ID coverage ' + coder)
            available.append(coder)
        else:
            checks.true(not sealing, 'final review required before sealing ' + coder)
    return dict(available=True, unique_units=len(actual), mapped_generation_units=len(source_mapping),
                final_ratings_available=available)


def audit_summary(checks, actual_calls, total_tokens, reviews):
    path = b.HERE / 'summary.json'
    if not path.exists():
        return False
    summary = b.load(path)
    checks.equal(summary['protocol_sha256'], b.sha(b.HERE / 'protocol.json'), 'summary generation protocol')
    checks.equal(summary['review_context_protocol_sha256'], b.sha(b.HERE / 'review_context_protocol.json'),
                 'summary review context protocol')
    checks.equal(summary['costs']['actual_requests'], actual_calls, 'summary actual requests')
    checks.equal(summary['costs']['total_tokens'], total_tokens, 'summary usage tokens')
    saved_gate = b.load(b.HERE / 'calibration_gate.json')
    checks.equal(summary['gate'], saved_gate, 'summary gate association')
    main = summary['main']
    checks.equal(main['planned_outputs'], 192, 'summary output denominator')
    checks.equal(main['planned_episodes'], 48, 'summary episode denominator')
    checks.equal(main['scenario_template_instances'], 6, 'summary template-instance denominator')
    checks.equal(main['semantic_eligible'], saved_gate['semantic_eligible'], 'summary semantic interpretation branch')
    records = b.load(b.HERE / 'run/results.json')['records']
    for method in ('program', 'prompt'):
        selected = [r for r in records if r['method'] == method]
        actual = dict(planned=len(selected), ok=sum(r['status'] == 'ok' for r in selected),
                      allowed=sum(r['allowed'] for r in selected),
                      request_scope_match=sum(r['request_scope_match'] for r in selected),
                      rule_match=sum(r['rule_match'] for r in selected),
                      executed=sum(r['executed_id'] is not None for r in selected))
        for field, count in actual.items():
            checks.equal(main['execution_by_method'][method][field], count, 'summary execution ' + method + '/' + field)
    for coder in reviews['final_ratings_available']:
        checks.equal(main['reviews'][coder]['status'], 'complete', 'summary final review refreshed ' + coder)
        checks.equal(main['reviews'][coder]['unique_review_units'], reviews['unique_units'],
                     'summary unique review denominator ' + coder)
        checks.equal(main['reviews'][coder]['mapped_outputs'], reviews['mapped_generation_units'],
                     'summary mapped review denominator ' + coder)
    return True


def check_links_and_secrets(checks, *, sealing):
    files = [p for p in sorted(b.HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    for path in files:
        if sealing:
            checks.true(path.suffix != '.tmp', 'no unfinished output file ' + path.relative_to(b.HERE).as_posix())
        raw = path.read_bytes()
        if path.suffix == '.gz':
            raw = gzip.decompress(raw)
        checks.true(not SENSITIVE.search(raw), 'credential-pattern scan ' + path.relative_to(b.HERE).as_posix())
        if path.suffix != '.md':
            continue
        text = raw.decode('utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            target = target.strip().split('#', 1)[0].strip('<>')
            if not target or re.match(r'^[a-zA-Z]+:', target):
                continue
            destination = path.parent / unquote(target)
            # During preparation the one-time seal artifacts do not exist yet.
            pending = path.parent == b.HERE and target in ('audit_report.json', 'artifact_manifest.json')
            if pending and sealing and not destination.exists():
                continue
            if not sealing and not destination.exists():
                continue
            checks.true(destination.exists(), 'relative link ' + path.relative_to(b.HERE).as_posix() + ' -> ' + target)
    return len(files)


def audit(*, sealing=False):
    checks = rp.Checks()
    protocol, review_contract = rp.verify_sources(checks)
    replayed = rp.replay()
    if (b.HERE / 'replay_report.json').exists():
        checks.equal(b.load(b.HERE / 'replay_report.json'), replayed, 'replay report association')
    else:
        checks.true(not sealing, 'replay report required before sealing')
    calibration = b.load(b.HERE / 'calibration_run/results.json')
    key = b.load(b.HERE / 'calibration_key.json')
    saved_gate = b.load(b.HERE / 'calibration_gate.json')
    assessments = {}
    for coder in ('1', '2'):
        rows = gate.ratings(b.HERE / 'calibration_reviews' / ('ratings_' + coder + '.json'))
        assessments[coder] = gate.assess(rows, key)
        checks.equal(saved_gate['assessments'][coder], assessments[coder], 'calibration decision score ' + coder)
    checks.equal(saved_gate['semantic_eligible'], all(v['passed'] for v in assessments.values()), 'semantic branch decision')
    checks.equal(saved_gate['protocol_sha256'], b.sha(b.HERE / 'protocol.json'), 'gate protocol binding')
    for record, saved in zip(calibration['records'], saved_gate['backend_repeats']):
        assessment = gate.assess(record['ratings'], key) if record['status'] == 'ok' else None
        checks.equal(saved, dict(repetition=record['repetition'], status=record['status'], assessment=assessment),
                     'calibration repeat score ' + str(record['repetition']))
    stages, total_tokens, calls_count = {}, 0, 0
    for stage in ('calibration_run', 'run'):
        calls = rp.calls_for(stage, checks)
        for row in calls.values():
            checks.true('Authorization' not in row and 'headers' not in row, 'no saved authorization header ' + row['id'])
            checks.equal(row['request']['temperature'], .7, 'declared temperature ' + row['id'])
            checks.equal(row['request']['model'], b.MODEL, 'declared requested model ' + row['id'])
            checks.true(datetime.fromisoformat(row['started_at']) >= datetime.fromisoformat(protocol['frozen_at']),
                        'generation sources frozen before request ' + row['id'])
            if stage == 'calibration_run':
                checks.equal(row['request']['messages'][0]['content'], (b.HERE / 'rubric.md').read_text(encoding='utf-8'),
                             'full visible rubric equality ' + row['id'])
            else:
                checks.true(datetime.fromisoformat(row['started_at']) >= datetime.fromisoformat(review_contract['registered_at']),
                            'review context registered before generation ' + row['id'])
            total_tokens += row.get('response', {}).get('usage', {}).get('total_tokens', 0)
        calls_count += len(calls)
        stages[stage] = dict(actual_calls=len(calls), statuses=dict(Counter(r['status'] for r in calls.values())))
        manifest = b.load(b.HERE / stage / 'manifest.json')
        checks.equal(manifest['protocol_sha256'], b.sha(b.HERE / 'protocol.json'), 'run manifest binding ' + stage)
        checks.equal(manifest['retries'], 0, 'no configured retries ' + stage)
    checks.true(calls_count <= 194, 'overall request budget')
    reviews = audit_reviews(checks, sealing=sealing)
    summary_available = audit_summary(checks, calls_count, total_tokens, reviews)
    if sealing:
        checks.true(summary_available, 'summary required before sealing')
        for file in ('paper.md', 'README.md', 'data_dictionary.md', 'requirements.txt'):
            checks.true((b.HERE / file).is_file(), 'completed research artifact ' + file)
    scanned = check_links_and_secrets(checks, sealing=sealing)
    return dict(status='passed', checks=checks.count,
                protocol_sha256=b.sha(b.HERE / 'protocol.json'),
                review_context_protocol_sha256=b.sha(b.HERE / 'review_context_protocol.json'),
                replay_checks=replayed['checks'], planned_episodes=48, planned_generation_units=192,
                actual_requests=calls_count, total_recorded_api_tokens=total_tokens, stages=stages,
                reviews=reviews, summary_available=summary_available, scanned_files=scanned,
                interpretation='Offline association and byte-integrity audit; not semantic validity, human preference, or new experimental observations.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    manifest = b.HERE / 'artifact_manifest.json'
    if args.seal and manifest.exists():
        raise ValueError('delivery already sealed; do not overwrite archived files')
    report = audit(sealing=args.seal)
    if args.seal:
        rp.save(b.HERE / 'audit_report.json', report)
        files = {p.relative_to(b.HERE).as_posix(): dict(bytes=p.stat().st_size, sha256=b.sha(p))
                 for p in sorted(b.HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
                 and p != manifest}
        rp.save(manifest, dict(algorithm='SHA-256', files=files))
    if manifest.exists():
        entries = b.load(manifest)['files']
        actual = {p.relative_to(b.HERE).as_posix() for p in b.HERE.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p != manifest}
        if set(entries) != actual:
            raise AssertionError('delivery manifest exact file coverage')
        for file, record in entries.items():
            path = b.HERE / file
            if path.stat().st_size != record['bytes'] or b.sha(path) != record['sha256']:
                raise AssertionError('delivery manifest mismatch ' + file)
        report['verified_artifacts'] = len(entries)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
