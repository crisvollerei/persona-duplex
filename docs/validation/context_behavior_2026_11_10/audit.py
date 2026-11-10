"""Offline material audit and optional one-time artifact sealing."""
import argparse
import gzip
import json
import re
from pathlib import Path
import benchmark as b
import mechanism as m


def audit(*, sealing=False):
    p = b.verify()
    checks = []
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    key = b.load(b.HERE / 'calibration_key.json')
    items = b.load(b.HERE / 'calibration_inputs.json')
    check(len(items) == len(key) == 24 and {i['id'] for i in items} == set(key), '24 unique public items')
    for i in items:
        check(set(i) == {'id', 'context', 'user', 'reply'}, 'no expected labels in public ' + i['id'])
    data = b.load(b.HERE / 'calibration_run/results.json')
    check(data['protocol_sha256'] == b.sha(b.HERE / 'protocol.json'), 'result protocol hash')
    calls = [b.load(f) for f in sorted((b.HERE / 'calibration_run/calls').glob('*.json'))]
    archived = json.loads(gzip.decompress((b.HERE / 'calibration_run/raw_calls.json.gz').read_bytes()))
    check(calls == archived and len(calls) == 2, 'exactly two calls and identical compressed archive')
    tokens = 0
    for call, record in zip(calls, data['records']):
        check(call['status'] == record['status'] == 'ok', 'complete call ' + call['id'])
        check(call['request']['messages'][0]['content'] == m.LABEL_SYSTEM, 'frozen instruction ' + call['id'])
        sent = json.loads(call['request']['messages'][1]['content'])
        check(sorted(sent, key=lambda r:r['id']) == sorted(items, key=lambda r:r['id']), 'identical inputs ' + call['id'])
        parsed = m.validate_labels(m.strict_json(call['output']), list(key))
        check(parsed == record['ratings'], 'raw parsing association ' + call['id'])
        tokens += call['response']['usage']['total_tokens']
    check(tokens == 6020, '6020 recorded API tokens')
    for c in ('1', '2'):
        ratings = b.load(b.HERE / 'calibration_reviews' / ('ratings_' + c + '.json'))['ratings']
        m.validate_labels({'ratings': ratings}, list(key))
        check(len(ratings) == 24, 'complete independent coding ' + c)
    gate = b.load(b.HERE / 'calibration_gate.json')
    check(gate['decision'] == 'stop' and not all(r['passed'] for r in gate['codex'].values()), 'frozen stop decision')
    check(not (b.HERE / 'run').exists(), 'main experiment not executed')
    summary = b.load(b.HERE / 'summary.json')
    check(summary['main'] is None and summary['costs']['actual_requests'] == 2, 'summary denominators')
    check(summary['costs']['total_tokens'] == tokens, 'summary usage')
    check(summary['gate'] == gate, 'summary gate association')
    secret = re.compile(rb'(?i)(?:sk-[a-z0-9_-]{24,}|Bearer\s+[a-z0-9._-]{20,})')
    for path in sorted(b.HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        raw = path.read_bytes()
        if path.suffix == '.gz':
            raw = gzip.decompress(raw)
        check(not secret.search(raw), 'no credential pattern ' + path.relative_to(b.HERE).as_posix())
        if path.suffix == '.md':
            text = path.read_text(encoding='utf-8')
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                target = target.split('#', 1)[0]
                if target and not re.match(r'^[a-z]+:', target):
                    if sealing and path.parent == b.HERE and target in ('audit_report.json', 'artifact_manifest.json'):
                        continue
                    check((path.parent / target).exists(), 'relative link ' + target)
    return dict(status='passed', checks=len(checks), scope='Material/association audit, not semantic validity.',
                protocol_sha256=b.sha(b.HERE / 'protocol.json'), actual_online_requests=2, main_outputs=0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seal', action='store_true'); a = parser.parse_args()
    report = audit(sealing=a.seal)
    manifest = b.HERE / 'artifact_manifest.json'
    if a.seal:
        if manifest.exists():
            raise ValueError('already sealed')
        b.save(b.HERE / 'audit_report.json', report)
        files = {p.relative_to(b.HERE).as_posix(): dict(bytes=p.stat().st_size, sha256=b.sha(p))
                 for p in sorted(b.HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
                 and p.name != 'artifact_manifest.json'}
        b.save(manifest, dict(algorithm='SHA-256', files=files))
    if manifest.exists():
        entries = b.load(manifest)['files']
        actual = {p.relative_to(b.HERE).as_posix() for p in b.HERE.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p != manifest}
        assert actual == set(entries)
        for name, record in entries.items():
            path = b.HERE / name
            assert path.stat().st_size == record['bytes'] and b.sha(path) == record['sha256'], name
        report['verified_artifacts'] = len(entries)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
