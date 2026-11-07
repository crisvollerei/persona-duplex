"""Offline reconciliation of fixed assignments, source bytes, streams and quotes."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=HERE / 'run')
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    data = args.data
    checks = []
    def check(name, passed):
        checks.append({'check': name, 'passed': bool(passed)})
        if not passed: raise AssertionError(name)
    protocol = analyze.load(data/'protocol.json')
    plan = analyze.load(data/'execution_plan.json')
    results = analyze.load(data/'results.json')
    raw = json.loads(gzip.decompress((data/'raw_calls.json.gz').read_bytes()))
    scorer = analyze.load_scorer(data, protocol)
    check('fixed_plan_digest', scorer.digest(plan) == protocol['execution_plan_digest'])
    check('fixed_runner_bytes', analyze.sha(data/'runner_snapshot.py') == protocol['runner_sha256'])
    for relative, expected in protocol['source_hashes'].items():
        check('archived_source:' + relative, analyze.sha(data/'implementation_snapshot'/relative) == expected)
    check('results_protocol_hash', results['protocol_hash'] == analyze.sha(data/'protocol.json'))
    check('results_plan_hash', results['plan_hash'] == analyze.sha(data/'execution_plan.json'))
    check('raw_hash', results['raw_calls_sha256'] == analyze.sha(data/'raw_calls.json.gz'))
    assigned = {job['trial_id']:job for job in plan}
    records = {row['trial_id']:row for row in results['records']}
    check('assigned_ids_unique', len(assigned) == len(plan) == 150)
    check('observed_ids_unique', len(records) == len(results['records']) == 150)
    check('assignment_complete', set(assigned) == set(records))
    call_map = {call['trial_id']+'/'+call['call_id']:call for call in raw}
    check('request_ids_unique', len(call_map) == len(raw) == results['calls_count'] == 210)
    check('budget_not_exceeded', len(raw) <= protocol['hard_api_request_limit'])
    for call in raw:
        check('request_success:' + call['trial_id']+'/'+call['call_id'], call['status'] == 'ok')
        check('requested_after_freeze:' + call['trial_id']+'/'+call['call_id'], call['started_at'] >= protocol['fixed_at'])
        if call['mode'] == 'stream':
            check('raw_chunks:' + call['trial_id']+'/'+call['call_id'], ''.join(call['chunks']) == call['output'])
    # Load the parser in an archive-only package, never today's framework.
    spec = importlib.util.spec_from_file_location('_generalization_archive', data/'implementation_snapshot/persona/__init__.py',
                                                submodule_search_locations=[str(data/'implementation_snapshot/persona')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from importlib import import_module
    StreamParser = import_module('_generalization_archive.stream').StreamParser
    scored = streams = quotes = 0
    for ident, row in records.items():
        calls = [call_map[key] for key in row['call_ids']]
        check('call_membership:' + ident, all(call['trial_id'] == ident for call in calls))
        if row['experiment'] == 'learned_update':
            observed = [item['turn'] for item in row['training']] + [item['turn'] for item in row['probes']]
            scored_pairs = [(item['turn'], item['target'], item['evaluation']) for item in row['probes']]
            check('workflow_shape:' + ident, len(row['training']) == 4 and len(row['stages']) == 2 and len(row['probes']) == 3)
            check('actual_view_rebuilt:' + ident, row['rebuild_equal'] is True)
            check('reopened_ledger:' + ident, row['reopened_ledger_equal'] is True)
            check('traits_unchanged:' + ident, row['traits_unchanged'] is True)
            training = {item['turn']['turn_id']:item['turn']['user'] for item in row['training']}
            for stage in row['stages']:
                for judgment in stage['judgments']:
                    for quote in judgment.get('evidence_quotes', []):
                        check('complete_user_quote:' + ident + '/' + quote['turn_id'], training.get(quote['turn_id']) == quote['quote'])
                        quotes += 1
        else:
            observed = [row['turn']]
            scored_pairs = [(row['turn'], row['target'], row['evaluation'])]
        generated = [call for call in calls if call['mode'] == 'stream']
        check('foreground_count:' + ident, len(observed) == len(generated))
        for index, (turn, call) in enumerate(zip(observed, generated)):
            check('one_foreground_call:' + ident + str(index), turn['calls_count'] == 1)
            check('saved_delivery:' + ident + str(index), ''.join(turn['delivered_chunks']) == turn['say'])
            if row.get('mode') == 'plain':
                check('plain_raw:' + ident, call['output'] == turn['say'])
            else:
                parser = StreamParser()
                pieces = []
                for chunk in call['chunks']: pieces.extend(parser.feed(chunk))
                parsed = parser.finish()
                check('archive_parser:' + ident + str(index), parsed.say == turn['say'] and pieces == turn['delivered_chunks'])
                streams += 1
        for turn, target, saved in scored_pairs:
            check('frozen_score:' + ident, scorer.score(turn['say'], target) == saved)
            scored += 1
    check('all_scored_outputs', scored == 162)
    check('all_framework_streams', streams == 138)
    check('all_training_quotes', quotes == 24)
    review_inputs = analyze.load(HERE/'format_failure_review_inputs.json')
    review_protocol = analyze.load(HERE/'format_failure_review_protocol.json')
    review_key = analyze.load(HERE/'format_failure_review_key.json')
    ratings = analyze.load(HERE/'format_failure_review_ratings.json')
    check('supplement_fixed_inputs', analyze.sha(HERE/'format_failure_review_inputs.json') == review_protocol['inputs_sha256'])
    check('supplement_ids', {item['id'] for item in review_inputs} == set(review_key) == {item['id'] for item in ratings})
    check('supplement_unique', len(ratings) == len({item['id'] for item in ratings}) == 27)
    failures = {row['trial_id'] for row in records.values() if row['experiment'] != 'learned_update' and not row['evaluation']['strict_correct']}
    check('all_strict_failures_reviewed', failures == {entry['trial_id'] for entry in review_key.values()})
    review_summary = dict(timing='post_hoc', reviewed=len(ratings),
        answer_supported=dict(Counter(item['answer_supported'] for item in ratings)),
        self_contradictions=sum(item['self_contradiction'] for item in ratings),
        scope='Requested conclusion only, one condition-anonymous Codex reviewer, not human validation or full-text truth')
    for path in HERE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.png', '.pyc'): continue
        content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        text = content.decode('utf-8')
        # Authentication fields may occur in source; nonempty actual credentials must not.
        check('no_credential_pattern:' + str(path.relative_to(HERE)),
              re.search(r'sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY', text) is None)
    report = dict(passed=len(checks), failed=0, scored_outputs=scored, framework_streams_replayed=streams,
                  full_user_quotes_checked=quotes, review_summary=review_summary,
                  meaning='Artifact agreement only; not a claim all experimental trials succeeded', checks=checks)
    if args.write_report:
        analyze.dump(HERE/'audit_report.json', report)
        analyze.dump(HERE/'format_failure_review_summary.json', review_summary)
    manifest = HERE/'artifact_manifest.json'
    manifest_count = 0
    if manifest.exists():
        for item in analyze.load(manifest)['files']:
            if analyze.sha(HERE/item['path']) != item['sha256']: raise AssertionError('Publication byte mismatch: ' + item['path'])
            manifest_count += 1
    print(json.dumps(dict(passed=len(checks), failed=0, publication_hashes=manifest_count,
                          reviewed=review_summary['reviewed']), ensure_ascii=False))


if __name__ == '__main__': main()
