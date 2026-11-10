"""Offline reconstruction of frozen requests, choices, and simulation states.

This module never calls a model or creates ratings. Replaying an archived response
does not create another experimental observation.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random

import benchmark as b
import mechanism as m


class Checks:
    def __init__(self):
        self.count = 0

    def equal(self, actual, expected, name):
        self.count += 1
        if actual != expected:
            # Never echo request/response content into diagnostics.
            raise AssertionError(name)

    def true(self, value, name):
        self.equal(bool(value), True, name)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.write_text(raw, encoding='utf-8', newline='\n')


def verify_sources(checks):
    protocol = b.verify()
    contract = b.load(b.HERE / 'review_context_protocol.json')
    checks.equal(contract['generation_protocol_sha256'], b.sha(b.HERE / 'protocol.json'),
                 'review contract generation protocol')
    for file, digest in contract['hashes'].items():
        checks.equal(b.sha(b.HERE / file), digest, 'review contract source ' + file)
    checks.equal(contract['effective_input_builder'], 'prepare_review_delivery.py', 'effective review input builder')
    return protocol, contract


def calls_for(stage, checks):
    folder = b.HERE / stage
    paths = sorted((folder / 'calls').glob('*.json'))
    calls = [b.load(p) for p in paths]
    for path, row in zip(paths, calls):
        checks.equal(path.stem, row['id'], stage + ' call filename association')
    archive = json.loads(gzip.decompress((folder / 'raw_calls.json.gz').read_bytes()))
    checks.equal(calls, archive, stage + ' raw archive association')
    ids = [r['id'] for r in calls]
    checks.equal(len(ids), len(set(ids)), stage + ' unique call IDs')
    return {r['id']: r for r in calls}


def check_response(row, checks):
    if row['status'] == 'ok':
        choice = row['response']['choices'][0]
        checks.equal(choice['finish_reason'], 'stop', 'complete raw response ' + row['id'])
        checks.equal(choice['message'].get('content'), row['output'], 'raw output association ' + row['id'])
    checks.true(row['status'] in ('ok', 'error'), 'finished call ' + row['id'])


def replay_calibration(checks):
    result = b.load(b.HERE / 'calibration_run/results.json')
    items = b.load(b.HERE / 'calibration_inputs.json')
    checks.equal(result['protocol_sha256'], b.sha(b.HERE / 'protocol.json'), 'calibration protocol')
    checks.equal(len(result['records']), 2, 'two planned calibration batches')
    calls = calls_for('calibration_run', checks)
    for record in result['records']:
        repetition = record['repetition']
        call_id = 'calibration-' + str(repetition)
        shuffled = list(items)
        random.Random(2026111030 + repetition).shuffle(shuffled)
        if call_id not in calls:
            checks.equal(record, dict(repetition=repetition, status='not_called', ratings=[]),
                         'uncalled calibration batch ' + call_id)
            continue
        call = calls[call_id]
        expected_request = dict(model=b.MODEL, messages=b.label_messages(shuffled),
                                temperature=.7, max_tokens=5000, stream=False)
        checks.equal(call['request'], expected_request, 'calibration exact request ' + call_id)
        check_response(call, checks)
        expected = dict(repetition=repetition, status=call['status'], ratings=[])
        if call['status'] == 'ok':
            try:
                expected['ratings'] = m.validate_labels(m.strict_json(call['output']), [r['id'] for r in items])
            except (ValueError, TypeError, KeyError):
                expected['status'] = 'invalid_labels'
        checks.equal(record, expected, 'calibration raw parse ' + call_id)
    checks.true(set(calls) <= {'calibration-' + str(r['repetition']) for r in result['records']},
                'no unassociated calibration calls')
    return len(calls)


def expected_jobs(cases, profiles):
    jobs = []
    for case in cases:
        for profile in profiles:
            for method in ('program', 'prompt'):
                for rep in (1, 2):
                    jobs.append(dict(id=f"{case['id']}-{profile['id']}-{method}-{rep}",
                                     case=case['id'], profile=profile['id'], method=method, rep=rep))
    random.Random(2026111040).shuffle(jobs)
    return jobs


def replay_main(checks):
    result = b.load(b.HERE / 'run/results.json')
    cases, profiles = b.load(b.HERE / 'cases.json'), b.load(b.HERE / 'profiles.json')
    jobs = expected_jobs(cases, profiles)
    checks.equal(len(jobs), 48, '48 planned episodes')
    checks.equal(b.load(b.HERE / 'run/execution_plan.json'), jobs, 'frozen execution order')
    checks.equal(result['protocol_sha256'], b.sha(b.HERE / 'protocol.json'), 'main protocol')
    calls = calls_for('run', checks)
    checks.equal(result['actual_calls'], len(calls), 'actual main call count')
    by_episode = {r['id']: r for r in result['episodes']}
    checks.equal(set(by_episode), {r['id'] for r in jobs}, 'complete episode ID set')
    checks.equal(len(result['episodes']), len(jobs), 'unique episodes')
    expected_episodes, call_ids = [], set()
    for job in sorted(jobs, key=lambda r:r['id']):
        case = next(c for c in cases if c['id'] == job['case'])
        profile = next(p for p in profiles if p['id'] == job['profile'])
        state = deepcopy(case['initial_state'])
        allowed_keys = set(state)
        history, records = [], []
        checks.equal(len(case['turns']), 4, 'four authored turns ' + job['id'])
        for turn in case['turns']:
            before = deepcopy(state)
            updates = turn.get('user_update', turn.get('confirmed_updates', {}))
            state = m.apply_confirmed_updates(state, updates, allowed_keys)
            after_user = deepcopy(state)
            data = m.payload(case, turn, profile, job['rep'], state, history)
            planned = m.choose(data)
            call_id = job['id'] + '-' + str(turn['id'])
            call = calls.get(call_id)
            if call is not None:
                call_ids.add(call_id)
                checks.equal(call['request'], dict(model=b.MODEL, messages=m.messages(data, job['method']),
                              temperature=.7, max_tokens=550, stream=False), 'exact generated request ' + call_id)
                check_response(call, checks)
            response = call or dict(id=call_id, status='not_called', reason='budget_or_error_stop')
            row = dict(job, id=call_id, turn=turn['id'], user=turn['user'], status=response['status'], payload=data,
                       state_before=before, state_after_user=after_user, user_update=deepcopy(updates),
                       program_selected_id=planned, static_ablation_id=m.choose(data, contextual=False),
                       model_reported_id=None, executed_id=None, say=None,
                       allowed=False, request_scope_match=False, rule_match=False,
                       execution_reason='no_valid_model_proposal', execution_error='no_valid_model_proposal')
            if response['status'] == 'ok':
                try:
                    parsed = m.parse_reply(response['output'], data)
                    action = next(a for a in data['candidates'] if a['id'] == parsed['action_id'])
                    row.update(model_reported_id=parsed['action_id'], say=parsed['say'],
                               allowed=parsed['action_id'] in data['allowed_actions'],
                               request_scope_match=not data['requested'] or action['kind'] in data['request_scope'],
                               rule_match=parsed['action_id'] == planned)
                    execution = m.execute_effects(data, parsed['action_id'], state, allowed_keys)
                    state = execution['state']
                    row.update(executed_id=execution['executed_id'], execution_reason=execution['reason'],
                               execution_error=None if execution['accepted'] else execution['reason'])
                except (ValueError, TypeError, KeyError):
                    row['status'] = 'invalid_response'
            history.append(dict(role='user', content=turn['user']))
            if row['say'] is not None:
                history.append(dict(role='assistant', content=row['say']))
            row['state_after_action'] = deepcopy(state)
            records.append(row)
        expected = dict(job, initial_state=deepcopy(case['initial_state']), turns=records,
                        final_state=state, complete=all(r['status'] == 'ok' for r in records))
        checks.equal(by_episode[job['id']], expected, 'IDs/states/history/final episode ' + job['id'])
        checks.equal(b.load(b.HERE / 'run/episodes' / (job['id'] + '.json')), expected,
                     'individual episode ' + job['id'])
        checks.equal(b.load(b.HERE / 'run/checkpoints' / (job['id'] + '.json')), dict(job, turns=records),
                     'final checkpoint ' + job['id'])
        expected_episodes.append(expected)
    checks.equal(call_ids, set(calls), 'every main call mapped to an authored turn')
    checks.equal(result['episodes'], expected_episodes, 'ordered episodes')
    checks.equal(result['records'], [r for e in expected_episodes for r in e['turns']], 'flat turn association')
    checks.equal(len(result['records']), 192, '192 planned generation units')
    gate = b.load(b.HERE / 'calibration_gate.json')
    checks.equal(result['semantic_eligible'], gate['semantic_eligible'], 'recorded semantic branch')
    return len(calls), len(expected_episodes), len(result['records'])


def replay():
    checks = Checks()
    verify_sources(checks)
    calibration_calls = replay_calibration(checks)
    main_calls, episodes, units = replay_main(checks)
    result = b.load(b.HERE / 'run/results.json')
    checks.equal(result['prior_stage_calls'], calibration_calls, 'carried calibration call count')
    checks.true(calibration_calls + main_calls <= b.MAX_CALLS, 'overall request ceiling')
    return dict(status='passed', checks=checks.count,
                protocol_sha256=b.sha(b.HERE / 'protocol.json'),
                review_context_protocol_sha256=b.sha(b.HERE / 'review_context_protocol.json'),
                raw_archive_sha256={s: b.sha(b.HERE / s / 'raw_calls.json.gz') for s in ('calibration_run', 'run')},
                actual_calibration_calls=calibration_calls, actual_generation_calls=main_calls,
                reconstructed_episodes=episodes, reconstructed_units=units,
                interpretation='Archived-response reconstruction only; no API calls, new samples, or semantic correctness guarantee.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=b.HERE / 'replay_report.json')
    args = parser.parse_args()
    report = replay()
    save(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
