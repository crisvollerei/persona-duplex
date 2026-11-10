"""Frozen calibration and four-turn situated dialogue study; never retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
import mechanism as m

HERE = Path(__file__).resolve().parent
MODEL = 'deepseek-chat'
ENDPOINT = 'https://api.deepseek.com/v1/chat/completions'
MAX_CALLS = 194


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    secret = os.environ.get('PERSONA_API_KEY')
    if secret and secret in raw:
        raise ValueError('credential in output')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(raw, encoding='utf-8', newline='\n')
    os.replace(temp, path)


def freeze():
    import materials
    if (HERE / 'protocol.json').exists():
        raise ValueError('already frozen')
    save(HERE / 'cases.json', materials.cases())
    save(HERE / 'profiles.json', materials.profiles())
    source = list(materials.calibration_items())
    random.Random(2026111021).shuffle(source)
    items, key = [], {}
    for i, row in enumerate(source, 1):
        public = deepcopy(row['public'])
        alias = f'K{i:03}'
        key[alias] = dict(original_id=public['id'], expected=row['expected'])
        for field in ('pair_id', 'contrast', 'probe', 'ambiguous', 'gate_eligible'):
            if field in row:
                key[alias][field] = deepcopy(row[field])
        public['id'] = alias
        items.append(public)
    save(HERE / 'calibration_inputs.json', items)
    save(HERE / 'calibration_key.json', key)
    files = ('benchmark.py', 'mechanism.py', 'test_mechanism.py', 'materials.py', 'gate.py', 'prepare_reviews.py',
             'rubric.md', 'protocol.md', 'cases.json', 'profiles.json',
             'calibration_inputs.json', 'calibration_key.json')
    missing = [f for f in files if not (HERE / f).is_file()]
    if missing:
        raise ValueError('freeze inputs missing: ' + ', '.join(missing))
    save(HERE / 'protocol.json', dict(version=2, frozen_at=now(),
         git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
         hashes={f: sha(HERE / f) for f in files}, max_calls=MAX_CALLS,
         planned_calibration_calls=2, planned_generation_calls=192,
         temperature=.7, server_seed=None, retries=0, workers=3, timeout_s=90,
         calibration_instruction='Full rubric.md, byte-identical to automated reviewers'))


def verify():
    protocol = load(HERE / 'protocol.json')
    for file, digest in protocol['hashes'].items():
        if sha(HERE / file) != digest:
            raise ValueError('frozen input changed: ' + file)
    return protocol


class Client:
    def __init__(self, folder, limit, *, prior_calls=0, prior_errors=0):
        self.folder, self.limit = Path(folder), min(limit, MAX_CALLS-prior_calls)
        self.calls, self.errors = 0, prior_errors
        self.lock = threading.Lock()

    def call(self, id, messages, max_tokens):
        with self.lock:
            if self.calls >= self.limit or self.errors >= 3:
                return dict(id=id, status='not_called', reason='budget_or_error_stop')
            self.calls += 1
        row = dict(id=id, status='running', started_at=now(), endpoint=ENDPOINT,
                   request=dict(model=MODEL, messages=messages, temperature=.7,
                                max_tokens=max_tokens, stream=False), timeout_s=90)
        path = self.folder / 'calls' / (id + '.json')
        save(path, row)
        start = time.monotonic()
        try:
            req = urllib.request.Request(ENDPOINT, data=json.dumps(row['request']).encode(),
                headers={'Content-Type':'application/json',
                         'Authorization':'Bearer ' + os.environ['PERSONA_API_KEY']}, method='POST')
            with urllib.request.urlopen(req, timeout=90) as response:
                payload = json.loads(response.read().decode())
            row['response'] = payload
            choice = payload['choices'][0]
            content = choice['message'].get('content')
            if choice.get('finish_reason') != 'stop' or not isinstance(content, str) or not content.strip():
                raise ValueError('incomplete output')
            row.update(status='ok', output=content)
        except Exception as exc:
            with self.lock:
                self.errors += 1
            row.update(status='error', error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                row['http_status'] = exc.code
                exc.close()
        finally:
            row.update(elapsed_s=time.monotonic()-start, finished_at=now())
            save(path, row)
        return row


def archive(folder):
    folder = Path(folder)
    calls = [load(f) for f in sorted((folder / 'calls').glob('*.json'))]
    raw = json.dumps(calls, ensure_ascii=False).encode()
    secret = os.environ.get('PERSONA_API_KEY', '')
    if secret and secret.encode() in raw:
        raise ValueError('credential in archive')
    (folder / 'raw_calls.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    return calls


def label_messages(items):
    # Do not abbreviate this instruction: reviewers read exactly this file.
    return [dict(role='system', content=(HERE / 'rubric.md').read_text(encoding='utf-8')),
            dict(role='user', content=json.dumps(items, ensure_ascii=False))]


def calibration(folder):
    items = load(HERE / 'calibration_inputs.json')
    client = Client(folder, 2)
    results = []
    for repetition in (1, 2):
        shuffled = list(items)
        random.Random(2026111030+repetition).shuffle(shuffled)
        response = client.call('calibration-' + str(repetition), label_messages(shuffled), 5000)
        record = dict(repetition=repetition, status=response['status'], ratings=[])
        if response['status'] == 'ok':
            try:
                record['ratings'] = m.validate_labels(m.strict_json(response['output']), [r['id'] for r in items])
            except (ValueError, TypeError, KeyError):
                record['status'] = 'invalid_labels'
        results.append(record)
        print('Calibration', repetition, record['status'], flush=True)
    save(Path(folder) / 'results.json', dict(protocol_sha256=sha(HERE / 'protocol.json'), records=results))
    archive(folder)


def episode(job, client, folder):
    folder = Path(folder)
    case = next(c for c in load(HERE / 'cases.json') if c['id'] == job['case'])
    profile = next(p for p in load(HERE / 'profiles.json') if p['id'] == job['profile'])
    state = deepcopy(case['initial_state'])
    allowed_keys = set(state)
    history, records = [], []
    for turn in case['turns']:
        before = deepcopy(state)
        updates = turn.get('user_update', turn.get('confirmed_updates', {}))
        state = m.apply_confirmed_updates(state, updates, allowed_keys)
        after_user = deepcopy(state)
        data = m.payload(case, turn, profile, job['rep'], state, history)
        planned = m.choose(data)
        call_id = job['id'] + '-' + str(turn['id'])
        response = client.call(call_id, m.messages(data, job['method']), 550)
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
        save(folder / 'checkpoints' / (job['id'] + '.json'), dict(**job, turns=records))
    result = dict(**job, initial_state=deepcopy(case['initial_state']), turns=records, final_state=state,
                  complete=all(r['status'] == 'ok' for r in records))
    save(folder / 'episodes' / (job['id'] + '.json'), result)
    return result


def run(folder):
    gate_path = HERE / 'calibration_gate.json'
    gate = load(gate_path) if gate_path.exists() else {}
    jobs = []
    cases, profiles = load(HERE / 'cases.json'), load(HERE / 'profiles.json')
    for c in cases:
        if len(c['turns']) != 4:
            raise ValueError('four turns per case required')
        for p in profiles:
            for method in ('program', 'prompt'):
                for rep in (1, 2):
                    jobs.append(dict(id=f"{c['id']}-{p['id']}-{method}-{rep}",
                                     case=c['id'], profile=p['id'], method=method, rep=rep))
    assert len(jobs) == 48
    random.Random(2026111040).shuffle(jobs)
    save(Path(folder) / 'execution_plan.json', jobs)
    prior = [load(p) for p in sorted((HERE / 'calibration_run/calls').glob('*.json'))]
    client = Client(folder, 192, prior_calls=len(prior), prior_errors=sum(r['status']=='error' for r in prior))
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(episode, job, client, folder) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            print(f'Completed {len(results)}/{len(jobs)} episodes; requests {client.calls}', flush=True)
    ordered = sorted(results, key=lambda r:r['id'])
    save(Path(folder) / 'results.json', dict(protocol_sha256=sha(HERE / 'protocol.json'),
         records=[r for e in ordered for r in e['turns']], episodes=ordered,
         semantic_eligible=gate.get('semantic_eligible'),
         actual_calls=client.calls, prior_stage_calls=len(prior)))
    archive(folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'calibration', 'run'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze()
        return
    verify()
    if not args.run or args.out is None:
        raise SystemExit('--run and a new --out directory are required')
    if args.out.exists():
        raise SystemExit('use a new output directory')
    args.out.mkdir(parents=True)
    save(args.out / 'manifest.json', dict(started_at=now(), stage=args.mode,
         protocol_sha256=sha(HERE / 'protocol.json'), requested_model=MODEL, endpoint=ENDPOINT,
         python=platform.python_version(), workers=3 if args.mode == 'run' else 1,
         retries=0, server_seed=None, max_stage_calls=192 if args.mode == 'run' else 2,
         semantic_eligible=(load(HERE / 'calibration_gate.json').get('semantic_eligible')
                            if (HERE / 'calibration_gate.json').exists() else None)))
    if not os.environ.get('PERSONA_API_KEY'):
        os.environ['PERSONA_API_KEY'] = getpass.getpass('Temporary API key (not saved): ')
    (calibration if args.mode == 'calibration' else run)(args.out)


if __name__ == '__main__':
    main()
