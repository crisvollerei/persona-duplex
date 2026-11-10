"""Frozen two-stage pilot. Calls are bounded, recorded, and never retried."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
ENDPOINT = 'https://api.deepseek.com/v1/chat/completions'
MODEL = 'deepseek-chat'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze():
    import materials
    if (HERE / 'protocol.json').exists():
        raise ValueError('already frozen')
    save(HERE / 'cases.json', materials.cases())
    save(HERE / 'profiles.json', materials.profiles())
    original = materials.calibration_items()
    shuffled = list(original)
    random.Random(20261110).shuffle(shuffled)
    items, key = [], {}
    for i, row in enumerate(shuffled, 1):
        public = dict(row['public'])
        alias = f'K{i:03}'
        key[alias] = dict(original_id=public['id'], expected=row['expected'])
        public['id'] = alias
        items.append(public)
    save(HERE / 'calibration_inputs.json', items)
    save(HERE / 'calibration_key.json', key)
    files = ('mechanism.py', 'benchmark.py', 'gate.py', 'materials.py', 'test_mechanism.py',
             'protocol.md', 'rubric.md', 'review_instructions.md', 'prepare_reviews.py', 'cases.json', 'profiles.json',
             'calibration_inputs.json', 'calibration_key.json')
    save(HERE / 'protocol.json', dict(version=1, frozen_at=now(),
         git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
         hashes={p: sha(HERE / p) for p in files}, max_calls=146,
         planned_calibration_calls=2, planned_main_calls=144,
         temperature=.7, server_seed=None, workers=3, retries=0, timeout_s=90))


def verify():
    p = load(HERE / 'protocol.json')
    for name, digest in p['hashes'].items():
        if sha(HERE / name) != digest:
            raise ValueError('frozen input changed: ' + name)
    return p


class Client:
    def __init__(self, folder, limit):
        self.folder, self.limit = folder, limit
        self.calls, self.errors = 0, 0
        self.lock = threading.Lock()

    def call(self, id, messages, max_tokens):
        with self.lock:
            if self.calls >= self.limit or self.errors >= 3:
                return dict(id=id, status='not_called')
            self.calls += 1
        row = dict(id=id, status='running', started_at=now(), endpoint=ENDPOINT,
                   request=dict(model=MODEL, messages=messages, temperature=.7,
                                max_tokens=max_tokens, stream=False))
        path = self.folder / 'calls' / (id + '.json')
        save(path, row)
        start = time.monotonic()
        req = urllib.request.Request(ENDPOINT,
            data=json.dumps(row['request']).encode(), headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + os.environ['PERSONA_API_KEY']}, method='POST')
        try:
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
            row.update(elapsed_s=time.monotonic() - start, finished_at=now())
            save(path, row)
        return row


def archive(folder):
    calls = [load(f) for f in sorted((folder / 'calls').glob('*.json'))]
    raw = json.dumps(calls, ensure_ascii=False).encode()
    secret = os.environ.get('PERSONA_API_KEY', '')
    if secret and secret.encode() in raw:
        raise ValueError('credential in compressed archive')
    (folder / 'raw_calls.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    return calls


def calibration(folder):
    items = load(HERE / 'calibration_inputs.json')
    client = Client(folder, 2)
    rows = []
    for rep in range(1, 3):
        shuffled = list(items)
        random.Random(20261110 + rep).shuffle(shuffled)
        response = client.call('calibration-' + str(rep), [dict(role='system', content=m.LABEL_SYSTEM),
            dict(role='user', content=json.dumps(shuffled, ensure_ascii=False))], 4200)
        result = dict(repetition=rep, status=response['status'], ratings=[])
        if response['status'] == 'ok':
            try:
                result['ratings'] = m.validate_labels(m.strict_json(response['output']), [i['id'] for i in items])
            except (ValueError, TypeError, KeyError):
                result['status'] = 'invalid_labels'
        rows.append(result)
        print('Calibration', rep, result['status'], flush=True)
    save(folder / 'results.json', dict(protocol_sha256=sha(HERE / 'protocol.json'), records=rows))
    archive(folder)


def trial(job, client, folder):
    cases = load(HERE / 'cases.json')
    profiles = load(HERE / 'profiles.json')
    case = next(c for c in cases if c['id'] == job['case'])
    variant = next(v for v in case['variants'] if v['id'] == job['variant'])
    profile = next(p for p in profiles if p['id'] == job['profile'])
    data = m.payload(case, variant, profile, job['rep'])
    response = client.call(job['id'], m.messages(data, job['method']), 550)
    result = dict(**job, status=response['status'], payload=data, selected_by_program=m.choose(data),
                  static_ablation=m.choose(data, contextual=False), action_id=None, say=None)
    if response['status'] == 'ok':
        try:
            parsed = m.parse_reply(response['output'], data)
            result.update(parsed)
            result['allowed'] = parsed['action_id'] in data['allowed_actions']
            result['rule_match'] = parsed['action_id'] == result['selected_by_program']
        except (ValueError, TypeError, KeyError):
            result['status'] = 'invalid_response'
    save(folder / 'records' / (job['id'] + '.json'), result)
    return result


def run(folder):
    gate = load(HERE / 'calibration_gate.json')
    if gate.get('decision') != 'proceed' or gate.get('protocol_sha256') != sha(HERE / 'protocol.json'):
        raise ValueError('calibration gate has not passed for this protocol')
    jobs = []
    for c in load(HERE / 'cases.json'):
        for v in c['variants']:
            for p in load(HERE / 'profiles.json'):
                for method in ('program', 'prompt'):
                    for rep in range(1, 4):
                        jobs.append(dict(id=f"{c['id']}-{v['id']}-{p['id']}-{method}-{rep}",
                                         case=c['id'], variant=v['id'], profile=p['id'], method=method, rep=rep))
    assert len(jobs) == 144
    random.Random(202611101).shuffle(jobs)
    save(folder / 'execution_plan.json', jobs)
    client = Client(folder, 144)
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(trial, job, client, folder) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            print(f'Completed {len(results)}/{len(jobs)}; calls {client.calls}', flush=True)
    save(folder / 'results.json', dict(protocol_sha256=sha(HERE / 'protocol.json'),
         records=sorted(results, key=lambda r: r['id']), actual_calls=client.calls))
    archive(folder)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('freeze', 'calibration', 'run'))
    p.add_argument('--out', type=Path)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if a.mode == 'freeze':
        freeze()
        return
    verify()
    if not a.run or a.out is None:
        raise SystemExit('--run and a new --out directory are required for online work')
    if a.out.exists():
        raise SystemExit('use a new output directory')
    a.out.mkdir(parents=True)
    save(a.out / 'manifest.json', dict(started_at=now(), protocol_sha256=sha(HERE / 'protocol.json'),
         stage=a.mode, python=platform.python_version(), endpoint=ENDPOINT, requested_model=MODEL,
         retries=0, server_seed=None, workers=3 if a.mode == 'run' else 1))
    if not os.environ.get('PERSONA_API_KEY'):
        os.environ['PERSONA_API_KEY'] = getpass.getpass('Temporary API key (not saved): ')
    (calibration if a.mode == 'calibration' else run)(a.out)


if __name__ == '__main__':
    main()
