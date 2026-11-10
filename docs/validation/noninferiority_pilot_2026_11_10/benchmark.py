"""Frozen, held-out default-text versus optimized direct-prompt pilot.

Preparation, review export and verification are offline. Online stages require
--run and a process-only PERSONA_API_KEY. No retries or post-hoc sample filling.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = 'deepseek-chat'
ENDPOINT = 'https://api.deepseek.com/v1/chat/completions'
PARAMS = dict(model=MODEL, temperature=.7, max_tokens=1600, stream=False)
MAX_CALLS = 132
SEED = 2026111073


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    secret = os.environ.get('PERSONA_API_KEY')
    if secret and secret in text:
        raise ValueError('credential in research output')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def modules():
    alias = '_noninferiority_frozen'
    directory = HERE / 'implementation_snapshot'
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, directory/'persona/__init__.py',
            submodule_search_locations=[str(directory/'persona')])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    return {name: importlib.import_module(alias+'.'+name)
            for name in ('config','session','store','prompt','llm')}


def make_config(m, profile):
    cfg = m['config'].PersonaConfig.load(HERE/'implementation_snapshot/personas'/profile['config'])
    cfg.role.name = profile['name']
    cfg.role.identity = profile['spec']
    cfg.role.voice = profile['voice']
    cfg.role.phrases = {}
    cfg.role.special = []
    cfg.role.attention = profile['attention']
    cfg.role.bias = profile['bias']
    for name, value in profile['traits'].items():
        trait = cfg.role.traits[name]
        trait.anchor = trait.value = value
        trait.lo = 0.; trait.hi = 1.; trait.drift = 1.
    cfg.validate()
    return cfg


def user_text(case):
    return ('[应用提供的当前资料；合成情境]\n'+json.dumps(case['facts'], ensure_ascii=False)
            +'\n[本轮用户原文]\n'+case['query'])


def seed_store(m, case, directory):
    store = m['store'].Store(directory)
    ids = {}
    for index, row in enumerate(case['memories']):
        tid = row.get('source_turn_id') or ('source-'+str(index))
        quote = row.get('source_quote')
        evidence = [tid] if quote else []
        if quote:
            store.append('turn', {'turn_id': tid, 'user': quote})
        out = store.add_judgment(row['text'], row.get('confidence', .8), evidence,
            source=row.get('source','explicit'),
            evidence_scope='specified' if quote else 'unsupported' if row.get('source')=='llm' else 'unknown',
            evidence_quotes=[dict(turn_id=tid, quote=quote, evidence_scope='specified')] if quote else [],
            supersedes=ids.get(row.get('supersedes')),
            memory_scope=row.get('scope'), valid_from=row.get('valid_from'), expires_at=row.get('expires_at'))
        ids[row['id']] = out['id']
    return store


def personality_reference(m, cfg):
    return json.dumps(dict(name=cfg.role.name, spec=cfg.role.identity, voice=cfg.role.voice,
        attention=cfg.role.attention, attention_hint=m['prompt']._ATTENTION_HINT[cfg.role.attention],
        bias=cfg.role.bias, legibility_max=cfg.role.legibility_max,
        trait_guidance=m['prompt']._trait_block(cfg).replace('baseline/SAY','实际回复').replace('direct','基础判断'),
        stance_guidance=m['prompt'].stance_policy(cfg).replace('baseline/SAY','实际回复').replace('direct','基础判断')),
        ensure_ascii=False)


def shared_context(m, cfg, store, case):
    matches = store.recall_judgments(user_text(case), scope=case['memory_scope'], budget_chars=6000)
    context = m['prompt'].build_context_block(cfg, relation=store.relation(), pendings=[], judgments=matches)
    return matches, context


def mechanical(case, text):
    """Literal requirements only, never a factual/personality oracle."""
    truth = case['ground_truth']
    issues = []
    for fragment in truth.get('must_include', []):
        if fragment not in text: issues.append('missing_literal:'+fragment)
    for fragment in truth.get('must_not_include', []):
        if fragment in text: issues.append('forbidden_literal:'+fragment)
    if truth.get('json_exact') is not None:
        try:
            obj = json.loads(text)
            if obj != truth['json_exact']: issues.append('json_value_mismatch')
        except ValueError:
            issues.append('not_exact_json')
    if truth.get('final_line') is not None and (not text.splitlines() or text.splitlines()[-1] != truth['final_line']):
        issues.append('final_line_mismatch')
    return dict(status='passed' if not issues else 'failed', reasons=issues,
                scope='literal_checks_not_semantic_truth')


class Client:
    def __init__(self, folder, limit):
        self.folder = Path(folder); self.limit = limit
        self.count = 0; self.errors = 0; self.lock = threading.Lock()

    def call(self, id, messages):
        with self.lock:
            if self.count >= self.limit or self.errors >= 3:
                raise RuntimeError('budget_or_error_stop')
            self.count += 1
        row = dict(id=id, status='started', started_at=now(), endpoint=ENDPOINT,
                   request={**PARAMS, 'messages': messages}, timeout_s=90)
        path = self.folder/'calls'/(id+'.json')
        save(path, row)
        start = time.monotonic()
        try:
            request = urllib.request.Request(ENDPOINT,
                data=json.dumps(row['request'],ensure_ascii=False).encode(),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['PERSONA_API_KEY']},
                method='POST')
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode('utf-8'))
            row['response'] = payload
            choice = payload['choices'][0]
            content = choice['message'].get('content')
            if choice.get('finish_reason') != 'stop' or not isinstance(content,str):
                raise ValueError('incomplete_generation')
            row.update(status='ok',output=content)
        except Exception as exc:
            with self.lock: self.errors += 1
            row.update(status='error',error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                row['http_status'] = exc.code
                exc.close()
        finally:
            row.update(finished_at=now(),elapsed_s=time.monotonic()-start)
            save(path,row)
        return row


def one(m, client, case, profile, method, candidate):
    id = case['id']+'-'+profile['id']+'-'+(candidate['id'] if candidate else 'framework')
    record = dict(id=id,case_id=case['id'],profile_id=profile['id'],method=method,
                  candidate_id=candidate['id'] if candidate else None,call_id=id,
                  status='error',say='',raw='',validity=None,elapsed_s=None)
    with tempfile.TemporaryDirectory(prefix='persona-ni-') as directory:
        cfg = make_config(m,profile); store=seed_store(m,case,directory)
        matches, context = shared_context(m,cfg,store,case)
        history = deepcopy(case['history'])
        record.update(selected_memory_ids=[j['id'] for j in matches],memory_context=context,
                      evidence_sha256=digest(dict(facts=case['facts'],history=history,context=context)))
        start = time.monotonic()
        try:
            if method == 'framework':
                holder = []
                def llm(messages):
                    out=client.call(id,messages);holder.append(out)
                    if out['status']!='ok': raise RuntimeError('generation_failed')
                    return out['output']
                llm.metadata = dict(kind='openai_compatible',model=MODEL,temperature=PARAMS['temperature'],
                    max_tokens=PARAMS['max_tokens'],endpoint=ENDPOINT)
                options=m['session'].SessionConfig(memory_mode='frozen',condition='noninferiority-pilot',
                    memory_scope=case['memory_scope'])
                with m['session'].Session(cfg,store,llm,options=options,protocol='text') as session:
                    session.engine.history=history
                    session.engine._turn_seq=len(history)//2
                    session.engine._topic=[h['content'] for h in history if h['role']=='user']
                    if history:
                        session.engine._last_user=next((h['content'] for h in reversed(history) if h['role']=='user'),'')
                        session.engine._last_say=next((h['content'] for h in reversed(history) if h['role']=='assistant'),'')
                    result=session.say(user_text(case))
                    record.update(say=str(result.say),validity=result.validity)
                if len(holder)!=1: raise ValueError('foreground_call_count')
                record['raw']=holder[0]['output']
                recalls=[r for r in store.read_ledger() if r['kind']=='memory_recall']
                if recalls[-1]['judgment_ids']!=record['selected_memory_ids']:
                    raise ValueError('evidence_mismatch')
                record['framework_ledger']=store.read_ledger()
            else:
                system=candidate['system']+'\n\n[共享人格规格]\n'+personality_reference(m,cfg)
                messages=[dict(role='system',content=system),dict(role='system',content=context),
                          *history,dict(role='user',content=user_text(case))]
                out=client.call(id,messages)
                if out['status']!='ok': raise RuntimeError('generation_failed')
                record.update(raw=out['output'],say=out['output'].strip())
            record['status']='ok' if record['say'] else 'empty_reply'
        except Exception as exc:
            record['error_type']=type(exc).__name__
        record['elapsed_s']=time.monotonic()-start
        record['mechanical']=mechanical(case,record['say'])
    return record


def verify():
    protocol=load(HERE/'protocol.json')
    for path,expected in protocol['hashes'].items():
        if sha(HERE/path)!=expected: raise ValueError('frozen source changed: '+path)
    return protocol


def freeze():
    if (HERE/'protocol.json').exists(): raise ValueError('already frozen')
    snapshot=HERE/'implementation_snapshot'
    for path in sorted((ROOT/'persona').glob('*.py'))+sorted((ROOT/'personas').glob('*.toml')):
        destination=snapshot/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(path.read_bytes())
    names=['benchmark.py','test_benchmark.py','protocol.md','method_review.md','rubric.md','calibration.json',
           'materials.json','materials_notes.md','prompts.json','profiles.json','analyze.py','generate_figures.py']
    paths=[HERE/name for name in names]+[p for p in snapshot.rglob('*') if p.is_file()]
    for path in paths:
        if not path.is_file(): raise ValueError('missing frozen input '+str(path))
    save(HERE/'protocol.json',dict(version=1,frozen_at=now(),git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),hashes={p.relative_to(HERE).as_posix():sha(p) for p in paths},
        max_endpoint_calls=MAX_CALLS,development_calls=36,main_calls=96,sampling=PARAMS,
        server_seed=None,workers=3,retries=0,timeout_s=90,first_token_latency_measured=False,
        evaluation_units='24 authored independent situations, 2 profiles; not representative random sampling'))
    items=load(HERE/'calibration.json')['items']
    public=[];key={}
    for row in items:
        key[row['id']]=dict(expected=row['expected'],rationale=row.get('rationale',''))
        public.append({k:v for k,v in row.items() if k not in ('expected','rationale')})
    save(HERE/'reviews/calibration_inputs.json',public)
    save(HERE/'reviews/calibration_key.json',key)


def ratings(stage,coder):
    rows=load(HERE/f'reviews/{stage}_ratings_{coder}.json')['ratings']
    expected={r['id'] for r in load(HERE/f'reviews/{stage}_inputs.json')}
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=expected:
        raise ValueError('incomplete/duplicate ratings')
    for r in rows:
        if any(r[k] is not None and (type(r[k]) is not int or not 1<=r[k]<=5) for k in ('persona_fit','context_fit')):
            raise ValueError('invalid score')
        for k,values in dict(facts=['supported','conflict','unknown'],
            boundary=['respected','violated','not_applicable','unclear'],
            fulfillment=['fulfilled','failed','not_applicable','unclear']).items():
            if r[k] not in values: raise ValueError('invalid label '+k)
        if not isinstance(r.get('reason'),str) or not r['reason'].strip(): raise ValueError('missing evidence reason')
    return {r['id']:r for r in rows}


def calibrate():
    verify();key=load(HERE/'reviews/calibration_key.json')
    assessments={}
    for coder in (1,2):
        rows=ratings('calibration',coder);correct=0;total=0;category_correct=0;category_total=0
        for id,row in rows.items():
            for field,expected in key[id]['expected'].items():
                total+=1
                ok=row[field] in expected if isinstance(expected,list) else row[field]==expected
                correct+=ok
                if field in ('facts','boundary','fulfillment'):
                    category_total+=1;category_correct+=ok
        sentinels={'CAL03':{'facts':'conflict'},'CAL05':{'facts':'conflict'},
                   'CAL06':{'facts':'conflict'},'CAL07':{'facts':'unknown'},
                   'CAL04':{'boundary':'violated','fulfillment':'fulfilled'}}
        hard=all(rows[id][field]==expected for id,fields in sentinels.items() for field,expected in fields.items())
        assessments[str(coder)]=dict(correct=correct,total=total,category_correct=category_correct,
            category_total=category_total,sentinels_passed=hard,
            passed=bool(hard and correct/total>=.9 and category_correct/category_total>=.9))
    save(HERE/'calibration_gate.json',dict(assessments=assessments,
        passed=all(a['passed'] for a in assessments.values()),protocol_sha256=sha(HERE/'protocol.json')))


def run(stage):
    verify()
    if not load(HERE/'calibration_gate.json')['passed']: raise ValueError('calibration did not pass')
    folder=HERE/('development' if stage=='development' else 'run')
    if folder.exists(): raise ValueError('output exists; no retry or overwrite')
    if not os.environ.get('PERSONA_API_KEY'): raise ValueError('process-only PERSONA_API_KEY required')
    m=modules();profiles=load(HERE/'profiles.json');materials=load(HERE/'materials.json')
    candidates=load(HERE/'prompts.json')['candidates']
    if stage=='main':
        selection=load(HERE/'selection.json')
        if selection['protocol_sha256']!=sha(HERE/'protocol.json'):raise ValueError('selection not bound')
        candidates=[next(c for c in candidates if c['id']==selection['selected_id'])]
    jobs=[]
    for case in materials['development' if stage=='development' else 'test']:
        for profile in profiles:
            for candidate in candidates: jobs.append((case,profile,'strong_prompt',candidate))
            if stage=='main':jobs.append((case,profile,'framework',None))
    random.Random(SEED+(stage=='main')).shuffle(jobs)
    client=Client(folder,len(jobs));records=[]
    save(folder/'manifest.json',dict(started_at=now(),stage=stage,protocol_sha256=sha(HERE/'protocol.json'),
        planned_calls=len(jobs),job_order=[(c['id'],p['id'],kind,x['id'] if x else None) for c,p,kind,x in jobs],
        selection_sha256=sha(HERE/'selection.json') if stage=='main' else None))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(one,m,client,*job) for job in jobs]
        for future in as_completed(futures):
            row=future.result();records.append(row);save(folder/'records'/(row['id']+'.json'),row)
            print(stage,len(records),'/',len(jobs),row['id'],row['status'],flush=True)
    records.sort(key=lambda r:r['id'])
    save(folder/'results.json',dict(stage=stage,protocol_sha256=sha(HERE/'protocol.json'),records=records))
    calls=[load(p) for p in sorted((folder/'calls').glob('*.json'))]
    (folder/'raw_calls.json.gz').write_bytes(gzip.compress(json.dumps(calls,ensure_ascii=False).encode(),mtime=0))
    export_reviews(stage)


def export_reviews(stage):
    folder=HERE/('development' if stage=='development' else 'run')
    records=load(folder/'results.json')['records']
    cases={c['id']:c for c in load(HERE/'materials.json')['development' if stage=='development' else 'test']}
    profiles={p['id']:p for p in load(HERE/'profiles.json')}
    random.Random(SEED+90+(stage=='main')).shuffle(records)
    items=[];key={}
    for i,r in enumerate(records,1):
        id=('D' if stage=='development' else 'M')+f'{i:03}'
        c=cases[r['case_id']];p=profiles[r['profile_id']]
        items.append(dict(id=id,profile={k:p[k] for k in ('name','spec','voice')},facts=c['facts'],
            history=c['history'],user=c['query'],reply=r['say'],memory_context=r['memory_context'],requirements=c['requirements']))
        key[id]=r['id']
    save(HERE/f'reviews/{stage}_inputs.json',items);save(HERE/f'reviews/{stage}_key.json',key)


def select():
    verify()
    if (HERE/'selection.json').exists():raise ValueError('selection already frozen')
    rows=load(HERE/'development/results.json')['records']
    key=load(HERE/'reviews/development_key.json');reverse={v:k for k,v in key.items()}
    reviews=[ratings('development',i) for i in (1,2)]
    scores=[]
    for candidate in load(HERE/'prompts.json')['candidates']:
        subset=[r for r in rows if r['candidate_id']==candidate['id']]
        values=[review[reverse[r['id']]] for review in reviews for r in subset]
        bad=sum(any(v[k]==label for k,label in [('facts','conflict'),('boundary','violated'),('fulfillment','failed')]) for v in values)
        failed=sum(r['status']!='ok' for r in subset)
        persona=sum(v['persona_fit'] if v['persona_fit'] is not None else 1 for v in values)/len(values)
        context=sum(v['context_fit'] if v['context_fit'] is not None else 1 for v in values)/len(values)
        scores.append(dict(id=candidate['id'],failed_generations=failed,error_ratings=bad,
            persona_mean=persona,context_mean=context,selection_key=[failed,bad,-min(persona,context),-(persona+context)/2,candidate['id']]))
    chosen=min(scores,key=lambda s:s['selection_key'])['id']
    save(HERE/'selection.json',dict(selected_id=chosen,selected_at=now(),scores=scores,
        protocol_sha256=sha(HERE/'protocol.json'),development_results_sha256=sha(HERE/'development/results.json'),
        review_sha256={str(i):sha(HERE/f'reviews/development_ratings_{i}.json') for i in (1,2)}))
    print('Selected baseline:',chosen)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','calibrate','development','select','main'])
    p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.stage in ('development','main'):
        if not a.run:raise SystemExit('online execution requires explicit --run')
        run(a.stage)
    else:{'freeze':freeze,'calibrate':calibrate,'select':select}[a.stage]()


if __name__=='__main__':main()
