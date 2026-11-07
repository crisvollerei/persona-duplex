"""Fixed four-workflow supplement: extract control, restart, realize follow-up."""
import argparse
from dataclasses import asdict
import getpass
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    text=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    key=os.environ.get('PERSONA_API_KEY')
    if key and key in text:raise ValueError('Credential in material')
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(text,encoding='utf-8',newline='\n')
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def modules():
    folder=HERE/'implementation_snapshot'
    for relative,expected in load(folder/'manifest.json')['sources'].items():
        if sha(folder/relative)!=expected:raise ValueError('Archived source changed')
    alias='_policy_state_supplement'
    if alias not in sys.modules:
        spec=importlib.util.spec_from_file_location(alias,folder/'persona/__init__.py',submodule_search_locations=[str(folder/'persona')])
        m=importlib.util.module_from_spec(spec);sys.modules[alias]=m;spec.loader.exec_module(m)
    return {k:importlib.import_module(alias+'.'+k) for k in ('policy','policy_session','config','llm','store')}


def prepare():
    if (HERE/'protocol.json').exists():raise SystemExit('Protocol exists')
    sources={}
    for source in sorted((ROOT/'persona').glob('*.py')):
        relative=source.relative_to(ROOT);target=HERE/'implementation_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);sources[relative.as_posix()]=sha(source)
    save(HERE/'implementation_snapshot/manifest.json',dict(sources=sources))
    shutil.copy2(ROOT/'personas/default.toml',HERE/'persona_config.toml')
    save(HERE/'cases.json',[
        dict(id='tonight',user='今晚不要给建议',topic='work',probe_topic='work',expected_permission='deny',expected_revision=1),
        dict(id='today_variant',user='今天不用给我建议',topic='work',probe_topic='work',expected_permission='deny',expected_revision=1),
        dict(id='global',user='以后所有话题都不要给建议',topic='work',probe_topic='social',expected_permission='deny',expected_revision=1),
        dict(id='turn',user='这轮不要给建议',topic='work',probe_topic='work',expected_permission='unspecified',expected_revision=0),
    ])
    save(HERE/'protocol.json',dict(fixed_at=modules()['store'].now_iso(),planned_workflows=4,planned_outputs=8,planned_calls=12,hard_limit=12,
        design='Four independently initialized natural-text controls (up to2calls) followed by restart and one supplied-content realization. Retain control rejection and all failures. No retry, checker or outcome-dependent replacement.',
        model='deepseek-chat',temperature=.7,max_tokens=900,timeout_s=90,clock='2026-11-07T21:00:00+08:00',
        boundary_policy='Actual complete command determines scope and duration; model scope is an advisory label.',
        runner_sha256=sha(__file__),hashes={n:sha(HERE/n) for n in ('cases.json','persona_config.toml')}))
    print('Prepared four workflows, maximum12calls, no network')


def run():
    protocol=load(HERE/'protocol.json')
    if sha(__file__)!=protocol['runner_sha256']:raise SystemExit('Runner changed')
    for n,h in protocol['hashes'].items():
        if sha(HERE/n)!=h:raise SystemExit('Fixed input changed')
    if (HERE/'results.json').exists():raise SystemExit('Results exist')
    if not os.environ.get('PERSONA_API_KEY'):os.environ['PERSONA_API_KEY']=getpass.getpass('Temporary API key: ')
    m=modules();raw=[];results=[];calls=0;errors=0
    def recorder(stage):
        fn=m['llm'].openai_compatible('deepseek-chat',base_url='https://api.deepseek.com/v1',temperature=0 if stage=='content' else .7,max_tokens=1300 if stage=='content' else 900,timeout=90)
        def call(messages):
            nonlocal calls,errors
            if calls>=12 or errors>=3:raise RuntimeError('Fixed budget/error stop')
            calls+=1;record=dict(stage=stage,messages=messages,settings=fn.metadata,started_at=m['store'].now_iso(),status='running');raw.append(record);start=time.monotonic()
            try:
                output=fn(messages);record.update(status='ok',output=output);return output
            except Exception as exc:errors+=1;record.update(status='error',error_type=type(exc).__name__);raise
            finally:record.update(elapsed_s=time.monotonic()-start,response=fn.observations.response)
        call.metadata=fn.metadata;call.observations=fn.observations
        return call
    content=recorder('content');render=recorder('render')
    cfg=m['config'].PersonaConfig.load(HERE/'persona_config.toml')
    policy=m['policy'].PolicyConfig(intervention=1,practical=0,feeling=1)
    runtime=ROOT/'data'/('policy-state-supplement-'+uuid4().hex)
    for case in load(HERE/'cases.json'):
        store=m['store'].Store(runtime/case['id'])
        def session():return m['policy_session'].PolicySession(cfg,store,render,analyze_llm=content,policy=policy,clock=lambda:protocol['clock'])
        first=session();row=dict(id=case['id'],stages=[])
        for phase in ('control','probe'):
            active=first if phase=='control' else session();start_calls=calls;start=time.monotonic()
            try:
                if phase=='control':result=active.say(case['user'],topic=case['topic'])
                else:
                    user='报告没写完，今天好累。'
                    frame=m['policy'].ContentFrame(facts=(m['policy'].Quote('user','报告没写完'),),focuses=(m['policy'].Focus('feeling','feeling',m['policy'].Quote('user','好累')),))
                    result=active.say(user,topic=case['probe_topic'],frame=frame)
                stage=dict(phase=phase,status='ok',say=result.say,decision=asdict(result.decision))
            except Exception as exc:stage=dict(phase=phase,status='error',error_type=type(exc).__name__)
            stage.update(calls_count=calls-start_calls,elapsed_s=time.monotonic()-start,state=asdict(active.state),
                         permission=active.state.advice_permission(case['probe_topic'],at=protocol['clock']))
            row['stages'].append(stage);active.close()
        results.append(row);print('COMPLETE',len(results),'/4',case['id'],'calls=',calls,flush=True)
    encoded=json.dumps(raw,ensure_ascii=False).encode()
    if os.environ['PERSONA_API_KEY'].encode() in encoded:raise ValueError('Credential in raw calls')
    (HERE/'raw_calls.json.gz').write_bytes(gzip.compress(encoded,mtime=0))
    save(HERE/'results.json',dict(protocol_sha256=sha(HERE/'protocol.json'),raw_sha256=sha(HERE/'raw_calls.json.gz'),calls_count=calls,request_errors=errors,workflows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    if a.run:run()
