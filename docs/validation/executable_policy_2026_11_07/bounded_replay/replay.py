"""Offline bounded realization with byte-checked reuse of original free replies.

No network requests. Changed non-advisory replies are compiled by current code;
unchanged free actions reuse archived response increments only after exact
request-message comparison. This is not another online replication.
"""
from dataclasses import asdict
import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;ROOT=HERE.parents[3]


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    if (HERE/'results.json').exists():raise SystemExit('Replay exists; preserve publication evidence')
    sources={}
    for source in sorted((ROOT/'persona').glob('*.py')):
        rel=source.relative_to(ROOT);target=HERE/'implementation_snapshot'/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);sources[rel.as_posix()]=sha(source)
    save(HERE/'implementation_snapshot/manifest.json',dict(sources=sources))
    alias='_bounded_policy_replay'
    folder=HERE/'implementation_snapshot/persona'
    spec=importlib.util.spec_from_file_location(alias,folder/'__init__.py',submodule_search_locations=[str(folder)])
    module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module)
    m={k:importlib.import_module(alias+'.'+k) for k in ('policy','policy_session','config','store')}
    original=load(STUDY/'results.json');raw=json.loads(gzip.decompress((STUDY/'raw_calls.json.gz').read_bytes()))
    cases={c['id']:c for c in load(STUDY/'cases.json')}
    records=[];inputs=[];mapping={};verified=0
    for index,row in enumerate(sorted((r for r in original['records'] if r['kind']=='controlled' and r['method']=='policy'),key=lambda r:r['id']),1):
        case=cases[row['case_id']];calls=[c for c in raw if c['trial_id']==row['id'] and c['stage']=='render'];assert len(calls)==1
        call=calls[0];used=[]
        def replay(messages):
            nonlocal verified
            if messages!=call['messages']:raise AssertionError('Changed free request: '+row['id'])
            verified+=1;used.append(1)
            yield from call['chunks']
        replay.metadata={'kind':'archived_response_replay'};replay.observations=SimpleNamespace(response=call['response'])
        cfg=m['config'].PersonaConfig.load(STUDY/'persona_config.toml')
        cfg.role.name='行为表达实验';cfg.role.identity='';cfg.role.voice='自然、清楚的中文，克制地关心，不用固定口癖。'
        policy=m['policy'].PolicyConfig(**row['policy_config'])
        store=m['store'].Store(ROOT/'data'/'policy-bounded-replay'/row['id'])
        with m['policy_session'].PolicySession(cfg,store,lambda messages:(_ for _ in ()).throw(AssertionError('Unexpected request')),
            stream_llm=replay,policy=policy,clock=lambda:'2026-11-07T21:00:00+08:00',realization='bounded') as session:
            if case['seed']=='deny':session.boundary('deny',user='以后不要给建议',quote='以后不要给建议',topic=case['topic'])
            frame=m['policy'].ContentFrame.from_json(json.dumps(case['frame']))
            delivered=[]
            result=session.say(case['user'],topic=case['topic'],frame=frame,sources=tuple(m['policy'].Source(**s) for s in case['sources']),on_delta=delivered.append)
            assert ''.join(delivered)==result.say
            item=dict(original_id=row['id'],case_id=row['case_id'],profile=row['profile'],replicate=row['replicate'],
                say=result.say,decision=asdict(result.decision),realization=result.realization,replayed_expression_calls=len(used),
                network_calls=0,unchanged_reply=result.say==row['say'],original_reply_sha256=hashlib.sha256(row['say'].encode()).hexdigest())
            records.append(item)
            if result.realization=='bounded_atoms':
                id=f'B{index:03d}';mapping[id]=row['id']
                inputs.append(dict(id=id,user=case['user'],sources=case['sources'],confirmed_boundary='当前work主题已有长期不主动建议边界' if case['seed']=='deny' else '无已确认建议边界',open_commitments=[],say=result.say))
    save(HERE/'results.json',dict(source_results_sha256=sha(STUDY/'results.json'),runner_sha256=sha(__file__),
        network_calls=0,records=records,byte_checked_reused_requests=verified,
        scope='Offline constrained compilation plus exact-request reuse of old free replies; not an independent online replication or new model usage.'))
    shutil.copy2(STUDY/'review_codebook.md',HERE/'review_codebook.md')
    save(HERE/'review_inputs.json',inputs);save(HERE/'review_key.json',mapping)
    print(json.dumps({'outputs':len(records),'compiled_changed':len(inputs),'byte_checked_reused_requests':verified,'network_calls':0}))


if __name__=='__main__':main()
