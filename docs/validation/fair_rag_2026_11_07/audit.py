"""Audit shared evidence, method balance, streams and blinded coding offline."""
import argparse
from collections import Counter,defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import benchmark as helper


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    protocol=helper.load(HERE/'protocol.json');results=helper.load(HERE/'results.json')
    plan=helper.load(HERE/'execution_plan.json');cases={r['id']:r for r in helper.load(HERE/'cases.json')}
    retrieval=helper.load(HERE/'retrieval.json');corpus=helper.load(HERE/'corpus.json')
    check('protocol_bytes',helper.sha(HERE/'protocol.json')==results['protocol_sha256'])
    check('runner_bytes',helper.sha(HERE/'runner_snapshot.py')==protocol['runner_sha256'])
    for name,expected in protocol['hashes'].items():check('fixed/'+name,helper.sha(HERE/name)==expected)
    for name,expected in protocol['profile_hashes'].items():check('fixed_profile/'+name,helper.sha(HERE/'personas'/name)==expected)
    for relative,expected in helper.load(HERE/'implementation_snapshot/manifest.json')['sources'].items():
        check('source/'+relative,helper.sha(HERE/'implementation_snapshot'/relative)==expected)
    spec=importlib.util.spec_from_file_location('_fair_rag_frozen_benchmark',HERE/'runner_snapshot.py')
    frozen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen;spec.loader.exec_module(frozen)
    modules=frozen.implementation()
    stream_parser=__import__(modules['engine'].__package__+'.stream',fromlist=['StreamParser']).StreamParser
    for ident,case in cases.items():
        computed=frozen.retrieve(case,corpus)
        check('retrieval/'+ident,computed==retrieval[ident])
        check('evidence_complete/'+ident,computed['required_sources_present'])
        check('context_budget/'+ident,computed['character_count']<=2600)
        check('opaque_visible_id/'+ident,case['family'] not in case['user'] and case['family'] not in computed['context'])
        check('no_diagnostic_types/'+ident,'"type":' not in computed['context'] and '"family":' not in computed['context'])
    assigned={row['trial_id']:row for row in plan};observed={row['trial_id']:row for row in results['records']}
    check('all_assignments',len(assigned)==len(plan)==224 and set(assigned)==set(observed) and len(observed)==len(results['records']))
    factorial=Counter((row['method'],row['profile']) for row in plan if row['kind']!='open')
    check('full_factorial',len(factorial)==6 and set(factorial.values())=={32})
    raw=json.loads(gzip.decompress((HERE/'raw_calls.json.gz').read_bytes()))
    check('raw_hash',helper.sha(HERE/'raw_calls.json.gz')==results['raw_sha256'])
    check('request_budget',len(raw)==results['calls_count']==224 and len(raw)<=protocol['hard_limit'])
    call_map={call['trial_id']+'/'+call['call_id']:call for call in raw}
    check('unique_calls',len(call_map)==len(raw))
    config=helper.load(HERE/'configurations.json');context_groups=defaultdict(set);framework_streams=0
    for ident,row in observed.items():
        check(ident+'/assigned_metadata',all(row[key]==assigned[ident][key] for key in assigned[ident]))
        check(ident+'/completed',row['status']=='ok' and row['calls_count']==1 and len(row['call_ids'])==1)
        call=call_map[row['call_ids'][0]];messages=call['request']['messages'];context=retrieval[row['case_id']]['context']
        check(ident+'/same_system_evidence',messages[0]['role']=='system' and messages[0]['content'].endswith(frozen.COMMON+'\n\n'+context))
        check(ident+'/raw_context_hash',call['context_sha256']==row['context_sha256']==retrieval[row['case_id']]['context_sha256'])
        check(ident+'/original_user',messages[-1]['content']==cases[row['case_id']]['user'])
        check(ident+'/same_config',row['config_hash']==frozen.digest(config[row['profile']]))
        settings=call['request']['settings']
        check(ident+'/sampling',all(settings[key]==protocol['parameters'][key] for key in ['model','temperature','max_tokens','timeout']))
        check(ident+'/after_fixed',call['started_at']>=protocol['fixed_at'])
        check(ident+'/raw_chunks',''.join(call['chunks'])==call['output'])
        check(ident+'/delivery',''.join(row['delivered_chunks'])==row['say'] and row['delivery_match'])
        if row['method']=='framework':
            parser=stream_parser();chunks=[]
            for piece in call['chunks']:chunks.extend(parser.feed(piece))
            parsed=parser.finish()
            check(ident+'/frozen_stream_replay',parsed.say==row['say'] and chunks==row['delivered_chunks']);framework_streams+=1
            check(ident+'/reparsed_protocol',parsed.validity()['valid']==row['protocol_valid'])
        else:
            check(ident+'/direct_reply',row['say']==call['output'])
            cfg=frozen.config(modules,row['profile'])
            expected=frozen.traditional_prompt(cfg,modules,cases[row['case_id']]['user'])+'\n\n'+frozen.COMMON+'\n\n'+context
            check(ident+'/full_traditional_prompt',messages[0]['content']==expected)
        check(ident+'/fixed_score',row['evaluation']==frozen.score(row['say'],cases[row['case_id']]))
        context_groups[row['case_id']].add(call['context_sha256'])
    check('shared_evidence_all_methods',len(context_groups)==20 and all(len(hashes)==1 for hashes in context_groups.values()))
    check('framework_replay_coverage',framework_streams==112)
    inputs=helper.load(HERE/'pair_inputs.json');key=helper.load(HERE/'pair_key.json');assignment=helper.load(HERE/'pair_assignment.json')
    check('all_pairs',len(inputs)==len(key)==len(assignment)==112 and {r['id'] for r in inputs}==set(key)=={r['id'] for r in assignment})
    check('side_balance',Counter(row['a_method'] for row in assignment)=={'traditional':56,'framework':56})
    contexts=helper.load(HERE/'review_contexts.json')
    for row in inputs:
        check(row['id']+'/no_method_tag',set(row)=={'id','context_id','profile','a','b'})
        check(row['id']+'/opaque_review_context',re.fullmatch(r'Q\d+',row['context_id']) is not None)
    for name,expected in helper.load(HERE/'review_inputs_manifest.json')['sha256'].items():check('fixed_review/'+name,helper.sha(HERE/name)==expected)
    ratings_present=[]
    for rater in [1,2]:
        path=HERE/f'pair_ratings_{rater}.json'
        if not path.exists():continue
        ratings=helper.load(path);ratings_present.append(rater)
        check(f'rater{rater}/coverage',len(ratings)==112 and {r['id'] for r in ratings}==set(key))
        for row in ratings:
            profile=next(p['profile'] for p in inputs if p['id']==row['id'])
            check(f'rater{rater}/{row["id"]}/neutral_null',profile!='neutral' or all(row[side]['role_fidelity'] is None for side in ['a','b']))
            check(f'rater{rater}/{row["id"]}/verdicts',all(row[side]['target_verdict'] in {'correct','incorrect','no_answer','ambiguous','not_applicable'} for side in ['a','b']))
    for path in HERE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.png','.pyc') or path.name=='artifact_manifest.json':continue
        content=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
        check('credential_scan/'+str(path.relative_to(HERE)),re.search(r'sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY',content.decode('utf-8')) is None)
    report=dict(checks=len(checks),failed=0,calls=len(raw),shared_contexts=len(context_groups),framework_streams_replayed=framework_streams,
        ratings_present=ratings_present,meaning='Material consistency, not correctness or advantage proof',details=checks)
    if args.write_report:helper.write(HERE/'audit_report.json',report)
    hashes=0
    if (HERE/'artifact_manifest.json').exists():
        entries=helper.load(HERE/'artifact_manifest.json')['files']
        actual={path.relative_to(HERE).as_posix() for path in HERE.rglob('*')
                if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.tmp') and path.name!='artifact_manifest.json'}
        if len(entries)!=len({item['path'] for item in entries}) or actual!={item['path'] for item in entries}:
            raise AssertionError('Publication manifest coverage changed')
        for item in entries:
            if helper.sha(HERE/item['path'])!=item['sha256']:raise AssertionError('Publication bytes changed: '+item['path'])
            hashes+=1
    print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False));print('publication_hashes',hashes)


if __name__=='__main__':main()
