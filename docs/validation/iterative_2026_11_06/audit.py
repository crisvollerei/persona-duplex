"""Reconcile archived source, assignments, raw streams and source quotes offline."""
import argparse
from collections import Counter,defaultdict
import gzip
import importlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_study as driver
import analyze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-report',action='store_true')
    parser.add_argument('--check-current',action='store_true');args=parser.parse_args()
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    protocol=driver.load(HERE/'protocol.json');plan=driver.load(HERE/'execution_plan.json')
    check('global_plan_fixed',driver.digest(plan)==protocol['plan_digest'])
    for stage in ['r3','r4']:
        extra=driver.load(HERE/(stage+'_plan.json'));p=driver.load(HERE/(stage+'_protocol.json'))
        check(stage+'_supplement_fixed',driver.digest(extra)==p['plan_digest']);plan[stage]=extra
    for name in ['baseline','revision_A','revision_B','revision_C','revision_D']:
        base=HERE/'versions'/name;manifest=driver.load(base/'manifest.json')
        for relative,expected in manifest['sources'].items():check(name+'/'+relative,driver.sha(base/relative)==expected)
    if args.check_current:
        for relative,expected in driver.load(HERE/'versions/revision_D/manifest.json')['sources'].items():
            check('current_matches_final/'+relative,driver.sha(driver.ROOT/relative)==expected)
    calls_count=outputs=framework_streams=quote_count=failures=0
    failed_memory=set()
    phase_counts={};prompt_matches=0
    for stage,jobs in plan.items():
        folder=HERE/'runs'/stage
        phase=driver.load(folder/'phase_protocol.json');results=driver.load(folder/'results.json')
        raw=json.loads(gzip.decompress((folder/'raw_calls.json.gz').read_bytes()))
        check(stage+'/raw_hash',driver.sha(folder/'raw_calls.json.gz')==results['raw_sha256'])
        check(stage+'/phase_hash',driver.sha(folder/'phase_protocol.json')==results['phase_protocol_sha256'])
        check(stage+'/plan_digest',driver.digest(jobs)==phase['jobs_digest'])
        check(stage+'/runner_hash',driver.sha(folder/'runner_snapshot.py')==phase['runner_sha256'])
        assigned={job['trial_id']:job for job in jobs};observed={r['trial_id']:r for r in results['records']}
        check(stage+'/all_assigned_recorded',set(assigned)==set(observed) and len(observed)==len(results['records']))
        call_map={c['trial_id']+'/'+c['call_id']:c for c in raw}
        check(stage+'/call_ids_unique',len(call_map)==len(raw)==results['calls_count'])
        calls_count+=len(raw);pairs=defaultdict(dict)
        scorer=analyze.frozen_scorer(folder)
        for ident,row in observed.items():
            modules=driver.version(row['version']);stream_module=importlib.import_module(modules['engine'].__package__+'.stream')
            calls=[call_map[cid] for cid in row['call_ids']]
            check(ident+'/call_membership',all(c['trial_id']==ident for c in calls))
            for call in calls:
                check(ident+'/'+call['call_id']+'/after_phase_fixed',call['started_at']>=phase['fixed_at'])
                if call['mode']=='stream' and call['status']=='ok':check(ident+'/'+call['call_id']+'/raw_chunks',''.join(call['chunks'])==call['output'])
            if row['experiment']=='memory':
                turns=[v['turn'] for v in row.get('training',[])]+[v['turn'] for v in row.get('probes',[])]
                user_by_id={v['turn']['turn_id']:v['turn']['user'] for v in row.get('training',[])}
                for item in row.get('stages',[]):
                    for judgment in item['judgments']:
                        for quote in judgment.get('evidence_quotes',[]):
                            check(ident+'/full_quote/'+quote['turn_id'],user_by_id.get(quote['turn_id'])==quote['quote']);quote_count+=1
                for probe in row.get('probes',[]):check(ident+'/memory_score/'+probe['condition'],scorer.memory_score(probe['turn']['say'],probe['target'])==probe['evaluation'])
            else:turns=[row['turn']] if row.get('turn') else []
            streams=[call for call in calls if call['mode']=='stream' and call['status']=='ok']
            check(ident+'/completed_stream_count',len(streams)==len(turns))
            for index,(turn,call) in enumerate(zip(turns,streams)):
                outputs+=1
                check(ident+f'/foreground_one/{index}',turn['calls_count']==1)
                check(ident+f'/delivered/{index}',''.join(turn['delivered_chunks'])==turn['say'])
                parser=stream_module.StreamParser();pieces=[]
                for chunk in call['chunks']:pieces.extend(parser.feed(chunk))
                parsed=parser.finish()
                check(ident+f'/frozen_parser/{index}',parsed.say==turn['say'] and pieces==turn['delivered_chunks']);framework_streams+=1
            if row['experiment']=='format':
                check(ident+'/frozen_score',scorer.score_format(row['turn']['say'],row['case'])==row['evaluation'])
                pairs[(row['role'],row['case']['id'])][row['arm']]=calls[0]['request']['messages']
            if row['status']!='ok':
                failures+=1
                if row['experiment']=='memory':failed_memory.add(ident)
        for key,pair in pairs.items():
            if 'baseline' in pair and 'revision_plain' in pair:
                check(stage+'/no_contract_prompt_equal/'+str(key),pair['baseline']==pair['revision_plain']);prompt_matches+=1
        phase_counts[stage]={'units':len(observed),'calls':len(raw)}
    check('within_hard_limit',calls_count<=800)
    check('all_framework_streams_replayed',framework_streams==outputs)
    failed_sidecar=json.loads(gzip.decompress((HERE/'failed_workflows.json.gz').read_bytes()))
    check('failed_workflow_sidecar_complete',len(failed_sidecar['items'])==6 and {row['trial_id'] for row in failed_sidecar['items']}==failed_memory)
    for row in failed_sidecar['items']:
        check(row['trial_id']+'/backlog_retained',row['queued_buffer_files']>0)
        for file in row['files']:
            check(row['trial_id']+'/'+file['path']+'/raw_bytes',driver.hashlib.sha256(file['text'].encode('utf-8')).hexdigest()==file['sha256'])
    key=driver.load(HERE/'behavior_review_key.json')
    inputs=driver.load(HERE/'behavior_review_inputs.json')
    for rater in [1,2]:
        ratings=driver.load(HERE/f'behavior_ratings_{rater}.json')
        check(f'ratings{rater}/coverage',len(ratings)==143 and {r['id'] for r in ratings}==set(key)=={r['id'] for r in inputs})
        for row in ratings:
            check(f'ratings{rater}/{row["id"]}/types',all(type(row[field]) is bool for field in ['action_advice','decision_pressure','clarification']) and (row['accepts_refusal'] is None or type(row['accepts_refusal']) is bool))
    for path in HERE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.png','.pyc') or path.name=='artifact_manifest.json':continue
        data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
        content=data.decode('utf-8')
        check('no_credentials/'+str(path.relative_to(HERE)),re.search(r'sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY',content) is None)
    report={'checks':len(checks),'failed':0,'actual_calls':calls_count,'observed_complete_outputs':outputs,
            'framework_streams_replayed':framework_streams,'whole_user_quotes_checked':quote_count,
            'no_contract_paired_prompts_identical':prompt_matches,'failed_units_retained':failures,
            'stages':phase_counts,'meaning':'Artifact agreement, not experimental success or full semantic correctness','details':checks}
    if args.write_report:driver.write(HERE/'audit_report.json',report)
    hashes=0
    if (HERE/'artifact_manifest.json').exists():
        for item in driver.load(HERE/'artifact_manifest.json')['files']:
            if driver.sha(HERE/item['path'])!=item['sha256']:raise AssertionError('Publication byte mismatch: '+item['path'])
            hashes+=1
    print(json.dumps({key:value for key,value in report.items() if key not in ('details','stages')},ensure_ascii=False))
    print('publication_hashes',hashes)


if __name__=='__main__':main()
