"""Verify archived observations and seal a completed, reproducible delivery."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import benchmark as b
import analyze


def audit():
    checks=[]
    def check(value,name):
        if not value:raise AssertionError(name)
        checks.append(name)
    p=b.verify()
    check(p['max_endpoint_calls']==132,'fixed endpoint budget')
    check(b.load(b.HERE/'calibration_gate.json')['passed'],'calibration gate')
    profiles=b.load(b.HERE/'profiles.json');cases=b.load(b.HERE/'materials.json')
    check(len(cases['test'])==24 and len(profiles)==2,'planned cases and profiles')
    check(all(v==3 for v in Counter(c['family'] for c in cases['test']).values()),'three cases per family')
    check(sum(c['local_gain_group'] for c in cases['test'])==9,'prespecified memory subgroup')
    count=0
    for stage,folder,n in [('development','development',36),('main','run',96)]:
        records=b.load(b.HERE/folder/'results.json')['records']
        raw=[b.load(f) for f in sorted((b.HERE/folder/'calls').glob('*.json'))]
        check(raw==json.loads(gzip.decompress((b.HERE/folder/'raw_calls.json.gz').read_bytes())),'exact compressed calls '+stage)
        check(len(records)==n,'all planned records '+stage)
        check(len({r['id'] for r in records})==n,'unique records '+stage)
        by_id={r['id']:r for r in raw};count+=len(raw)
        for row in raw:
            check('headers' not in row and 'Authorization' not in row,'no authorization in call '+row['id'])
            check(row['request']['temperature']==.7 and row['request']['max_tokens']==1600,'shared resource ceiling '+row['id'])
            check(row['request']['model']=='deepseek-chat','requested model '+row['id'])
            check(row['started_at']>=p['frozen_at'],'request after freeze '+row['id'])
        for r in records:
            if r['raw']:
                check(r['call_id'] in by_id and r['raw']==by_id[r['call_id']]['output'],'raw association '+r['id'])
            if r['method']=='framework':
                matching=[x for x in r.get('framework_ledger',[]) if x['kind']=='memory_recall']
                check(len(matching)==1 and matching[0]['judgment_ids']==r['selected_memory_ids'],'actual framework recall '+r['id'])
        key=b.load(b.HERE/f'reviews/{stage}_key.json');items=b.load(b.HERE/f'reviews/{stage}_inputs.json')
        check(len(key)==len(items)==n and set(key.values())=={r['id'] for r in records},'complete anonymous mapping '+stage)
        indexed={r['id']:r for r in records}
        for item in items:
            check(item['reply']==indexed[key[item['id']]]['say'],'actual text reviewed '+item['id'])
            check(not {'method','candidate_id','case_id','expected','raw','validity'}&set(item),'metadata hidden '+item['id'])
        for coder in (1,2):
            check(len(b.ratings(stage,coder))==n,'complete ratings '+stage+str(coder))
    check(count==132,'all actual endpoint calls')
    selection=b.load(b.HERE/'selection.json')
    check(selection['development_results_sha256']==b.sha(b.HERE/'development/results.json'),'selection dev binding')
    for coder in (1,2):
        check(selection['review_sha256'][str(coder)]==b.sha(b.HERE/f'reviews/development_ratings_{coder}.json'),'selection review binding')
    main=b.load(b.HERE/'run/results.json')['records']
    for case in cases['test']:
        for profile in profiles:
            pair=[r for r in main if r['case_id']==case['id'] and r['profile_id']==profile['id']]
            check(len(pair)==2 and {r['method'] for r in pair}=={'framework','strong_prompt'},'planned pair '+case['id']+profile['id'])
            check(pair[0]['evidence_sha256']==pair[1]['evidence_sha256'],'identical shared evidence '+case['id']+profile['id'])
            strong=next(r for r in pair if r['method']=='strong_prompt')
            check(strong['candidate_id']==selection['selected_id'],'frozen chosen baseline '+case['id']+profile['id'])
    check(analyze.summarize()==b.load(b.HERE/'summary.json'),'exact analysis replay')
    secret=re.compile(rb'(?i)(?:sk-[a-z0-9_-]{24,}|Bearer\s+[a-z0-9._-]{20,})')
    for path in b.HERE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:continue
        content=path.read_bytes()
        if path.suffix=='.gz':content=gzip.decompress(content)
        check(not secret.search(content),'credential scan '+str(path.relative_to(b.HERE)))
        if path.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
                target=target.split('#',1)[0]
                if not target or re.match(r'^[a-z]+:',target):continue
                # Manifest files are created only when sealing this completed delivery.
                if target in ('artifact_manifest.json','audit_report.json') and path.parent==b.HERE:continue
                check((path.parent/target).exists(),'relative link '+target)
    return dict(status='passed',checks=len(checks),endpoint_calls=count,
        main_outputs=96,paired_situations=24,protocol_sha256=b.sha(b.HERE/'protocol.json'),
        scope='Data association, frozen inputs and bytes; not independent semantic truth or user-experience proof.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    manifest=b.HERE/'artifact_manifest.json'
    if args.seal and manifest.exists():raise ValueError('already sealed')
    report=audit()
    if args.seal:
        b.save(b.HERE/'audit_report.json',report)
        files={p.relative_to(b.HERE).as_posix():dict(bytes=p.stat().st_size,sha256=b.sha(p))
            for p in sorted(b.HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=manifest}
        b.save(manifest,dict(algorithm='SHA-256',files=files))
    if manifest.exists():
        entries=b.load(manifest)['files']
        actual={p.relative_to(b.HERE).as_posix() for p in b.HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=manifest}
        if actual!=set(entries):raise AssertionError('manifest exact coverage')
        for name,row in entries.items():
            path=b.HERE/name
            if path.stat().st_size!=row['bytes'] or b.sha(path)!=row['sha256']:raise AssertionError('changed artifact '+name)
        report['verified_artifacts']=len(entries)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
