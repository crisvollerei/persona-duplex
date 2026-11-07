"""Read-only phase, provenance, compilation and publication-byte checks."""
from dataclasses import asdict
import argparse
import gzip
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import analyze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    summary,records,cases=analyze.collect();analyze.ratings(summary,records);analyze.extensions(summary,records)
    assert len(summary['ratings'])==2 and len(summary['bounded_replay']['ratings'])==2
    assert summary['state_validation']['state_matches']==4
    data=analyze.load(HERE/'bounded_replay/results.json')
    folder=HERE/'bounded_replay/implementation_snapshot/persona';alias='_policy_bounded_audit'
    spec=importlib.util.spec_from_file_location(alias,folder/'__init__.py',submodule_search_locations=[str(folder)])
    module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module)
    policy=importlib.import_module(alias+'.policy');session=importlib.import_module(alias+'.policy_session');config=importlib.import_module(alias+'.config')
    raw=json.loads(gzip.decompress((HERE/'raw_calls.json.gz').read_bytes()))
    verified=0;compiled=0
    for row in data['records']:
        original=records[row['original_id']];case=cases[original['case_id']]
        frame=policy.ContentFrame.from_json(json.dumps(original['content']))
        decision=policy.Decision(**row['decision'])
        text=session.bounded_reply(frame,decision,())
        if text is not None:
            assert text==row['say'] and row['replayed_expression_calls']==0;compiled+=1
        else:
            cfg=config.PersonaConfig.load(HERE/'persona_config.toml')
            cfg.role.name='行为表达实验';cfg.role.identity='';cfg.role.voice='自然、清楚的中文，克制地关心，不用固定口癖。'
            messages=session.render_messages(cfg,case['user'],tuple(policy.Source(**s) for s in case['sources']),frame,decision,(),None)
            call=next(c for c in raw if c['trial_id']==row['original_id'] and c['stage']=='render')
            assert messages==call['messages'] and row['say']==call['output'];verified+=1
    assert compiled==verified==16
    publication=0;manifest=HERE/'artifact_manifest.json'
    if manifest.exists():
        entries=analyze.load(manifest)['files']
        actual={p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.tmp') and p.name!='artifact_manifest.json'}
        assert actual=={e['path'] for e in entries} and len(entries)==len(actual)
        for entry in entries:assert analyze.sha(HERE/entry['path'])==entry['sha256']
        publication=len(entries)
    report=dict(material_checks=summary['audit']['checks'],failed=0,online_calls=summary['calls']+summary['state_validation']['calls'],
        main_complete=summary['complete'],state_outputs=summary['state_validation']['complete_outputs'],
        state_matches=4,compiled_recomputed=compiled,free_requests_byte_verified=verified,
        coding_sets=2,scope='Material and execution-path consistency; not proof of semantic correctness or personality quality')
    if args.write_report:analyze.save(HERE/'audit_report.json',report)
    print(json.dumps({**report,'publication_files_verified':publication},ensure_ascii=False))


if __name__=='__main__':main()
