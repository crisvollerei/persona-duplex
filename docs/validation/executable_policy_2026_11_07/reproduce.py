"""Prepare an isolated archived online phase; requests require --run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['main','state'],default='main');p.add_argument('--run',action='store_true');a=p.parse_args()
    target=a.out.resolve()
    if target.exists() or not target.is_relative_to((ROOT/'data').resolve()) or target==(ROOT/'data').resolve():
        raise SystemExit('Choose a new child directory under data/')
    source=HERE if a.phase=='main' else HERE/'state_validation'
    study=target/'docs/validation/executable_policy_2026_11_07'
    if a.phase=='state':study=study/'state_validation'
    study.mkdir(parents=True)
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    for name in ['protocol.json',*protocol['hashes']]:shutil.copy2(source/name,study/name)
    shutil.copytree(source/'implementation_snapshot',study/'implementation_snapshot')
    entry='benchmark.py' if a.phase=='main' else 'run.py'
    archived=source/'runner_snapshot.py' if a.phase=='main' else source/'run.py'
    shutil.copy2(archived,study/entry)
    if a.phase=='main':
        for name in ['profiles.json','automatic_cases.json','execution_plan.json']:
            if not (study/name).exists():shutil.copy2(source/name,study/name)
    if a.run:subprocess.run([sys.executable,'-B','-X','utf8',str(study/entry),'--run'],cwd=target,check=True)
    print(json.dumps({'prepared':str(study),'phase':a.phase,'network':a.run},ensure_ascii=False))


if __name__=='__main__':main()
