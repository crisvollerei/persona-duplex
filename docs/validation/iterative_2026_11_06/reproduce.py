"""Prepare or rerun the archived comparisons in a new ignored workspace."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    destination=args.out.resolve();runtime=(ROOT/'data').resolve()
    if not destination.is_relative_to(runtime) or destination==runtime or destination.exists():
        raise SystemExit('Choose a new child directory under repository data/')
    study=destination/'docs/validation/iterative_2026_11_06';study.mkdir(parents=True)
    for name in ['versions','personas']:shutil.copytree(HERE/name,study/name)
    for name in ['protocol.json','execution_plan.json','scoring_amendment.json','behavior_codebook.md','behavior_rating_protocol.json',
                 'r3_plan.json','r3_protocol.json','r4_plan.json','r4_protocol.json']:
        if (HERE/name).exists():shutil.copy2(HERE/name,study/name)
    shutil.copy2(HERE/'run_study.py',study/'run_study.py')
    origin={'original_study':'docs/validation/iterative_2026_11_06','source':'archived versions and per-stage runners',
            'interpretation':'A new run of the archived comparisons; not a fresh adaptive code-development study',
            'automatic_content_ratings':False,'network':args.run}
    (destination/'reproduction_origin.json').write_text(json.dumps(origin,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.run:
        for stage in ['r0','r1','r2','b1','r3','r4']:
            phase=json.loads((HERE/'runs'/stage/'phase_protocol.json').read_text(encoding='utf-8'))
            shutil.copy2(HERE/'runs'/stage/'runner_snapshot.py',study/'run_study.py')
            subprocess.run([sys.executable,'-B','-X','utf8',str(study/'run_study.py'),'--stage',stage,
                            '--candidate',phase['candidate'],'--run'],cwd=destination,check=True)
    print(json.dumps({'prepared':str(study),'network':args.run},ensure_ascii=False))


if __name__=='__main__':main()
