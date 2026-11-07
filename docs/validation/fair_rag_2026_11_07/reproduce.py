"""Prepare an isolated same-evidence rerun; network requires --run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args();target=args.out.resolve()
    if target.exists() or not target.is_relative_to((ROOT/'data').resolve()) or target==(ROOT/'data').resolve():
        raise SystemExit('Choose a new child directory under repository data/')
    study=target/'docs/validation/fair_rag_2026_11_07';study.mkdir(parents=True)
    for name in ['implementation_snapshot','personas']:shutil.copytree(HERE/name,study/name)
    fixed=['protocol.json','cases.json','corpus.json','retrieval.json','execution_plan.json','traditional_prompts.json','configurations.json',
           'review_codebook.md','review_protocol.json','pair_assignment.json']
    for name in fixed:shutil.copy2(HERE/name,study/name)
    shutil.copy2(HERE/'runner_snapshot.py',study/'benchmark.py')
    (target/'reproduction_origin.json').write_text(json.dumps({'source':'archived implementation and runner','new_outputs_require_new_coding':True,'network':args.run},indent=2)+'\n',encoding='utf-8')
    if args.run:subprocess.run([sys.executable,'-B','-X','utf8',str(study/'benchmark.py'),'--run'],cwd=target,check=True)
    print(json.dumps({'prepared':str(study),'network':args.run},ensure_ascii=False))


if __name__=='__main__':main()
