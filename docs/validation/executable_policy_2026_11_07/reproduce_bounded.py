"""Reproduce archived bounded replay in a new isolated tree, with no network."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    target=args.out.resolve()
    if target.exists() or not target.is_relative_to((ROOT/'data').resolve()) or target==(ROOT/'data').resolve():
        raise SystemExit('Choose a new directory under data/')
    study=target/'docs/validation/executable_policy_2026_11_07';study.mkdir(parents=True)
    shutil.copytree(HERE/'bounded_replay/implementation_snapshot/persona',target/'persona')
    for name in ('results.json','raw_calls.json.gz','cases.json','persona_config.toml','review_codebook.md'):
        shutil.copy2(HERE/name,study/name)
    bounded=study/'bounded_replay';bounded.mkdir()
    shutil.copy2(HERE/'bounded_replay/runner_snapshot.py',bounded/'replay.py')
    subprocess.run([sys.executable,'-B','-X','utf8',str(bounded/'replay.py')],cwd=target,check=True)


if __name__=='__main__':main()
