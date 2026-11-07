"""Prepare an isolated archived-source run; model requests require --run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    target = args.out.resolve()
    # The archived runner records the enclosing repository HEAD. Keeping runs
    # under ignored data also avoids mixing new calls with published evidence.
    runtime_root = (ROOT/'data').resolve()
    if not target.is_relative_to(runtime_root) or target == runtime_root:
        raise SystemExit('Choose a new child directory under repository data/')
    if target.exists(): raise SystemExit('Output already exists; use a new directory')
    shutil.copytree(HERE/'run/implementation_snapshot/persona', target/'persona')
    isolated = target/'docs/validation/generalization_2026_11_06'
    isolated.mkdir(parents=True)
    shutil.copy2(HERE/'run/runner_snapshot.py', isolated/'study.py')
    shutil.copytree(HERE/'personas', isolated/'personas')
    output = isolated/'run'
    command = [sys.executable,'-B','-X','utf8',str(isolated/'study.py'),'--out',str(output),'--prepare']
    if args.run: command.append('--run')
    subprocess.run(command,cwd=target,check=True)
    (target/'reproduction_origin.json').write_text(json.dumps(dict(
        original_protocol=str((HERE/'run/protocol.json').relative_to(ROOT)).replace('\\','/'),
        original_source_commit=json.loads((HERE/'run/implementation_snapshot/manifest.json').read_text(encoding='utf-8'))['git_head'],
        source='archived implementation and runner; new execution and service sampling'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(network=args.run, data_dir=str(output)),ensure_ascii=False))


if __name__ == '__main__': main()
