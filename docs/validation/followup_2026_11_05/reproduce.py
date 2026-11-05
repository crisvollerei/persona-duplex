"""Re-run study conditions with archived or current code in an isolated tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def prepare_output(target, implementation):
    source = HERE / "implementation_snapshot/persona" if implementation == "study" else ROOT / "persona"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))}
    if implementation == "study":
        expected = json.loads((HERE / "implementation_snapshot/manifest.json").read_text(encoding="utf-8"))
        expected = {row["path"].split("/")[-1]: row["sha256"] for row in expected["files"] if row["copied"]}
        if hashes != expected:
            raise ValueError("Archived study implementation does not match its manifest")
    target.mkdir(parents=True)
    shutil.copytree(source, target / "persona", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    study_dir = target / HERE.relative_to(ROOT)
    study_dir.mkdir(parents=True)
    shutil.copytree(HERE / "personas", study_dir / "personas")
    for filename in ("protocol.json", "supplement_protocol.json", "run_experiments.py", "supplement_memory.py"):
        shutil.copyfile(HERE / filename, study_dir / filename)
    (target / "reproduction.json").write_text(json.dumps({
        "implementation": implementation, "source_sha256": hashes,
        "study_directory": study_dir.relative_to(target).as_posix(),
        "scope": "New execution; archived published evidence is unchanged",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return study_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory under this repository's data/")
    parser.add_argument("--implementation", choices=("study", "current"), default="study",
                        help="Use the archived study code (default) or copy the current framework")
    parser.add_argument("--run", action="store_true", help="Execute real API requests after preparation")
    args = parser.parse_args()
    target = (ROOT / args.out).resolve()
    if not target.is_relative_to((ROOT / "data").resolve()) or target == (ROOT / "data").resolve():
        parser.error("--out must be a new child directory of repository data/")
    if target.exists():
        parser.error("--out already exists; choose a new directory to preserve previous results")
    if args.run and not os.environ.get("PERSONA_API_KEY"):
        parser.error("Set PERSONA_API_KEY in the process environment before --run")
    study_dir = prepare_output(target, args.implementation)
    if not args.run:
        print("Prepared:", target)
        print("Study directory:", study_dir)
        print("No API requests were made. Choose a new --out with --run to execute.")
        return
    for filename, workers in (("run_experiments.py", "8"), ("supplement_memory.py", "4")):
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(study_dir / filename),
                        "--run", "--workers", workers], cwd=target, check=True)
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "analyze.py"),
                    "--input", str(study_dir / "experiment_results.json"), "--out", str(study_dir),
                    "--supplement", str(study_dir / "supplement_results.json"),
                    "--annotations", str(study_dir / "annotation_summary.json")], cwd=ROOT, check=True)
    print("Reproduction artifacts:", study_dir)


if __name__ == "__main__":
    main()
