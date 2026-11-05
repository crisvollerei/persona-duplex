"""Focused online verification of the revised framework; no embedded credentials."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / "docs/validation/followup_2026_11_05"
sys.path.insert(0, str(STUDY))
from run_experiments import Recorder, PARAMS, FACTS, configs, build_engine, turn, run_job, write_json
from analyze import corrected_answer
from supplement_memory import TRAINING, PROBE, score
from persona.consolidate import Consolidator
from persona.llm import text_prompt
from persona.store import now_iso, rebuild_views


def memory_trial(rep, run_dir):
    calls = []
    record = {"trial_id": f"memory_r{rep}", "experiment": "memory", "replicate": rep,
              "status": "ok", "training": [], "probes": [], "calls": calls}
    try:
        cfg = configs()["base"]
        engine = build_engine(copy.deepcopy(cfg), run_dir / record["trial_id"] / "training", "structured", calls)
        before = engine.store.traits()
        for user in TRAINING:
            record["training"].append(turn(engine, user, calls))
        record["consolidation"] = Consolidator(engine.store, llm=text_prompt(Recorder("text", calls))).run_once()
        judgments = engine.store.user_judgments(limit=None)
        record["judgments"] = judgments
        record["traits_unchanged"] = before == engine.store.traits()
        order = ["learned", "empty"]
        random.Random(20261105 + rep).shuffle(order)
        for condition in order:
            probe = build_engine(copy.deepcopy(cfg), run_dir / record["trial_id"] / condition, "structured", calls)
            if condition == "learned":
                for j in judgments:
                    probe.store.add_judgment(j["judgment"], j["confidence"], j.get("evidence_ids"),
                        source=j.get("source", "llm"), evidence_scope=j.get("evidence_scope", "unknown"),
                        evidence_quotes=j.get("evidence_quotes"))
            probe.memory_snapshot = probe.store.user_judgments(limit=None)
            observed = turn(probe, PROBE, calls)
            record["probes"].append({"condition": condition, "memory_injected": probe.memory_snapshot,
                                      "turn": observed, "evaluation": score(observed["say"])})
        before_rebuild = engine.store.user_judgments(limit=None)
        engine.store.save_view("judgments", {"items": []})
        rebuild_views(engine.store)
        record["quotes_survive_rebuild"] = before_rebuild == engine.store.user_judgments(limit=None)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc).replace(os.environ.get("PERSONA_API_KEY", "not-a-key"), "[REDACTED]")}
    return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    jobs = []
    for rep in range(2):
        for role in ("base", "blue"):
            for case in FACTS:
                jobs.append({"experiment": "facts", "condition": "structured_" + role,
                    "case_id": case["id"], "replicate": rep, "mode": "structured", "role": role,
                    "case": case, "trial_id": f"format_{role}_{case['id']}_r{rep}"})
    random.Random(20261105).shuffle(jobs)
    if not args.run:
        print(json.dumps({"format_trials": len(jobs), "memory_replicates": 4, "network": False})); return
    if not os.environ.get("PERSONA_API_KEY"): raise SystemExit("Set PERSONA_API_KEY in process environment")
    if (HERE / "results.json").exists(): raise SystemExit("Results already exist; preserve the previous run")
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "persona").glob("*.py"))}
    data = {"protocol": protocol, "manifest": {"started_at": now_iso(), "settings": PARAMS,
        "sources": source_hashes, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "records": []}
    run_dir = ROOT / "data" / ("framework_revision_" + uuid4().hex)
    raw = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs = [pool.submit(run_job, j, run_dir) for j in jobs] + [pool.submit(memory_trial, rep, run_dir) for rep in range(4)]
        for f in as_completed(fs):
            r = f.result()
            if r["experiment"] == "facts" and r.get("turns"):
                r["corrected_strict_match"] = corrected_answer(r["turns"][-1]["say"], r["gold"]) is True
            calls = r.pop("calls")
            r["call_ids"] = [r["trial_id"] + "/" + c["call_id"] for c in calls]
            raw.extend({**c, "trial_id": r["trial_id"]} for c in calls)
            data["records"].append(r)
            write_json(HERE / "results.json", data)
            print("VERIFICATION", len(data["records"]), "/36", r["status"], flush=True)
    data["manifest"].update(completed_at=now_iso(), calls_count=len(raw))
    encoded = json.dumps(raw, ensure_ascii=False)
    if os.environ["PERSONA_API_KEY"] in encoded: raise RuntimeError("Credential found in output")
    with gzip.open(HERE / "raw_calls.json.gz", "wt", encoding="utf-8") as out: out.write(encoded)
    data["manifest"]["raw_calls_sha256"] = hashlib.sha256((HERE / "raw_calls.json.gz").read_bytes()).hexdigest()
    write_json(HERE / "results.json", data)


if __name__ == "__main__": main()
