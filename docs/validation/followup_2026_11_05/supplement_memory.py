"""Append-only end-to-end memory supplement; credentials stay in environment."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from run_experiments import (BASE_URL, HERE, PARAMS, ROOT, SEED, Recorder,
                             build_engine, configs, digest, turn, write_json)
from persona.consolidate import Consolidator
from persona.llm import text_prompt
from persona.store import Store, now_iso, rebuild_views

REPEATS = 4
TRAINING = [
    "我试过几种学习节奏，现在明确采用17分钟专注、3分钟休息。这是我自己的固定偏好，今天也这样安排，请记住。",
    "今天又用了17分钟专注、3分钟休息的节奏，适合我。我要继续沿用这个个人习惯。",
    "再确认一次，我明确选定的日常节奏仍是17分钟专注、3分钟休息。以后我问起时，请按我说过的习惯回答。",
]
PROBE = ("我之前明确告诉过你的个人专注时长和休息时长分别是多少？"
         "请只按你能看到的明确记忆回答，没有依据就说未知。"
         "末行写结论：专注<分钟数>分钟，休息<分钟数>分钟；若没有依据，末行写结论：未知。")


def score(text):
    matches = re.findall(r"结论\s*[:：]\s*([^\n]+)", text)
    final = re.sub(r"[\s*`，,。.!！；;]", "", matches[-1]) if matches else None
    return {"conclusion": final, "recall_both": final == "专注17分钟休息3分钟",
            "unknown": final == "未知", "format_present": final is not None}


def buffer_snapshot(store):
    files = sorted(store.buffer_dir.glob("*.json"))
    processing = sorted((store.buffer_dir / "processing").glob("*.json"))
    def contents(paths):
        return [{"filename": p.name, "payload": json.loads(p.read_text(encoding="utf-8"))} for p in paths]
    return {"queued": contents(files), "processing": contents(processing)}


def import_judgments(store, judgments):
    imported = []
    for item in judgments:
        imported.append(store.add_judgment(
            item["judgment"], item["confidence"], list(item.get("evidence_ids", [])),
            affects=copy.deepcopy(item.get("affects", {})), source=item.get("source", "llm"),
            batch_id=item.get("batch_id", ""), evidence_scope=item.get("evidence_scope", "batch")))
    return imported


def replay_check(store):
    before = {"user_judgments": store.user_judgments(limit=None), "traits": store.traits()}
    store.save_view("judgments", {"items": []})
    store.save_view("traits", {"traits": {}})
    rebuilt = rebuild_views(store)
    after = {"user_judgments": store.user_judgments(limit=None), "traits": store.traits()}
    views = {"judgments": store.load_view("judgments"), "traits": store.load_view("traits")}
    return {"before": before, "after": after, "rebuilt_summary": rebuilt,
            "user_judgments_equal": before["user_judgments"] == after["user_judgments"],
            "traits_equal": before["traits"] == after["traits"], "regenerated_views": views,
            "judgment_view_matches": views["judgments"] == {"items": before["user_judgments"]},
            "trait_view_matches": views["traits"] == {"traits": before["traits"]}}


def run_replicate(rep, run_dir):
    calls = []
    record = {"replicate": rep, "trial_id": f"memory_e2e_r{rep}", "status": "ok",
              "started_at": now_iso(), "training": [], "probes": [], "calls": calls}
    def tag_calls(start, stage):
        for call in calls[start:]: call["stage"] = stage
    try:
        cfg = copy.deepcopy(configs()["base"])
        engine = build_engine(cfg, run_dir / record["trial_id"] / "training", "structured", calls)
        store = engine.store
        record.update(config=asdict(cfg), config_hash=digest(asdict(cfg)), run_manifest=engine.run,
                      traits_before=store.traits(), memory_before=store.user_judgments(limit=None))
        for idx, user in enumerate(TRAINING):
            start = len(calls)
            training_turn = turn(engine, user, calls)
            ledger_turn = [r for r in store.read_ledger() if r.get("kind") == "turn"][-1]
            training_turn["turn_id"] = ledger_turn["turn_id"]
            record["training"].append(training_turn)
            tag_calls(start, f"training_{idx+1}")
        record["evidence_turn_ids"] = [t["turn_id"] for t in record["training"]]
        record["buffer_before_consolidation"] = buffer_snapshot(store)
        record["history_before_new_sessions"] = copy.deepcopy(engine.history)
        start = len(calls)
        try:
            record["consolidation"] = Consolidator(
                store, llm=text_prompt(Recorder("text", calls)), heuristic_fallback=False,
                audit_context={"supplement": record["trial_id"]}).run_once()
        except Exception as exc:
            record["consolidation_error"] = {"type": type(exc).__name__, "message": str(exc)}
            record["status"] = "consolidation_error"
        tag_calls(start, "consolidation")
        judgments = store.user_judgments(limit=None)
        record.update(consolidated_judgments=judgments, traits_after=store.traits(),
                      unconsumed_buffer=buffer_snapshot(store),
                      self_story=store.self_story_path.read_text(encoding="utf-8") if store.self_story_path.exists() else "")
        record["traits_unchanged"] = record["traits_before"] == record["traits_after"]
        evidence = set(record["evidence_turn_ids"])
        record["evidence_ids_valid"] = all(set(j.get("evidence_ids", [])) <= evidence for j in judgments)
        record["replay"] = replay_check(store)
        order = ["learned", "empty"]
        random.Random(SEED + rep).shuffle(order)
        record["probe_order"] = order
        for condition in order:
            probe_engine = build_engine(copy.deepcopy(cfg), run_dir / record["trial_id"] / condition,
                                        "structured", calls)
            imported = import_judgments(probe_engine.store, judgments) if condition == "learned" else []
            probe_engine.memory_snapshot = copy.deepcopy(imported)
            initial_history = copy.deepcopy(probe_engine.history)
            assert not initial_history
            start = len(calls)
            observed = turn(probe_engine, PROBE, calls)
            tag_calls(start, "probe_" + condition)
            record["probes"].append({"condition": condition, "history_before": initial_history,
                "memory_injected": imported, "turn": observed, "evaluation": score(observed["say"]),
                "run_manifest": probe_engine.run})
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["completed_at"] = now_iso()
    key = os.environ.get("PERSONA_API_KEY")
    if key:
        def redact(value):
            if isinstance(value, str): return value.replace(key, "[REDACTED]")
            if isinstance(value, list): return [redact(v) for v in value]
            if isinstance(value, dict): return {k:redact(v) for k,v in value.items()}
            return value
        record = redact(record)
    return record


def save_raw(path, raw):
    encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    key = os.environ.get("PERSONA_API_KEY")
    if key and key in encoded: raise RuntimeError("Credential found in raw publication payload")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as target: target.write(encoded)
    tmp.replace(path)


def self_check():
    assert "17" not in PROBE and not re.search(r"\d", PROBE)
    assert all("17分钟专注" in u and "3分钟休息" in u for u in TRAINING)
    assert score("结论：专注17分钟，休息3分钟。 ")["recall_both"]
    assert not score("结论：专注25分钟，休息5分钟")["recall_both"]
    assert score("结论：未知")["unknown"]
    assert not score("我不知道")["format_present"]
    with tempfile.TemporaryDirectory(prefix="persona-memory-supplement-") as temp:
        store = Store(Path(temp) / "training")
        store.init_traits(configs()["base"].role.traits_state(), persona="offline_check")
        for idx, user in enumerate(TRAINING):
            store.push_buffer({"turn_id": f"t_{idx+1:04d}", "user": user, "say": "收到。", "need": "认同"})
        # The first memory stage requests JSON; the second requests story text.
        responses = iter([json.dumps([{"judgment":"用户偏好17分钟专注、3分钟休息。", "confidence":.8,
                                     "evidence_ids":["t_0001","t_0002","t_0003"]}],ensure_ascii=False),
                          "用户反复说明了自己选择的专注节奏。"])
        out = Consolidator(store, llm=lambda prompt: next(responses)).run_once()
        assert out["judgments"] == 1
        replay = replay_check(store)
        assert all(replay[k] for k in ["user_judgments_equal","traits_equal","judgment_view_matches","trait_view_matches"])
        destination = Store(Path(temp) / "learned")
        copied = import_judgments(destination, store.user_judgments(limit=None))
        assert copied[0]["evidence_ids"] == ["t_0001","t_0002","t_0003"]
        assert not buffer_snapshot(store)["queued"]
    print("SUPPLEMENT_SELF_CHECK_PASS (no network)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.self_check: self_check(); return
    if not args.run:
        print(json.dumps({"replicates": REPEATS,"training_turns":12,"probe_turns":8,
                          "model_settings":PARAMS,"network_calls":False},ensure_ascii=False)); return
    if not os.environ.get("PERSONA_API_KEY"): raise SystemExit("Set PERSONA_API_KEY in the process environment")
    output, raw_path = HERE / "supplement_results.json", HERE / "supplement_raw_calls.json.gz"
    if output.exists() or raw_path.exists(): raise SystemExit("Supplement evidence exists; refusing overwrite")
    protocol = json.loads((HERE / "supplement_protocol.json").read_text(encoding="utf-8"))
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [*sorted((ROOT / "persona").glob("*.py")), *sorted((HERE / "personas").glob("*.toml")),
                         HERE / "run_experiments.py", Path(__file__)]}
    manifest = {"started_at": now_iso(), "settings":PARAMS,"endpoint":BASE_URL,"replicates":REPEATS,
                "workers":args.workers,"local_randomization_seed":SEED,"protocol_hash":digest(protocol),
                "source_hash":digest(sources),"sources":sources,
                "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    data = {"protocol":protocol,"manifest":manifest,"training_inputs":TRAINING,"probe":PROBE,"records":[]}
    run_dir = ROOT / "data" / "followup_2026_11_05_memory_supplement"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(output, data)
    raw = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_replicate, rep, run_dir) for rep in range(REPEATS)]
        for future in as_completed(futures):
            record = future.result()
            calls = record.pop("calls")
            record["call_ids"] = [record["trial_id"] + "/" + c["call_id"] for c in calls]
            raw.extend({**c,"trial_id":record["trial_id"],"global_call_id":record["trial_id"]+"/"+c["call_id"]} for c in calls)
            data["records"].append(record)
            save_raw(raw_path, raw)
            write_json(output, data)
            print(json.dumps({"supplement_completed":len(data["records"]),"planned":REPEATS,
                              "status":record["status"]},ensure_ascii=False),flush=True)
    data["records"].sort(key=lambda r:r["replicate"])
    manifest.update(completed_at=now_iso(),calls_count=len(raw),raw_calls_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
    write_json(output, data)
    print("SUPPLEMENT_COMPLETE",flush=True)


if __name__ == "__main__": main()
