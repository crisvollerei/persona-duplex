"""Run only the detached observers; replay consumes receipts, never human ballots."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import argparse
import getpass
import gzip
import hashlib
import itertools
import json
import os
import random
import sys

from observer import decode, messages, selfcheck

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from persona.llm import openai_compatible


def load(path):
    return json.loads(path.read_bytes())


def wire(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def run(mode, out):
    selfcheck()
    design = load(HERE / "design.json")
    cases = {c["id"]: c for c in load(HERE / "cases.json")}
    assert len(cases) == 16 and all(c["focus_turn"] == len(c["turns"]) for c in cases.values())
    rng = random.Random(design["seed"])
    ids = sorted(cases)
    rng.shuffle(ids)
    plan = []
    for ident in ids:
        cell = list(itertools.product([ident], design["methods"], design["replicates"]))
        rng.shuffle(cell)
        plan.extend(cell)
    order = [".".join(j) for j in plan]
    sources = [HERE / n for n in ("design.json", "cases.json", "study.py", "observer.py", "score.py")]
    sources.append(ROOT / "persona/llm.py")
    reg = dict(order=order, model=design["model"], hashes={
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    def save(name, obj):
        if mode == "dry":
            return
        dest = out / name
        if dest.exists() or mode == "replay":
            if load(dest) != obj:
                raise ValueError(f"{name}: frozen input or output differs")
        else:
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes((json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
            tmp.replace(dest)

    user_messages = {}
    for ident, method, repeat in plan:
        msgs = messages(cases[ident], design["location"], method)
        content = json.loads(msgs[1]["content"])
        assert set(content) == {"location", "target", "dialogue", "focus_turn"}
        assert all(set(t) == {"turn", "speaker", "text"} for t in content["dialogue"])
        assert user_messages.setdefault(ident, msgs[1]) == msgs[1]
    if mode == "live":
        out.mkdir(parents=True, exist_ok=True)
    save("registration.json", reg)
    lock, cache = Lock(), {}
    log = out / "calls.jsonl.gz"
    if mode != "dry" and log.exists():
        with gzip.open(log, "rb") as f:
            rr = [json.loads(line) for line in f]
        cache = {r["id"]: r for r in rr}
        assert len(cache) == len(rr) and set(cache) <= set(order)
    secret = None
    if mode == "live" and len(cache) < len(plan):
        secret = os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")

    def sample(job):
        ident, method, repeat = job
        key, c = ".".join(job), cases[ident]
        msgs = messages(c, design["location"], method)
        if key in cache:
            record = cache[key]
            assert record["messages"] == msgs and record["params"] == reg["model"]
            raw = record["raw"]
        elif mode == "dry":
            n = c["focus_turn"]
            raw = json.dumps(dict(act="unclear", act_quote=dict(turn=n, quote=c["turns"][n-1]["text"]),
                state="unknown", state_evidence=[], other_position="unlocated", position_evidence=[],
                arrival_pending="no_evidence", need_evidence=[]), ensure_ascii=False)
        else:
            if mode != "live" or not secret:
                raise ValueError("Missing receipt or credential")
            api = openai_compatible(**reg["model"], base_url="https://api.deepseek.com/v1", api_key=secret, timeout=90)
            try:
                raw = api(msgs)
                record = dict(ok=True, response=api.observations.response, raw=raw)
            except Exception as exc:
                raw = ""
                record = dict(ok=False, response=api.observations.response, raw=raw, error=type(exc).__name__)
            record.update(id=key, params=reg["model"], messages=msgs)
            with lock:
                with gzip.open(log, "ab") as fp:
                    fp.write(wire(record) + b"\n")
                cache[key] = record
                if len(cache) % 16 == 0:
                    print(len(cache), "receipts", flush=True)
        parsed = decode(raw, c, method)
        if mode == "dry":
            assert parsed["reason"] == "accepted"
        return key, dict(request=hashlib.sha256(wire(msgs)).hexdigest(), **parsed)

    rows = dict([sample(plan[0])])
    if mode == "live" and not cache[order[0]]["ok"]:
        raise RuntimeError("First provider call failed; stopped before dispatching other slots.")
    with ThreadPoolExecutor(max_workers=4) as workers:
        rows.update(workers.map(sample, plan[1:]))
    for ident in ids:
        for method in design["methods"]:
            assert rows[f"{ident}.{method}.A"]["request"] == rows[f"{ident}.{method}.A2"]["request"]
    if mode != "dry":
        assert set(cache) == set(rows)
    save("predictions.json", rows)
    print(mode, len(rows), "observer slots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry", "live", "replay"))
    parser.add_argument("--out", type=Path, default=HERE / "observed")
    args = parser.parse_args()
    run(args.mode, args.out)
