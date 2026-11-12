"""Generate fixed paired conversations and reproduce the blind packet. Human ballots are never inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from threading import Lock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona.config import PersonaConfig
from persona.llm import openai_compatible
from persona.session import Session, SessionConfig
from persona.store import Store
from baseline import make_messages

OUT = HERE / "observed"
METHODS = ("framework", "examples")
PRIOR = HERE.parent / "noninferiority_pilot_2026_11_10"


def load(p):
    return json.loads(p.read_bytes())


def dump(p, obj, verify=False):
    if verify or p.exists():
        if load(p) != obj:
            raise ValueError(f"Archive mismatch: {p.name}")
        return
    temp = p.with_suffix(".tmp")
    temp.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2)+"\n").encode())
    temp.replace(p)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key():
    return os.getenv("PERSONA_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("Provider credential: ")


def get_api(secret):
    return openai_compatible(**load(HERE / "protocol.json")["provider"], base_url="https://api.deepseek.com/v1",
                             api_key=secret, timeout=90)


def options():
    return SessionConfig(memory_mode="frozen", consolidate_on_exit=False, generation_path="full")


def sources():
    paths = list((ROOT / "persona").glob("*.py"))
    paths += [ROOT / "personas/shorekeeper.toml", *(HERE / n for n in
              ("protocol.json", "materials.json", "baseline.py", "study.py"))]
    from baseline import SOURCE_FILES
    paths.extend(SOURCE_FILES)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}


def make_registration(scenes):
    rng = random.Random(366132)
    ids = [s["id"] for s in scenes]
    rng.shuffle(ids)
    orientations = [dict(A=a, B=next(m for m in METHODS if m != a)) for a in
                    (METHODS[0], METHODS[1], rng.choice(METHODS))]
    order = [(i, m) for i in ids for m in METHODS]
    rng.shuffle(order)
    return dict(hashes=sources(), order=order, codebook=dict(zip(ids, orientations)),
                provider=load(HERE / "protocol.json")["provider"])


def main(mode):
    mat = load(HERE / "materials.json"); scenes = mat["scenes"]
    assert len(scenes) == 3 and len({s["id"] for s in scenes}) == 3
    assert all(len(s["turns"]) == 4 and all(isinstance(t, str) and t.strip() for t in s["turns"]) for s in scenes)
    reg = make_registration(scenes)
    # Lists, not tuples, round-trip through JSON.
    reg = json.loads(json.dumps(reg))
    if mode != "dry":
        if mode == "live":
            OUT.mkdir(exist_ok=True)
        dump(OUT / "registration.json", reg, mode == "replay")
    records = {}
    log = OUT / "calls.jsonl.gz"
    if mode != "dry" and log.is_file():
        with gzip.open(log, "rb") as f:
            for line in f:
                row = json.loads(line)
                assert row["id"] not in records
                records[row["id"]] = row
    stored_ids = set(records)
    secret = key() if mode == "live" and not (OUT / "trajectories.json").is_file() else None
    lock, visited = Lock(), set()
    def trajectory(job):
        sid, method = job
        turns = next(s["turns"] for s in scenes if s["id"] == sid)
        cfg = PersonaConfig.load(ROOT / "personas/shorekeeper.toml")
        ctx = {}; history = []; trajectory = []
        def invoke(msgs):
            uid = f"{sid}.{method}.{ctx['t']}"
            if ctx.get("called"): raise AssertionError("extra foreground call")
            ctx["called"] = True
            with lock: visited.add(uid)
            if mode == "dry":
                if method == "examples": return "好。"
                return "【HEAD】\nneed: 只是想说说话\nconfidence: 0.5\ndirect: 注意。\nbaseline: 好。\ndisplace: none\n【SAY】\n好。\n【TAIL】\nsettle: -\ndelta: -\npending: -"
            if uid in records:
                record = records[uid]
                if record["messages"] != msgs or record["params"] != reg["provider"]:
                    raise ValueError("changed replay request: " + uid)
            else:
                if not secret or mode != "live": raise RuntimeError("Missing receipt: " + uid)
                api = get_api(secret)
                record = dict(id=uid, params=reg["provider"], messages=deepcopy(msgs))
                try:
                    text = api(msgs)
                    record.update(ok=True, raw=text, response=api.observations.response)
                except Exception as exc:
                    record.update(ok=False, raw="", response=api.observations.response, error=type(exc).__name__)
                with lock:
                    with gzip.open(log, "ab") as f:
                        f.write(json.dumps(record, ensure_ascii=False).encode()+b"\n")
                    records[uid] = record
                    if len(records) % 6 == 0: print(len(records), "receipts", flush=True)
            if not record["ok"]: raise ConnectionError("archived provider failure")
            return record["raw"]
        with tempfile.TemporaryDirectory(prefix="persona-character-") as work:
            guard = Session(cfg, Store(work), invoke, options=options()) if method == "framework" else nullcontext()
            with guard as session:
                interrupted = False
                for n, user in enumerate(turns, 1):
                    ctx.update(t=n, called=False)
                    if interrupted:
                        trajectory.append(dict(user=user, say="", status="upstream_missing", validity=None))
                        continue
                    try:
                        if session is not None:
                            reply = session.say(user)
                            say, validity = reply.say, reply.validity
                        else:
                            say = invoke(make_messages(cfg, history, user)).strip(); validity = None
                        if not ctx["called"]: raise AssertionError("no foreground call")
                        history.extend([dict(role="user", content=user), dict(role="assistant", content=say)])
                        trajectory.append(dict(user=user, say=say, status="delivered" if say else "empty", validity=validity))
                    except ConnectionError:
                        trajectory.append(dict(user=user, say="", status="call_failed", validity=None))
                        interrupted = True
        return f"{sid}.{method}", trajectory

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = dict(pool.map(trajectory, reg["order"]))
    assert len(rows) == 6 and all(len(t) == 4 for t in rows.values())
    blind = dict(scenes=[dict(id=s["id"], lines={side:[r["say"] for r in rows[s["id"]+"."+m]]
                        for side,m in reg["codebook"][s["id"]].items()}) for s in scenes])
    if mode != "dry":
        assert set(records) == visited and stored_ids <= visited
        receipts = [r["response"]["id"] for r in records.values() if r["ok"]]
        assert all(receipts) and len(receipts) == len(set(receipts))
        dump(OUT / "trajectories.json",rows,mode=="replay")
        dump(OUT / "blind.json",blind,mode=="replay")
    print(mode, len(rows), "trajectories;", len(visited), "requests")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("dry","live","replay"))
    main(p.parse_args().mode)
