"""Offline registration, real default Session delivery, and offline replay. Standard library only."""
import argparse
import getpass
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona import prompt
from persona.config import PersonaConfig
from persona.llm import openai_compatible
from persona.protocol import visible_dialogue
from persona.session import Session, SessionConfig
from persona.store import Store

OLD = HERE.parent / "noninferiority_pilot_2026_11_10"
PARAMS = dict(model="deepseek-chat", temperature=0.7, max_tokens=900)
ENDPOINT = "https://api.deepseek.com/v1"


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def config():
    p = next(x for x in load(OLD / "profiles.json") if x["id"] == "lan")
    cfg = PersonaConfig.load(OLD / "implementation_snapshot/personas" / p["config"])
    for k in ("name", "voice", "attention", "bias"):
        setattr(cfg.role, k, p[k])
    cfg.role.identity = p["spec"]
    cfg.role.phrases, cfg.role.special = {}, []
    for k, v in p["traits"].items():
        trait = cfg.role.traits[k]
        trait.anchor = trait.value = v
        trait.lo, trait.hi, trait.drift = 0., 1., 1.
    cfg.validate()
    return cfg


def baseline(cfg, native):
    selected = load(OLD / "selection.json")["selected_id"]
    handbook = next(c["system"] for c in load(OLD / "prompts.json")["candidates"] if c["id"] == selected)
    # Exact context from the actual full request (including recalled versions).
    split = "\n\n[你和这个人之间]\n"
    if native[0]["content"].count(split) != 1:
        raise ValueError("could not isolate the shared state")
    context = "[你和这个人之间]\n" + native[0]["content"].split(split)[1]
    role = cfg.role
    spec = dict(name=role.name, spec=role.identity, voice=role.voice, attention=role.attention,
                attention_hint=prompt._ATTENTION_HINT[role.attention], bias=role.bias,
                legibility_max=role.legibility_max, trait_guidance=prompt._trait_block(cfg).replace(
                    "baseline/SAY", "实际回复").replace("direct", "基础判断"),
                stance_guidance=prompt.stance_policy(cfg).replace("baseline/SAY", "实际回复").replace("direct", "基础判断"))
    return [dict(role="system", content=handbook+"\n\n[共享人格规格]\n"+json.dumps(spec, ensure_ascii=False)
                 +"\n\n"+context), *visible_dialogue(native[1:-1]), native[-1]]


def trial(query, route, send, snapshot=None):
    with tempfile.TemporaryDirectory() as td:
        if snapshot:
            shutil.copytree(snapshot, td, dirs_exist_ok=True)
        cfg = config()
        captured = []
        def turn(msgs):
            if captured:
                raise RuntimeError("extra foreground call")
            actual = msgs if route == "full" else baseline(cfg, msgs)
            captured.append(actual)
            return send(actual) if route == "full" else "占位"
        with Session(cfg, Store(td), turn, options=SessionConfig(memory_mode="frozen",
                     consolidate_on_exit=False)) as session:
            result = session.say(query)
        if len(captured) != 1 or captured[0][-1] != {"role":"user","content":query}:
            raise ValueError("unexpected foreground input")
        if route == "strong":
            say = send(captured[0]).strip()
            return captured[0], say, {"valid":None, "scope":"no_duplex_protocol"}
        return captured[0], result.say, result.validity


def short_jobs():
    jobs = []
    for case in load(HERE/"cases.json"):
        for mode in ("choice", "bounded", "open"):
            for route in ("full", "strong"):
                for rep in (1, 2):
                    jobs.append(dict(id=f"{case['id']}-{mode}-{route}-{rep}", case=case["id"],
                        mode=mode, route=route, rep=rep, user=case["situation"]+case[mode]))
    random.Random(371119).shuffle(jobs)
    return jobs


def freeze(stage, jobs, snapshot=None):
    path = HERE/f"{stage}_registration.json"
    if path.exists():
        raise FileExistsError(path)
    reqs, plan = {}, []
    for j in jobs:
        msgs, _, _ = trial(j["user"], j["route"], lambda _: "预备占位", snapshot)
        body = {**PARAMS, "messages":msgs}
        h = digest(body)
        reqs[h] = body
        plan.append({**j, "request":h})
    tracked = [HERE/"study.py", HERE/"cases.json", HERE/"protocol.md",
               OLD/"profiles.json", OLD/"prompts.json", OLD/"selection.json",
               OLD/"implementation_snapshot/personas/shorekeeper.toml", *sorted((ROOT/"persona").glob("*.py"))]
    if stage == "long":
        tracked += [HERE/"trajectory.py", HERE/"trajectory.json", HERE/"long_preflight.json"]
    write(path, dict(stage=stage, source_sha256={p.relative_to(ROOT).as_posix():
          hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
          endpoint=ENDPOINT, jobs=plan, requests=reqs))


def execute(stage, key, snapshot=None):
    reg = load(HERE/f"{stage}_registration.json")
    if not all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in reg["source_sha256"].items()):
        raise ValueError("registration source mismatch")
    out = HERE/f"{stage}_calls.json"
    # A failed or interrupted real run cannot be silently retried.
    with out.open("x", encoding="utf-8") as f:
        f.write("[]")
    records = []
    failures = 0
    for j in reg["jobs"]:
        row = dict(id=j["id"], status="not_sent", say=None, raw=None, usage=None)
        records.append(row)
        if failures >= 2:
            write(out, records)
            continue
        fn = openai_compatible(**PARAMS, base_url=ENDPOINT, api_key=key, timeout=45)
        def send(msgs):
            if digest({**PARAMS, "messages":msgs}) != j["request"]:
                raise ValueError("request mismatch")
            row.update(status="started")
            write(out, records)
            start=time.monotonic()
            raw=fn(msgs)
            row.update(raw=raw, elapsed_s=round(time.monotonic()-start,3))
            return raw
        try:
            _, say, valid = trial(j["user"], j["route"], send, snapshot)
            row.update(status="ok" if say else "empty", say=say, validity=valid)
        except (OSError, RuntimeError) as exc:
            import re
            # Do not export provider messages, credentials or local paths.
            http=re.search(r"HTTP (\d{3})", str(exc))
            row.update(status="error", error_type=type(exc).__name__, http_status=http.group(1) if http else None)
            failures += 1
            if http and http.group(1) in ("401","403","402"):
                failures = 2
        response = getattr(fn.observations, "response", None) or {}
        row.update(usage=response.get("usage"), response_model=response.get("model"),
                   finish=response.get("finish_reason"))
        write(out, records)
        print(j["id"], row["status"], flush=True)


def packet(stage):
    reg=load(HERE/f"{stage}_registration.json")
    calls={r["id"]:r for r in load(HERE/f"{stage}_calls.json")}
    items=[(j,calls[j["id"]]) for j in reg["jobs"] if calls[j["id"]]["say"]]
    random.Random(91871).shuffle(items)
    public=[]; mapping={}
    for i,(j,r) in enumerate(items,1):
        ident=f"{stage[0]}{i:02d}"
        public.append(dict(id=ident,user=j["user"],say=r["say"]))
        mapping[ident]=j["id"]
    write(HERE/f"{stage}_blind.json",public)
    return mapping


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("command",choices=("freeze","live","packet"))
    ap.add_argument("stage",choices=("short","long"))
    args=ap.parse_args()
    if args.command=="packet":
        packet(args.stage)
    else:
        with tempfile.TemporaryDirectory() as td:
            snap=None
            if args.stage=="short":
                jobs=short_jobs()
            else:
                import trajectory
                snap=Path(td)
                jobs=trajectory.build(snap, register=args.command=="freeze")
            if args.command=="freeze":
                freeze(args.stage,jobs,snap)
            else:
                execute(args.stage,os.getenv("PERSONA_API_KEY") or getpass.getpass("临时实验凭据: "),snap)
