"""Three-act role-view prototype; replay requires no key. Does not alter the persona package."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from viewpoint import compile_view, render_view, role_for_view

OUT = HERE / "observed"


def read(p):
    return json.loads(p.read_bytes())


def wire(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_or_check(path, obj, mode):
    if mode == "dry":
        return
    if mode == "replay" or path.exists():
        if read(path) != obj:
            raise ValueError(f"Frozen artifact differs: {path.name}")
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode())
        tmp.replace(path)


def character(v, act, condition, base):
    view = compile_view(v, act)
    cfg = role_for_view(base, view)
    if condition == "full_notes":
        notes = [e["world"] for e in v["events"]]
        cfg.role.identity += ("\n\n[作者的全篇大纲，仅供创作参考；含尚未发生的事件，不代表此刻人物已经得知。]"
                              + "\n" + "\n".join(notes) + "\n作者档案校对码：" + v["private_nonce"])
    return cfg, view


def main(mode):
    plan, scenes = read(HERE/"protocol.json"), read(HERE/"scenes.json")
    q = scenes["question"]; vs = {v["id"]:v for v in scenes["variants"]}
    assert q == plan["question"] and len(vs) == len(scenes["variants"]) == 3
    role_path = ROOT/"personas/shorekeeper.toml"
    jobs = [(i,c,r) for i in sorted(vs) for c in plan["methods"] for r in plan["replicates"]]
    random.Random(plan["seed"]).shuffle(jobs)
    files = [*sorted((ROOT/"persona").glob("*.py")), role_path,
             *(HERE/p for p in ["study.py","viewpoint.py","protocol.json","scenes.json","rubric.md","metrics.py"])]
    reg = dict(trajectories=[".".join(j) for j in jobs], model=plan["provider"],
               sha256={p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in files})
    if mode == "live": OUT.mkdir(exist_ok=True)
    write_or_check(OUT/"registration.json", reg, mode)
    log = OUT/"calls.jsonl.gz"
    cache = {}
    if mode != "dry" and log.exists():
        with gzip.open(log,"rb") as f:
            for line in f:
                e=json.loads(line)
                assert e["id"] not in cache
                cache[e["id"]]=e
    lock=Lock()
    secret=None
    if mode=="live" and not (OUT/"turns.json").exists():
        secret=os.getenv("PERSONA_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("Provider credential: ")
    def run_job(job):
        i,c,r=job; v=vs[i]; rows=[]; ctx={}; upstream=False
        lookups=[v["source"],*v["aliases"]]
        def call(msgs):
            act=ctx["act"]; ix=f"{i}.{c}.{r}.{act}"; assert not ctx["called"]; ctx["called"]=True
            all_text="\n".join(m["content"] for m in msgs)
            assert msgs[-1]=={"role":"user","content":q}
            assert [m["content"] for m in msgs if m["role"]=="user"]==[q]*act
            # Inspect the messages actually about to be sent, not just a helper's return value.
            assert ctx["render"] in all_text or json.dumps(ctx["render"],ensure_ascii=False)[1:-1] in all_text
            if c=="scoped":
                assert v["private_nonce"] not in all_text
                if act==1: assert not any(n in all_text for n in lookups)
                if act<3:
                    for e in v["events"][act:]:
                        assert e["shorekeeper"] not in all_text and e["traveler"] not in all_text
            else:
                assert v["private_nonce"] in all_text and v["source"] in all_text
            if act>1: assert v["source"] in all_text
            if ix in cache:
                e=cache[ix]
                if msgs != e["messages"] or e["params"] != reg["model"]:
                    raise ValueError("Changed replay request: "+ix)
            elif mode=="dry":
                return "<<<HEAD>>>\nneed: 认同\nconfidence: 0.6\ndirect: 好。\nbaseline: 好。\ndisplace: none\n<<<SAY>>>\n好。\n<<<TAIL>>>\nsettle: -\ndelta: -\npending: -"
            else:
                if mode!="live" or not secret: raise ValueError("Missing receipt: "+ix)
                api=openai_compatible(**reg["model"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90)
                e=dict(id=ix,params=reg["model"],messages=deepcopy(msgs))
                try:
                    raw=api(msgs)
                    e.update(ok=True,raw=raw,response=api.observations.response)
                except Exception as exc:
                    e.update(ok=False,raw="",response=api.observations.response,error=type(exc).__name__)
                with lock:
                    with gzip.open(log,"ab") as fp: fp.write(wire(e)+b"\n")
                    cache[ix]=e
                    if len(cache)%9==0: print(len(cache),"receipts",flush=True)
            if not e["ok"]: raise ConnectionError("Archived call failed")
            return e["raw"]
        base=PersonaConfig.load(role_path)
        # A change to hidden future and author facts must not alter the first scoped role's identity.
        swapped=deepcopy(v)
        swapped["source"]="另一个纸面名字"; swapped["private_nonce"]="另一个校对码"
        swapped["events"][0]["world"]="另一版作者资料"
        for e in swapped["events"][1:]:
            for f in ("shorekeeper","traveler","world"): e[f]="尚未发生的替代情节"
        assert character(swapped,1,"scoped",base)[0].role.identity==character(v,1,"scoped",base)[0].role.identity
        with tempfile.TemporaryDirectory(prefix="persona-view-") as work:
            for act in plan["acts"]:
                row=dict(variant=i,condition=c,replicate=r,act=act,user=q,say="",validity=None,status="upstream_missing")
                if not upstream:
                    cfg,view=character(v,act,c,base)
                    ctx.update(act=act,called=False,render=render_view(view))
                    # Reopen the same private branch ledger with the next, independently copied role view.
                    # The default engine restores this branch's real (not synthetic) prior histories.
                    with Session(cfg,Store(work),call,protocol="text",
                                 options=SessionConfig(memory_mode="frozen",consolidate_on_exit=False)) as session:
                        try:
                            out=session.say(q)
                            row.update(status="delivered" if out.say else "empty",say=out.say,validity=out.validity)
                        except ConnectionError:
                            row.update(status="error");upstream=True
                    assert ctx["called"]
                rows.append(row)
        return rows
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for group in pool.map(run_job,jobs): rows.extend(group)
    rows.sort(key=lambda e:(e["variant"],e["condition"],e["replicate"],e["act"]))
    assert len(rows)==36
    if mode=="dry": assert all(e["say"]=="好。" for e in rows)
    if mode!="dry":
        assert set(cache)=={f"{e['variant']}.{e['condition']}.{e['replicate']}.{e['act']}" for e in rows if e["status"]!="upstream_missing"}
        ids=[e["response"]["id"] for e in cache.values() if e["ok"]]
        assert all(ids) and len(set(ids))==len(ids)
    write_or_check(OUT/"turns.json",rows,mode)
    print(mode,len(rows),"slots,",sum(e["status"]=="delivered" for e in rows),"visible replies")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("dry","live","replay"))
    main(ap.parse_args().mode)
