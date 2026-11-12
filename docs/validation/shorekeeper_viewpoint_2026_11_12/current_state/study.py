"""One-step counterfactuals from frozen foreground requests; no chat or memory store writes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import getpass
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
from threading import Lock

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parents[2]
OUT=HERE/"observed"
sys.path.insert(0,str(ROOT))
from persona.config import PersonaConfig
from persona.generation import parse_output, validity
from persona.llm import openai_compatible
from persona.parse import plain_text
from statecard import render_card


def read(path): return json.loads(path.read_bytes())
def digest(b): return hashlib.sha256(b).hexdigest()
def wire(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()


def calls(path):
    if not path.exists(): return {}
    with gzip.open(path,"rb") as f: rows=[json.loads(line) for line in f]
    by_id={r["id"]:r for r in rows}
    if len(rows)!=len(by_id):raise ValueError("duplicate receipts")
    return by_id


def save(path,obj,mode):
    if mode=="dry":return
    if mode=="replay" or path.exists():
        if read(path)!=obj:raise ValueError("frozen artifact differs: "+path.name)
    else:
        tmp=path.with_suffix(".tmp")
        tmp.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+"\n").encode())
        tmp.replace(path)


def visible(raw, cfg):
    parsed=parse_output(raw,"full")
    return (parsed.say or plain_text(raw),
            validity(parsed,"full",protocol="text",
                     allowed=cfg.can_displace(parsed.head.displace) if parsed.head else True))


def run(mode):
    plan=read(HERE/"protocol.json")
    for p,h in plan["parents"].items():
        if digest((PARENT/p).read_bytes())!=h:raise ValueError("parent fingerprint: "+p)
    src=calls(PARENT/"observed/calls.jsonl.gz")
    old={(r["variant"],r["replicate"],r["act"]):r for r in read(PARENT/"observed/turns.json") if r["condition"]=="scoped"}
    variants=read(PARENT/"scenes.json")["variants"]
    cfg=PersonaConfig.load(ROOT/"personas/shorekeeper.toml")
    points=list(itertools.product(sorted(v["id"] for v in variants),"AB",plan["acts"]))
    rng=random.Random(plan["seed"]);rng.shuffle(points)
    jobs=[]
    for v,h,a in points:
        block=list(itertools.product([v],[h],[a],plan["conditions"],plan["samples"]))
        rng.shuffle(block);jobs.extend(block)
    def ident(job):return ".".join(str(x) for x in job)
    def inputs(job):
        v,h,a,c,s=job; parent_id=f"{v}.scoped.{h}.{a}"
        original=src[parent_id]
        if not original["ok"] or original["params"]!=plan["provider"]:raise ValueError(parent_id)
        msgs=deepcopy(original["messages"])
        assert [m["content"] for m in msgs if m["role"]=="user"]==[plan["question"]]*a
        assert msgs[-1]==dict(role="user",content=plan["question"]) and msgs[0]["role"]=="system"
        produced=visible(original["raw"],cfg); archived=old[v,h,a]
        assert produced==(archived["say"],archived["validity"]),(parent_id,"parser mismatch")
        if c=="state_card":msgs[0]["content"]+="\n\n"+render_card(a)
        return parent_id,msgs
    sources=[*sorted((ROOT/"persona").glob("*.py")),ROOT/"personas/shorekeeper.toml",
             *(PARENT/p for p in plan["parents"]),*(HERE/p for p in ("study.py","statecard.py","protocol.json","semantic.py"))]
    frozen=dict(order=[ident(j) for j in jobs],model=plan["provider"],
                sha256={p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in sources},
                request_sha256={ident(j):digest(wire(inputs(j)[1])) for j in jobs})
    if mode=="live":OUT.mkdir(exist_ok=True)
    save(OUT/"registration.json",frozen,mode)
    log=OUT/"calls.jsonl.gz"
    cache=calls(log) if mode!="dry" else {}
    if not set(cache)<=set(frozen["order"]):raise ValueError("unexpected slots")
    lock=Lock()
    key=None
    if mode=="live" and len(cache)<len(jobs):
        key=os.getenv("PERSONA_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("Provider credential: ")
    def sample(job):
        v,h,a,c,s=job;uid=ident(job);parent_id,msgs=inputs(job)
        if uid in cache:
            rec=cache[uid]
            if msgs!=rec["messages"] or rec["params"]!=plan["provider"] or rec["parent_id"]!=parent_id:
                raise ValueError("changed request: "+uid)
        elif mode=="dry":
            rec=dict(ok=True,raw="<<<HEAD>>>\nneed: 认同\nconfidence: 0.6\ndirect: 好。\nbaseline: 好。\ndisplace: none\n<<<SAY>>>\n好。\n<<<TAIL>>>\nsettle: -\ndelta: -\npending: -")
        else:
            if mode!="live" or not key: raise ValueError("missing receipt: "+uid)
            client=openai_compatible(**plan["provider"],base_url="https://api.deepseek.com/v1",api_key=key,timeout=90)
            rec=dict(id=uid,parent_id=parent_id,messages=msgs,params=plan["provider"])
            try:
                raw=client(msgs)
                rec.update(ok=True,raw=raw,response=client.observations.response)
            except Exception as exc:
                rec.update(ok=False,raw="",response=client.observations.response,error=type(exc).__name__)
            with lock:
                with gzip.open(log,"ab") as stream:stream.write(wire(rec)+b"\n")
                cache[uid]=rec
                if len(cache)%12==0:print(len(cache),"receipts",flush=True)
        say,valid=visible(rec["raw"],cfg) if rec["ok"] else ("",None)
        if mode=="dry":assert say=="好。" and valid["valid"]
        return dict(id=uid,parent_id=parent_id,variant=v,history=h,act=a,condition=c,sample=s,
                    status="delivered" if say else ("empty" if rec["ok"] else "error"),say=say,validity=valid)
    rows=[sample(jobs[0])]
    if mode=="live" and not cache[ident(jobs[0])]["ok"]:
        raise RuntimeError("First provider request failed; halted before remaining calls.")
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows.extend(pool.map(sample,jobs[1:]))
    rows.sort(key=lambda r:r["id"])
    if mode!="dry":
        assert len(cache)==len(rows)==48
        rid=[r["response"]["id"] for r in cache.values() if r["ok"]]
        assert all(rid) and len(set(rid))==len(rid) and not set(rid)&{r["response"]["id"] for r in src.values() if r["ok"]}
    save(OUT/"turns.json",rows,mode)
    print(mode,dict(Counter(r["status"] for r in rows)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("dry","live","replay"))
    run(p.parse_args().mode)
