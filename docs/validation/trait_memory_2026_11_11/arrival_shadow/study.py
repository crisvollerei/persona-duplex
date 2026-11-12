"""Offline copy -> research-only observer -> local receipts. No Session, Store, or user-reply callback."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import gzip
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import random
import sys
from threading import Lock
from observer import messages,classify,selfcheck,group
from score import load,dump

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from persona.llm import openai_compatible
def wire(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
def run(mode,out):
    selfcheck()
    cases={x["id"]:x for x in load(HERE/"cases.json")};d=load(HERE/"design.json")
    plan=list(itertools.product(cases,d["forms"],d["orders"]))
    random.Random(d["seed"]).shuffle(plan)
    assert len(plan)==len(set(plan))==64
    src=[HERE/f for f in ("study.py","observer.py","score.py","design.json","cases.json")]+[ROOT/"persona/llm.py"]
    reg=dict(order=[".".join(p) for p in plan],inputs={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in src},
             model=d["model"])
    def save(f,v):
        if mode=="dry":return
        p=out/f
        if mode=="replay" or p.exists():assert load(p)==v,f
        else:dump(p,v)
    if mode=="live":out.mkdir(parents=True,exist_ok=True)
    save("registration.json",reg)
    log=out/"calls.jsonl.gz";cache={};lock=Lock()
    if mode!="dry" and log.exists():
        with gzip.open(log,"rb") as fp:rr=[json.loads(l) for l in fp]
        cache={r["id"]:r for r in rr};assert len(rr)==len(cache) and set(cache)<=set(reg["order"])
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")) if mode=="live" and len(cache)<len(plan) else None
    def sample(job):
        case,f,o=job;k=".".join(job);msgs=messages(cases[case],f,o)
        if k in cache:
            rec=cache[k];assert rec["messages"]==msgs and rec["params"]==reg["model"];raw=rec["raw"]
        elif mode=="dry":
            raw=json.dumps({"people":[dict(person=p,state="unknown",evidence=[]) for p in d["people"]]},ensure_ascii=False)
        else:
            if mode!="live" or not secret:raise RuntimeError("Missing receipt or credential")
            api=openai_compatible(**reg["model"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=60)
            try:raw=api(msgs);rec=dict(ok=True,raw=raw,response=api.observations.response)
            except Exception as exc:
                if not cache or any(z in str(exc) for z in ("401","402","403")):raise RuntimeError(type(exc).__name__) from None
                raw="";rec=dict(ok=False,raw=raw,response=api.observations.response,error=type(exc).__name__)
            rec.update(id=k,messages=msgs,params=reg["model"])
            with lock:
                with gzip.open(log,"ab") as fp:fp.write(wire(rec)+b"\n")
                cache[k]=rec
        people=classify(raw,cases[case],f)
        return k,dict(people=people,group=group(r["state"] for r in people.values()),request=sha256(wire(msgs)).hexdigest())
    rows=dict([sample(plan[0])])
    with ThreadPoolExecutor(max_workers=4) as ex:rows.update(ex.map(sample,plan[1:]))
    if mode!="dry":assert set(rows)==set(cache)
    for case in cases.values():
        for o in d["orders"]:
            assert messages(case,"evidence",o)[1]==messages(case,"direct",o)[1]
    save("predictions.json",rows)
    print(mode,len(rows),"observer calls checked; no dialogue reply generated")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"));ap.add_argument("--out",type=Path,default=HERE/"observed")
    args=ap.parse_args();run(args.mode,args.out)
