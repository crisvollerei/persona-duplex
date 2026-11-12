"""Exploratory reread of the 12 existing natural items with the frozen relation method.
No human files, reference states, model predictions from earlier arms, or scores are inputs.
Only live contacts the provider; replay reproduces the registered requests and derived states.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from threading import Lock
import runner as frozen

HERE=Path(__file__).resolve().parent
OUT=HERE/"bridge_observed"
SEED=7111361

def run(mode,out):
    frozen.selftest()
    main=frozen.load(HERE.parent/"design.json");d=frozen.load(HERE/"design.json")
    ppl=main["people"];loc=main["location"];src_cases=HERE.parent/"cases.json"
    raw=frozen.load(src_cases)
    assert len(raw)==12 and len({x["id"] for x in raw})==12
    # Construct a data-only input containing neither the pair identifiers nor any annotations.
    cases={x["id"]:dict(location=loc,turns=[dict(speaker=t["speaker"],text=t["text"]) for t in x["turns"]]) for x in raw}
    plan=sorted(cases);random.Random(SEED).shuffle(plan)
    def key(i):return i+".relation.A"
    msgs={i:frozen.stimulus(c,ppl,"relation") for i,c in cases.items()}
    for i,c in cases.items():
        assert json.loads(msgs[i][1]["content"])==dict(location=loc,people=ppl,dialogue=[dict(turn=n,**t) for n,t in enumerate(c["turns"],1)])
        assert msgs[i][0]=={"role":"system","content":frozen.REL}
    srcs=[Path(__file__),HERE/"runner.py",HERE/"design.json",src_cases,HERE.parent/"design.json",HERE.parent/"study.py",HERE.parent/"method.py",
          HERE.parent.parent/"arrival_shadow/observer.py",frozen.ROOT/"persona/llm.py"]
    reg=dict(purpose="post_hoc_bridge_existing_natural_items",seed=SEED,model=d["model"],order=list(map(key,plan)),
             inputs={p.relative_to(frozen.ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in srcs},
             request_sha256={key(i):frozen.digest(m) for i,m in msgs.items()})
    def save(fn,v):
        if mode=="dry":return
        f=out/fn
        if mode=="replay" or f.exists():assert frozen.load(f)==v,fn
        else:frozen.dump(f,v)
    if mode=="live":out.mkdir(parents=True,exist_ok=True)
    save("registration.json",reg)
    cache={};lock=Lock();log=out/"calls.jsonl.gz"
    if mode!="dry" and log.exists():
        with gzip.open(log,"rb") as fp:rr=[json.loads(l) for l in fp]
        cache={c["id"]:c for c in rr};assert len(rr)==len(cache) and set(cache)<=set(reg["order"])
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")) if mode=="live" and len(cache)<len(plan) else None
    def sample(i):
        m=msgs[i];k=key(i)
        if k in cache:
            rec=cache[k];assert rec["messages"]==m and rec["params"]==reg["model"];ans=rec["raw"]
        elif mode=="dry":
            ans=json.dumps(dict(people=[dict(person=p,evidence=[],decisive_turn=None,place="undetermined",time="undetermined",arrival_need="not_established",basis="none") for p in ppl]),ensure_ascii=False)
        else:
            if mode!="live" or not secret:raise RuntimeError("Missing receipt or credential")
            api=frozen.openai_compatible(**reg["model"],api_key=secret,base_url="https://api.deepseek.com/v1",timeout=90)
            try:ans=api(m);rec=dict(ok=True,raw=ans,response=api.observations.response)
            except Exception as exc:
                if not cache or any(c in str(exc) for c in ("401","402","403")):raise RuntimeError("Provider unavailable: "+type(exc).__name__) from None
                ans="";rec=dict(ok=False,raw=ans,error=type(exc).__name__,response=api.observations.response)
            rec.update(id=k,messages=m,params=reg["model"])
            with lock:
                with gzip.open(log,"ab") as fp:fp.write(frozen.wire(rec)+b"\n")
                cache[k]=rec
        return k,dict(people=frozen.relation(ans,cases[i],ppl),request=frozen.digest(m))
    pred=dict([sample(plan[0])])
    with ThreadPoolExecutor(max_workers=4) as ex:pred.update(ex.map(sample,plan[1:]))
    if mode=="dry":assert all(v["state"]=="unknown" for c in pred.values() for v in c["people"].values())
    else:assert set(cache)==set(pred)
    save("predictions.json",pred);print(mode,len(pred),"unscored bridge receipts checked",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"))
    ap.add_argument("--out",type=Path,default=OUT)
    args=ap.parse_args();run(args.mode,args.out)
