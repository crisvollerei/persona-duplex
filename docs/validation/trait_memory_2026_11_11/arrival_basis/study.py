"""Blinded model arm of the study. No human files, conversation engine, or memory store are read."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from method import messages,classify,selfcheck,old
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from persona.llm import openai_compatible
OLD=HERE.parent/"arrival_shadow"
def load(p):return json.loads(p.read_bytes())
def wire(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def dump(p,v):
    tmp=p.with_suffix(".tmp");tmp.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+"\n").encode());tmp.replace(p)
def inputs():
    d=load(HERE/"design.json");cases={x["id"]:x for x in load(HERE/"cases.json")};cal={x["id"]:x for x in load(OLD/"cases.json")}
    assert len(cases)==12 and len(cal)==16
    for pair in {c["pair"] for c in cases.values()}:
        left,right=[c["turns"] for c in cases.values() if c["pair"]==pair]
        assert [t["speaker"] for t in left]==[t["speaker"] for t in right] and sum(u!=v for u,v in zip(left,right))==1
    ctl={i:dict(id=i,turns=c["turns"]+[dict(speaker="现场核验员",text=d["updates"][(int(i)-1)%3]["text"])]) for i,c in cases.items()}
    data={"new":cases,"old":cal,"control":ctl}
    plan=[(stage,i,m,o) for stage,tbl in data.items() for i in tbl for o in (("A","B","A2") if stage=="new" else ("A",)) for m in ("legacy","typed")]
    assert len(plan)==len(set(plan))==128
    random.Random(d["seed"]).shuffle(plan)
    return d,data,plan
def stimulus(d,data,job):
    stage,i,m,o=job;c=data[stage][i]
    ppl=old.PEOPLE if stage=="old" else d["people"]
    loc="展厅内的签收台" if stage=="old" else d["location"]
    msgs=messages(c,ppl,loc,m,o)
    if stage=="old" and m=="legacy":assert msgs==old.messages(c,"evidence","YC")
    return msgs,c,ppl
def run(mode,out):
    selfcheck();d,data,plan=inputs()
    source=[HERE/n for n in ("cases.json","design.json","study.py","method.py")]+[OLD/"observer.py",OLD/"cases.json",ROOT/"persona/llm.py"]
    reg=dict(order=[".".join(j) for j in plan],model=d["model"],sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    def save(f,v):
        if mode=="dry":return
        p=out/f
        if mode=="replay" or p.exists():assert load(p)==v,f
        else:dump(p,v)
    if mode=="live":out.mkdir(exist_ok=True,parents=True)
    save("registration.json",reg)
    log=out/"calls.jsonl.gz";lock=Lock();cache={}
    if mode!="dry" and log.exists():
        with gzip.open(log,"rb") as fp:rr=[json.loads(l) for l in fp]
        cache={r["id"]:r for r in rr};assert len(cache)==len(rr) and set(cache)<=set(reg["order"])
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")) if mode=="live" and len(cache)<len(plan) else None
    def sample(job):
        stage,i,m,o=job;k=".".join(job);msgs,c,ppl=stimulus(d,data,job)
        if k in cache:
            assert cache[k]["messages"]==msgs and cache[k]["params"]==d["model"]
            raw=cache[k]["raw"]
        elif mode=="dry":raw=json.dumps(dict(people=[dict(person=p,state="unknown",basis="none",evidence=[]) for p in ppl]),ensure_ascii=False)
        else:
            if mode!="live" or not secret:raise ValueError("Missing credential or receipt")
            api=openai_compatible(**d["model"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=80)
            try:
                raw=api(msgs);rec=dict(ok=True,response=api.observations.response,raw=raw)
            except Exception as exc:
                if not cache or any(w in str(exc) for w in ("401","402","403")):raise RuntimeError(type(exc).__name__) from None
                raw="";rec=dict(ok=False,response=api.observations.response,raw=raw,error=type(exc).__name__)
            rec.update(id=k,messages=msgs,params=d["model"])
            with lock:
                with gzip.open(log,"ab") as fp:fp.write(wire(rec)+b"\n")
                cache[k]=rec
                if len(cache)%32==0:print(len(cache),"archived",flush=True)
        return k,dict(request=hashlib.sha256(wire(msgs)).hexdigest(),people=classify(raw,c,ppl,m))
    rows=dict([sample(plan[0])])
    with ThreadPoolExecutor(max_workers=4) as ex:rows.update(ex.map(sample,plan[1:]))
    for i in data["new"]:
        for m in ("legacy","typed"):assert rows[f"new.{i}.{m}.A"]["request"]==rows[f"new.{i}.{m}.A2"]["request"]
    for stage,i,m,o in plan:
        assert stimulus(d,data,(stage,i,"typed",o))[0][1]==stimulus(d,data,(stage,i,"legacy",o))[0][1]
    for stage,name in (("new","predictions"),("new","repeats"),("old","calibration"),("control","controls")):
        save(name+".json",{k.split(".",1)[1]:v for k,v in rows.items() if k.startswith(stage+".") and (stage!="new" or (name=="repeats")==k.endswith(".A2"))})
    if mode!="dry":assert set(cache)==set(rows)
    print(mode,len(rows),"observer slots checked")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"));ap.add_argument("--out",type=Path,default=HERE/"observed")
    args=ap.parse_args();run(args.mode,args.out)
