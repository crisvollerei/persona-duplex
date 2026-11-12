"""Fixed-evidence disclosure experiment. No provider access in dry or replay."""
import argparse
from collections import Counter
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
from threading import Lock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"attribution"))
import study as attr

base=attr.prev
read,write=base.read,base.write
design=read(HERE/"design.json")
scenes=read(HERE/"scenes.json")
ARMS={"tang":("F","S","G","N","H"),"yu":("F","S","G")}
HINT="\n- 几乎不谈自己。"

def states():
    pairs,labels=attr.exchanges()
    refs=read(attr.HERE/"reference.json")["rows"];record={}
    for k,p in pairs.items():
        if not k.startswith("H"):continue
        ep,a=k.split(".");label=labels[k]
        reason=refs[int(ep[1:])-1][base.ROLES.index(a)][1]
        record[k]={"reference":label,"reason":reason,"quotes":[p.first,p.second],
                   "eligible":p.protocol_ok and label is not None,
                   "vote":label if p.protocol_ok and label is not None else 0}
    traj=attr.evolve(record)
    expected=read(attr.HERE/"results.json")["reference_only_states"]
    for a,rows in traj.items():
        assert [i for i,row in enumerate(rows,1) if row["push"]]==expected[a]["pushed"]
        assert [base.BASE[a],rows[11]["growing"],rows[-1]["growing"]]==expected[a]["values"]
        assert all(row["frozen"]==base.BASE[a] for row in rows)
    return record,traj

def prompt_cell(messages,v,adjust):
    """Return the transmitted prompt and its independently normalised comparison copy."""
    header=f"- self_disclosure={v:.4f}（0：几乎不谈自己。 1：可以说说自己。）"
    m=deepcopy(messages);s=m[0]["content"]
    assert s.count(header)==1
    before=s.count(header+HINT)
    assert before==int(v<=.35) and v<.65
    if adjust=="add":
        assert not before;s=s.replace(header,header+HINT,1)
    elif adjust=="drop":
        assert before;s=s.replace(header+HINT,header,1)
    else:assert adjust is None
    m[0]["content"]=s
    present=s.count(header+HINT)
    norm=deepcopy(m);norm[0]["content"]=s.replace(header+(HINT if present else ""),"- DISCLOSURE_CELL")
    return m,base.digest(norm),bool(present)

def suite():
    jobs=[(a,b,q,r) for a,arms in ARMS.items() for b in arms for q in scenes for r in (1,2)]
    random.Random(831196).shuffle(jobs)
    return jobs

def key(job):
    a,b,q,r=job
    return f"{a}.{b}.{q['id']}.{r}"

def fingerprints():
    oldreg=read(attr.OLD/"observed/registration.json")["files"]
    for f,h in oldreg.items():assert hashlib.sha256((base.ROOT/f).read_bytes()).hexdigest()==h
    files=[HERE/f for f in ("experiment.py","design.json","scenes.json")]
    files.extend([attr.OLD/"run.py",attr.HERE/"study.py",attr.HERE/"reference.json",attr.HERE/"results.json",
                  attr.OLD/"observed/trajectory.json",attr.OLD/"observed/registration.json"])
    return {str(p.relative_to(base.ROOT).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def experiment(mode,out):
    marks,traj=states();jobs=suite()
    reg={"files":fingerprints(),"order":[key(j) for j in jobs],
         "foreground":base.D["foreground"],"grader":dict(attr.judge,max_tokens=1400)}
    def save(f,v):
        if mode=="dry":return
        if mode=="live":write(out/f,v)
        else:assert read(out/f)==v,f
    records={};seen=set();lock=Lock();token=None
    if mode!="dry":
        if mode=="live":out.mkdir(parents=True,exist_ok=True)
        rp=out/"registration.json";log=out/"calls.jsonl.gz"
        if rp.exists():assert read(rp)==reg
        else:
            assert mode=="live" and not any(out.iterdir());save("registration.json",reg)
        if log.exists():
            entries=base.lines(log);records={r["id"]:r for r in entries};assert len(records)==len(entries)
        if mode=="live":token=os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")
    save("states.json",dict(experiences=marks,trajectories=traj))
    def send(k,m,params):
        with lock:
            assert k not in seen;seen.add(k)
            if k in records:
                z=records[k];assert z["messages"]==m and z["params"]==params,k
                assert z["ok"],"Failed request exists; do not replace samples."
                return z["raw"]
        if mode=="dry":return base.stub("turn",m)
        if mode=="replay":raise ValueError("missing archived call: "+k)
        f=base.openai_compatible(**params,base_url="https://api.deepseek.com/v1",api_key=token,timeout=90)
        try:raw=f(m);record=dict(ok=True,raw=raw,response=f.observations.response)
        except Exception as ex:record=dict(ok=False,error=type(ex).__name__)
        record.update(id=k,messages=m,params=params)
        with lock:
            with gzip.open(log,"ab") as fh:fh.write(base.wire(record)+b"\n")
            records[k]=record
        if not record["ok"]:raise RuntimeError(f"{k}: {record['error']}")
        return raw
    contexts={a:base.context(attr.hist,a) for a in ARMS}
    def sample(j):
        a,b,q,r=j;k=key(j);v=traj[a][-1]["growing"] if b in ("G","N") else base.BASE[a]
        adjust={"N":"add","H":"drop"}.get(b);captured=[]
        def callback(m):
            actual,n,h=prompt_cell(m,v,adjust);captured.append((base.digest(actual),n,h))
            return send("sample."+k,actual,reg["foreground"])
        answer,_=base.turn(a,v,q["public"],q[a],contexts[a],q["line"],callback)
        assert len(captured)==1
        req,normalized,hint=captured[0]
        return k,dict(**answer,value=v,request=req,normalized=normalized,low_hint=hint)
    with ThreadPoolExecutor(max_workers=4) as ex:probes=dict(ex.map(sample,jobs))
    for a in ARMS:
        for q in scenes:
            variants={b:[probes[key((a,b,q,r))] for r in (1,2)] for b in ARMS[a]}
            assert len({v["normalized"] for rows in variants.values() for v in rows})==1
            assert len({v["request"] for b in ("F","S") for v in variants[b]})==1
            assert all(z[0]["request"]==z[1]["request"] for z in variants.values())
            assert len({z[0]["request"] for z in variants.values()})==len(ARMS[a])-1
            assert all(v["low_hint"]==(a=="tang" and b in ("F","S","N")) for b,vs in variants.items() for v in vs)
    save("probes.json",probes)
    if mode=="dry":
        print(len(seen),"matched sample requests; all factorial cells and historical states verified.");return
    mapping_file=out/"key.json"
    if not mapping_file.exists():
        assert mode=="live"
        ids=sorted(probes);random.SystemRandom().shuffle(ids)
        save("key.json",{f"P{i:02}":v for i,v in enumerate(ids,1)})
    mapping=read(mapping_file);assert len(mapping)==len(probes)==96 and set(mapping.values())==set(probes)
    packet={}
    for bid,k in mapping.items():
        a,_,qid,_=k.split(".");q=next(q for q in scenes if q["id"]==qid)
        packet[bid]=dict(actor=base.NAME[a],public=q["public"],private=q[a],user=q["line"],say=probes[k]["say"])
    save("blind.json",packet)
    def grade(i):
        group={b:packet[b] for b in list(packet)[i:i+4]}
        m=[{"role":"system","content":design["coding"]+' 只输出JSON对象，按匿名编号各给 {"score":0/1/2或null,"error":true/false,"why":"简短理由"}。'},
           {"role":"user","content":json.dumps(group,ensure_ascii=False)}]
        raw=send(f"grade.{i//4+1:02}",m,reg["grader"])
        try:d=json.loads(raw)
        except ValueError:d={}
        def entry(b):
            x=d.get(b) if isinstance(d,dict) else None
            if (isinstance(x,dict) and type(x.get("score")) is int and x["score"] in range(3) and
                    type(x.get("error")) is bool and x.get("why") and group[b]["say"].strip()):return x
            return dict(score=None,error=None,why="无可用评分或正文")
        return {b:entry(b) for b in group}
    with ThreadPoolExecutor(max_workers=3) as ex:ratings={b:v for batch in ex.map(grade,range(0,96,4)) for b,v in batch.items()}
    save("model_coding.json",ratings)
    assert seen==set(records) and len(seen)==120
    print(Counter(k.split(".")[0] for k in seen),"unique calls checked.")

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"));ap.add_argument("--out",type=Path,default=HERE/"observed")
    args=ap.parse_args();experiment(args.mode,args.out)
