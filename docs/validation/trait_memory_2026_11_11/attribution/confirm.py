"""Prospective independent test of temporal separation in the attribution reader."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
from threading import Lock

import study as s
from persona.disclosure_claim import attribute_in_order
from persona.disclosure_evidence import Exchange
from persona.llm import openai_compatible

def metrics(keys,ref,votes):
    tp=sum(votes[k]==ref[k] and ref[k]!=0 for k in keys)
    return {"tp":tp,"booked":sum(votes[k]!=0 for k in keys),
            "reference":sum(ref[k]!=0 for k in keys),"n":len(keys)}

def run(out,live):
    original=s.HERE/"observed";out.mkdir(parents=True,exist_ok=True)
    reg=s.register()
    for p in (Path(__file__),s.HERE/"confirmation.json",s.prev.ROOT/"persona/disclosure_claim.py",
              original/"attribution.json",original/"probes.json",original/"reading.json"):
        reg[p.relative_to(s.prev.ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    def save(f,v):
        if live:s.write(out/f,v)
        else:assert s.read(out/f)==v,f
    if (out/"registration.json").exists():assert s.read(out/"registration.json")==reg
    else:
        assert live and not any(out.iterdir());save("registration.json",reg)
    p=out/"calls.jsonl.gz"; recs=s.prev.lines(p) if p.exists() else [];cached={r["id"]:r for r in recs}
    assert len(cached)==len(recs)
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("DeepSeek credential: ")) if live else None
    seen=set();lock=Lock()
    def send(k,m,par):
        with lock:
            assert k not in seen;seen.add(k)
            if k in cached:
                r=cached[k];assert r["messages"]==m and r["params"]==par;return r["raw"]
        if not live:raise ValueError("missing cached call "+k)
        api=openai_compatible(base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90,**par)
        try:raw=api(m)
        except Exception as ex:raise RuntimeError(f"{k}: {type(ex).__name__}") from None
        r=dict(id=k,messages=m,params=par,raw=raw,response=api.observations.response)
        with lock:
            with gzip.open(p,"ab") as f:f.write(s.prev.wire(r)+b"\n")
            cached[k]=r
        return raw
    pairs,ref=s.exchanges();pairs={k:p for k,p in pairs.items() if k.startswith("H")}
    for i,(_,a,x,y,v) in enumerate(s.read(s.HERE/"confirmation.json")["controls"],1):
        k=f"N{i:02}";pairs[k]=Exchange(a,s.prev.OTHER[a],x,y,True);ref[k]=v
    def gate(k):return k,attribute_in_order(pairs[k],lambda stage,m:send(stage+"."+k,m,s.judge))
    ids=list(pairs);random.Random(2711).shuffle(ids)
    with ThreadPoolExecutor(max_workers=4) as pool:rows=dict(pool.map(gate,ids))
    save("attribution.json",rows);traj=s.evolve(rows);save("trajectories.json",traj)
    hs=[k for k in pairs if k.startswith("H") and ref[k] is not None and pairs[k].protocol_ok]
    ov={k:s.hist[int(k[1:3])-1]["evidence"][k.split(".")[1]]["verified"] for k in hs}
    old=metrics(hs,ref,ov);new=metrics(hs,ref,{k:rows[k]["vote"] for k in hs})
    score=sum(rows[k]["label"]==ref[k] for k in pairs if k[0]=="N")
    cross=sum(bool(rows[f"N{i:02}"]["vote"]) for i in (3,7,11,15))
    ok=(new["booked"]>0 and new["tp"]*old["booked"]>old["tp"]*new["booked"] and
        new["tp"]>old["tp"] and score>=14 and not cross)
    before=s.read(original/"trajectories.json")
    roles=[a for a in s.prev.ROLES if ok and before[a][-1]["growing"]!=traj[a][-1]["growing"]]
    result=dict(history=new,original=old,correct_controls=score,wrong_owner=cross,passes=ok,retest=roles)
    save("decision.json",result);print(result,flush=True)
    if roles:
        original_answers=s.read(original/"probes.json")
        jobs=[(a,q,r) for a in roles for q in s.qs for r in (1,2)];random.Random(7121).shuffle(jobs)
        def test(j):
            a,q,r=j;k=f"{a}.growth.{q['id']}.{r}";v=traj[a][-1]["growing"]
            entry,m=s.prev.turn(a,v,q["public"],q[a],s.prev.context(s.hist,a),q["line"],
                                lambda m:send("probe."+k,m,s.prev.D["foreground"]))
            norm=s.prev.digest(s.prev.norm(m,a,v))
            assert norm==original_answers[k.replace(".growth.",".frozen.")]["normalized"]
            return k,dict(**entry,value=v,request=s.prev.digest(m),normalized=norm)
        with ThreadPoolExecutor(max_workers=4) as pool:answers=dict(pool.map(test,jobs))
        save("probes.json",answers)
        if not (out/"key.json").exists():
            assert live
            ids=sorted(answers);random.SystemRandom().shuffle(ids)
            save("key.json",{f"B{i:02}":v for i,v in enumerate(ids,1)})
        key=s.read(out/"key.json");assert set(key.values())==set(answers)
        # Mix new replies with previously scored frozen answers, so readers cannot identify the arm.
        if not (out/"blind_key.json").exists():
            assert live
            ids=list(key.values())+[k for k in original_answers if k.split(".")[0] in roles and ".growth." not in k]
            random.SystemRandom().shuffle(ids);save("blind_key.json",{f"R{i:02}":v for i,v in enumerate(ids,1)})
        mapping=s.read(out/"blind_key.json")
        packet={}
        for b,k in mapping.items():
            a,arm,qid,_=k.split(".");q=next(q for q in s.qs if q["id"]==qid)
            z=(answers if arm=="growth" else original_answers)[k]
            packet[b]=dict(actor=s.prev.NAME[a],public=q["public"],private=q[a],user=q["line"],say=z["say"])
        save("blind.json",packet)
        def grade(i):
            batch={k:packet[k] for k in list(packet)[i:i+4]}
            m=[{"role":"system","content":s.grade_sys},{"role":"user","content":json.dumps(batch,ensure_ascii=False)}]
            try:d=s.prev.obj(send(f"grade.{i//4:02}",m,dict(s.judge,max_tokens=1250)))
            except ValueError:d={}
            return {k:(d[k] if isinstance(d,dict) and isinstance(d.get(k),dict) and type(d[k].get("score")) is int
                        and d[k]["score"] in (0,1,2) and packet[k]["say"].strip() and isinstance(d[k].get("why"),str)
                       else {"score":None,"why":"missing"}) for k in batch}
        with ThreadPoolExecutor(max_workers=3) as pool:
            save("model_coding.json",{k:v for d in pool.map(grade,range(0,len(packet),4)) for k,v in d.items()})
    assert seen==set(cached)
    print(len(seen),"verified calls",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["live","replay"]);p.add_argument("--out",type=Path,default=s.HERE/"confirmed")
    a=p.parse_args();run(a.out,a.mode=="live")
