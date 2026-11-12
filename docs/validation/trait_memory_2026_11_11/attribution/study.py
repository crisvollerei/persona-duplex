"""One archived longitudinal story, two attribution gates, one frozen resampling control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from threading import Lock

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"longitudinal"
sys.path.insert(0,str(OLD))
import run as prev
from persona.disclosure_evidence import Exchange,attribute
from persona.llm import openai_compatible
from persona.consolidate import Consolidator
from persona.store import replay_traits

read,write=prev.read,prev.write
hist=read(OLD/"observed/trajectory.json")
qs=read(HERE/"probes.json")
arms=("growth","frozen","sham")
judge=dict(model="deepseek-chat",temperature=0.0,max_tokens=900)
grade_sys=prev.D["rubric"]+' 你只看到匿名正文。只输出JSON，按每个编号返回 {"score":整数0至2，无正文时为null,"why":"理由"}。'

def exchanges():
    pairs={};ref={}
    refs=read(HERE/"reference.json")["rows"]
    for e,labels in zip(hist,refs,strict=True):
        for i in (0,1):
            t,u=e["turns"][i:i+2];a=t["actor"];k=f'H{e["episode"]:02}.{a}'
            pairs[k]=Exchange(a,u["actor"],t["say"],u["say"],all(z["validity"]["valid"] for z in (t,u)))
            ref[k]=labels[prev.ROLES.index(a)][0]
    for i,(_,a,x,y,v) in enumerate(read(HERE/"controls.json"),1):
        k=f"C{i:02}";pairs[k]=Exchange(a,prev.OTHER[a],x,y,True);ref[k]=v
    assert len(pairs)==64
    return pairs,ref

def evolve(outcomes):
    pairs,_=exchanges()
    with TemporaryDirectory() as tmp:
        stores=prev.start_states(Path(tmp)); result={}
        for a in prev.ROLES:
            rows=[];result[a]=rows
            for e in hist:
                k=f'H{e["episode"]:02}.{a}';v=outcomes[k];p=pairs[k]
                row={"id":k}
                for arm in ("growing","frozen"):
                    st=stores[a,arm]
                    st.append("turn",{"turn_id":k,"say":p.first,"user":p.second})
                    if v["vote"]:
                        st.add_judgment(v["reason"],prev.P["weight"],[k],affects={prev.K:v["vote"]},
                            source="experiment_attributed_exchange",memory_scope=a,
                            evidence_quotes=[{"turn_id":k,"quote":q,"evidence_scope":"specified"} for q in v["quotes"]])
                    ch=Consolidator(st).push_traits(min_consensus=prev.D["threshold"]) if arm=="growing" else []
                    value=st.traits()[prev.K]["value"]
                    assert value==replay_traits(st.read_ledger())[prev.K]["value"]
                    row[arm]=value
                    if arm=="growing":row["push"]=bool(ch)
                rows.append(row)
        return result

def register():
    files=[HERE/k for k in ("study.py","design.json","reference.json","controls.json","probes.json")]
    files += [OLD/"run.py",OLD/"observed/trajectory.json",OLD/"observed/registration.json",
              prev.ROOT/"persona/disclosure_evidence.py"]
    return {p.relative_to(prev.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def run(out,live):
    out.mkdir(parents=True,exist_ok=True)
    reg=register()
    if not (out/"registration.json").exists():
        assert live and not any(out.iterdir())
        write(out/"registration.json",reg)
    assert read(out/"registration.json")==reg
    for f,h in read(OLD/"observed/registration.json")["files"].items():
        assert hashlib.sha256((prev.ROOT/f).read_bytes()).hexdigest()==h
    def save(name,data):
        f=out/name
        if live:write(f,data)
        else:assert read(f)==data,name
    log=out/"calls.jsonl.gz"
    entries=prev.lines(log) if log.exists() else []
    cache={r["id"]:r for r in entries};assert len(cache)==len(entries)
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("DeepSeek credential: ")) if live else None
    lock=Lock();used=set()
    def send(k,msgs,params):
        with lock:
            assert k not in used;used.add(k)
            if k in cache:
                assert cache[k]["messages"]==msgs and cache[k]["params"]==params,k
                return cache[k]["raw"]
        if not live:raise RuntimeError("missing call "+k)
        f=openai_compatible(base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90,**params)
        try:raw=f(msgs)
        except Exception as ex:raise RuntimeError(f"{k}: {type(ex).__name__}") from None
        record=dict(id=k,messages=msgs,params=params,raw=raw,response=f.observations.response)
        with lock:
            with gzip.open(log,"ab") as fp:fp.write(prev.wire(record)+b"\n")
            cache[k]=record
        return raw
    pairs,_=exchanges()
    def gate(k):return k,attribute(pairs[k],send("gate."+k,pairs[k].messages(),judge))
    order=list(pairs);random.Random(71031).shuffle(order)
    with ThreadPoolExecutor(max_workers=4) as pool:outcomes=dict(pool.map(gate,order))
    save("attribution.json",outcomes)
    trajectories=evolve(outcomes);save("trajectories.json",trajectories)
    print("Attributions complete. End states:", {a:vs[-1]["growing"] for a,vs in trajectories.items()},flush=True)
    jobs=[(a,b,q,r) for a in prev.ROLES for b in arms for q in qs for r in (1,2)]
    random.Random(42306).shuffle(jobs)
    def sample(job):
        a,b,q,r=job;k=f"{a}.{b}.{q['id']}.{r}";v=trajectories[a][-1]["growing"] if b=="growth" else prev.BASE[a]
        result,m=prev.turn(a,v,q["public"],q[a],prev.context(hist,a),q["line"],
                     lambda m:send("probe."+k,m,prev.D["foreground"]))
        return k,dict(**result,value=v,request=prev.digest(m),normalized=prev.digest(prev.norm(m,a,v)))
    with ThreadPoolExecutor(max_workers=4) as pool:answers=dict(pool.map(sample,jobs))
    for a,_,q,r in jobs:
        trio=[answers[f"{a}.{b}.{q['id']}.{r}"] for b in arms]
        assert len({x["normalized"] for x in trio})==1 and trio[1]["request"]==trio[2]["request"]
        if trio[0]["value"]==prev.BASE[a]:assert trio[0]["request"]==trio[1]["request"]
    save("probes.json",answers)
    mapfile=out/"key.json"
    if not mapfile.exists():
        assert live
        ids=sorted(answers);random.SystemRandom().shuffle(ids)
        write(mapfile,{f"A{i:02}":v for i,v in enumerate(ids,1)})
    mapping=read(mapfile);assert len(mapping)==48 and set(mapping.values())==set(answers)
    packet={}
    for b,k in mapping.items():
        a,_,qid,_=k.split(".");q=next(q for q in qs if q["id"]==qid)
        packet[b]=dict(actor=prev.NAME[a],public=q["public"],private=q[a],user=q["line"],say=answers[k]["say"])
    save("blind.json",packet)
    def grade(i):
        batch={k:packet[k] for k in list(packet)[i:i+4]}
        m=[{"role":"system","content":grade_sys},{"role":"user","content":json.dumps(batch,ensure_ascii=False)}]
        raw=send(f"grade.{i//4:02}",m,dict(judge,max_tokens=1250))
        try:d=prev.obj(raw)
        except ValueError:d={}
        return {k:(d[k] if isinstance(d,dict) and isinstance(d.get(k),dict) and
                   type(d[k].get("score")) is int and d[k]["score"] in (0,1,2) and
                   packet[k]["say"].strip() and isinstance(d[k].get("why"),str) else {"score":None,"why":"missing"}) for k in batch}
    with ThreadPoolExecutor(max_workers=3) as pool:ratings={k:v for chunk in pool.map(grade,range(0,48,4)) for k,v in chunk.items()}
    save("model_coding.json",ratings)
    assert len(used)==len(cache)==124
    print(f"{len(used)} unique calls checked.",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("live","replay"));ap.add_argument("--out",type=Path,default=HERE/"observed")
    args=ap.parse_args();run(args.out,args.mode=="live")
