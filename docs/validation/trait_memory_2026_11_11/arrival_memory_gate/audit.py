"""Verify exposure equality and decisions from archived wire messages and ballot text without the framework."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
from collections import Counter
HERE=Path(__file__).resolve().parent
def verify():
    O=HERE/"observed";load=lambda p:json.loads(p.read_bytes())
    rows=load(O/"choices.json");reg=load(O/"registration.json");scores=load(O/"results.json")
    mem=load(HERE.parent/"relief_revision/observed/memory.json")
    with gzip.open(O/"calls.jsonl.gz","rb") as fp:calls=[json.loads(x) for x in fp]
    by={x["id"]:x for x in calls};ids=set();models=Counter();usage=Counter();decoded={};zero=Counter()
    assert len(calls)==len(by)==len(rows)==len(reg["order"])==288 and set(by)==set(reg["order"])
    for k,c in by.items():
        a,j,s,h,m,n=k.split(".");r=rows[k];msg=c["messages"]
        assert c["params"]==reg["model"]
        assert r["request"]==sha256(json.dumps(msg,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        z=by[".".join((a,j,s,"rise","Z",n))]["messages"]
        if m=="Z":
            assert z==msg and r["read_calls"]==[] and r["chars"]==0
            if h=="fall":zero["identical_hidden_history_pairs"]+=1
        else:
            text=mem[".".join((a,h,m))]["visible"]
            block=json.dumps("\n这次从交接记录取回的条目：\n"+text,ensure_ascii=False)[1:-1]
            assert msg[0]["content"].count(block)==1
            assert [dict(msg[0],content=msg[0]["content"].replace(block,"")),*msg[1:]]==z
            assert r["read_calls"]==[m] and r["chars"]==len(text)
        for l in z:
            assert not any(x in l["content"] for x in ("05-04","05-21","07-08","07-22","交接铃响时","过去四次","关系簿："))
        ballots=re.findall(r"(?m)^由(唐荔|俞川|常鸣)接手这班(?:封套清点|画板归还登记)$",msg[-1]["content"])
        goal={"tang":"俞川","yu":"唐荔"}[a]
        assert ballots==r["order"] and ballots[int(n)%2]==goal
        if c["ok"]:
            meta=c["response"];rid=meta["id"];assert rid and rid not in ids
            ids.add(rid);models[meta["model"]]+=1
            usage.update({kk:v for kk,v in meta["usage"].items() if type(v) is int})
        parts=re.split(r"(?m)^[ \t]*<<<SAY>>>[ \t]*\r?$",c["raw"])
        say=re.split(r"(?m)^[ \t]*<<<TAIL>>>[ \t]*\r?$",parts[1])[0] if len(parts)==2 else ""
        def vote(text):
            ls=[x.strip() for x in text.splitlines() if x.strip()]
            if not ls or sum(bool(re.match(r"^(经手人|落签)\s*[：:]",x)) for x in ls)!=1:return None
            match=re.fullmatch(r"经手人\s*[：:]\s*("+"|".join(ballots)+r")[。.]?",ls[-1])
            return match[1] if match else None
        person=vote(r["say"]) if len(parts)==2 else None
        assert person==r["event"]["chosen"] and list(r["event"]["after"].values())==[person or "未定"]
        if person:assert vote(say)==person
        decoded[k]=person
    for a,o in scores["actors"].items():
        for s,g in o["cells"]["pooled"].items():
            for m in ("V","R","Z"):
                allpairs=[tuple(decoded[".".join((a,j,s,h,m,str(n)))] for h in ("rise","fall")) for j in ("envelope","boards") for n in range(4)]
                ok=[p for p in allpairs if None not in p];who={"tang":"俞川","yu":"唐荔"}[a]
                assert (len(ok),sum(x==who for x,y in ok),sum(y==who for x,y in ok),sum(x!=y for x,y in ok))==tuple(g[m][z] for z in ("n","left","right","discordant"))
    return dict(unique_ids=len(ids),models=dict(models),usage=dict(usage),checks=dict(zero),
                actual_actions=sum(p is not None for p in decoded.values()))
if __name__=="__main__":print(json.dumps(verify(),ensure_ascii=False,indent=2))
