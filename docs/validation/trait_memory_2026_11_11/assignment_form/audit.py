"""Read candidate order and outcome again from the original service receipts."""
from collections import Counter
import gzip
import json
from hashlib import sha256
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent
O=HERE/"observed"
def load(p):return json.loads(p.read_bytes())
def records(p):
    with gzip.open(p,"rb") as f:return [json.loads(l) for l in f]

def audit():
    reg,rows,result=[load(O/f) for f in ("registration.json","choices.json","results.json")]
    calls=records(O/"calls.jsonl.gz")
    assert len(calls)==len(reg["order"])==len(set(reg["order"]))==336
    cur={r["id"]:r for r in calls};assert set(cur)==set(rows)==set(reg["order"])
    old=HERE.parent/"relief_revision"/"observed"
    baseline={r["id"]:r for r in records(old/"calls.jsonl.gz")}
    ids=set(); models=Counter();outcomes={}; usage=Counter()
    for c in calls:
        k=c["id"];a,h,m,q,f,n=k.split(".");n=int(n);r=rows[k]
        assert c["params"]==reg["model"]
        if c["ok"]:
            meta=c["response"];rid=meta["id"];assert rid and rid not in ids
            ids.add(rid);models[meta["model"]]+=1
            usage.update({key:v for key,v in meta["usage"].items() if type(v) is int})
        messages=c["messages"]
        digest=sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        assert digest==r["request"]
        if f=="T":assert baseline[".".join((a,h,m,q,str(n)))]["messages"]==messages
        if f=="B":
            # The only B/N change occurs in the two menu copies.
            paired=cur[".".join((a,h,m,q,"N",str(n)))]["messages"]
            assert sum(msg["content"].count("签｜") for msg in messages)==4
            assert [dict(msg,content=re.sub(r"[青白]签｜","",msg["content"])) for msg in messages]==paired
        if f=="N":assert cur[".".join((a,h,m,q,f,str((n+2)%4)))]["messages"]==messages
        matches=re.findall(r"(?m)^(?:(青签|白签)｜)?由(俞川|唐荔|池岚|常鸣)",messages[-1]["content"])
        names=[w for _,w in matches];assert len(names)==2 and len(set(names))==2
        named=("池岚" if q=="other_person" else {"tang":"俞川","yu":"唐荔"}[a])
        assert names[n%2]==named
        delivered=r["say"].strip(); rawparts=re.split(r"(?m)^[ \t]*<<<SAY>>>[ \t]*\r?$",c["raw"])
        lines=[x.strip() for x in delivered.splitlines() if x.strip()]
        value=None
        if len(rawparts)==2 and lines and sum(bool(re.match(r"^(经手人|落签)\s*[：:]",x)) for x in lines)==1:
            expr=r"落签\s*[：:]\s*(青签|白签)[。.]?" if f=="T" else r"经手人\s*[：:]\s*("+"|".join(names)+r")[。.]?"
            match=re.fullmatch(expr,lines[-1])
            if match:
                value=dict(matches)[match[1]] if f=="T" else match[1]
                public=re.split(r"(?m)^[ \t]*<<<TAIL>>>[ \t]*\r?$",rawparts[1])[0]
                assert lines[-1] in public
        assert value==r["event"]["chosen"] and list(r["event"]["after"].values())==[value or "未定"]
        outcomes[a,h,m,q,f,n]=value
    # Use actual identities for the flip count (no menu-goal or target flag).
    for a in ("tang","yu"):
        for f in ("T","B","N"):
            pairs=[(outcomes[a,h,"R","relief",f,n],outcomes[a,h,"R","relief",f,n+1])
                for h in ("rise","fall") for n in (0,2)]
            ok=[(x,y) for x,y in pairs if None not in (x,y)]
            assert [len(ok),sum(x!=y for x,y in ok)]==[result["actors"][a]["native"][f][k] for k in ("n","flips")]
    return dict(calls=len(calls),unique_ids=len(ids),returned_models=dict(models),tokens=dict(usage),
                decoded_actions=sum(v is not None for v in outcomes.values()))
if __name__=="__main__":print(json.dumps(audit(),ensure_ascii=False,indent=2))
