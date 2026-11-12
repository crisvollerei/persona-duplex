"""Independent, narrowly scoped confirmation after the form comparison. No production changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import gzip
import hashlib
import itertools
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from threading import Lock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import forms as parent
b,t,prev=parent.b,parent.t,parent.prev
D=b.read(HERE/"design.json")
O=HERE/"observed"
def key(*xs):return ".".join(map(str,xs))

def route(record,actor,subject,requires_arrival):
    note=next((v for v in record["versions"] if v["id"]==record["active"]),{})
    if requires_arrival is True and note.get("memory_scope")==key(actor,subject,"arrival"):return "V"
    return "R"

def measurement(rows):
    out={}
    for a in D["role"]:
        cells={}
        for q,m in (("future","V"),("present","V"),("present","R")):
            vals=[]
            for n in range(4):
                u,v=[rows[key(a,h,q,m,n)]["event"]["target"] for h in D["history"]]
                vals.append((u,v))
            ok=[(u,v) for u,v in vals if None not in (u,v)]
            l=sum(u for u,v in ok);r=sum(v for u,v in ok);n=len(ok)
            lo=sum((u or 0)-(1 if v is None else v) for u,v in vals)/4
            hi=sum((1 if u is None else u)-(v or 0) for u,v in vals)/4
            cells[key(q,m)]=dict(left=l,right=r,n=n,d=(l-r)/n if n else None,bounds=[lo,hi])
        x,y,z=[cells[k] for k in ("future.V","present.V","present.R")]
        # Compare the two present policies on exactly the same layouts as well.
        common=[]
        for n in range(4):
            vals=[rows[key(a,h,"present",m,n)]["event"]["target"] for m in ("V","R") for h in D["history"]]
            if None not in vals:common.append((vals[0]-vals[1],vals[2]-vals[3]))
        nn=len(common);cv=sum(i for i,j in common)/nn if nn else None;cr=sum(j for i,j in common)/nn if nn else None
        passed=min(x["n"],y["n"],z["n"],nn)>=3 and x["d"]>=.5 and abs(cr)<=.25 and abs(cv)-abs(cr)>=.25 and abs(z["d"])<=.25 and abs(y["d"])-abs(z["d"])>=.25
        out[a]=dict(cells=cells,common=dict(n=nn,V=cv,R=cr),passed=passed)
    return dict(count=len(rows),signed=sum(r["event"]["chosen"] is not None for r in rows.values()),actors=out,
                supported=all(r["passed"] for r in out.values()))

def study(mode):
    plan=[(a,h,q,m,n) for a,h,n in itertools.product(D["role"],D["history"],range(4))
          for q,m in (("future","V"),("present","V"),("present","R"))]
    random.Random(D["seed"]).shuffle(plan)
    assert len(plan)==48
    p_reg=HERE.parent/"observed/registration.json"
    deps=[HERE/"scope.py",HERE/"design.json",p_reg,HERE.parent/"observed/results.json"]
    reg=dict(files={p.relative_to(b.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},order=[key(*j) for j in plan],model=parent.D["model"])
    for f,h in b.read(p_reg)["files"].items():assert hashlib.sha256((b.ROOT/f).read_bytes()).hexdigest()==h
    mem=b.read(parent.OLD/"memory.json");frozen=b.read(parent.OLD/"traits.json")
    def save(fn,obj):
        fp=O/fn
        if mode=="replay" or mode=="live" and fp.exists():assert b.read(fp)==obj,fn
        elif mode=="live":b.write(fp,obj)
    if mode=="live":O.mkdir(exist_ok=True)
    save("registration.json",reg)
    cache={};lock=Lock();log=O/"calls.jsonl.gz"
    if mode!="dry" and log.exists():
        logs=b.lines(log);cache={c["id"]:c for c in logs};assert len(cache)==len(logs)
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")) if mode=="live" and len(cache)<48 else None
    def job(j):
        a,h,q,m,n=j;k=key(*j);person=b.OTHER[a];name=b.NAME[person]
        record=mem[key(a,h,"V")];chosen=route(record,a,person,D["probes"][q]["requires_arrival"])
        assert chosen==("V" if q=="future" else "R")
        for fake in ((a,"other",True),(person,person,True),(a,person,None)):assert route(record,*fake)=="R"
        policies=(["always","scoped"] if m==chosen=="V" else ["always"] if m=="V" else ["scoped"])
        view=mem[key(a,h,m)];job=D["post"]
        scene=dict(field=job,family="assignment",public=D["probes"][q]["detail"]+D["facts"],
                   target=[name,"由"+name+job],other=["常鸣","由常鸣"+job],line=prev.M["form"])
        menu,goal=t.menu(scene,a,n);setting,user=parent.wording(scene,menu,"N")
        ctx=prev.M["common"]+"\n这次从交接记录取回的条目：\n"+view["visible"]
        c=b.cfg(a,b.BASE[a],setting,"这张岗位单由你本人签。",ctx)
        old="对面是"+name+"。";assert c.role.identity.count(old)==1
        c.role.identity=c.role.identity.replace(old,"操作单由登记员林穗递交，林穗不是候选人。")
        caught=[]
        def send(msgs):
            if k in cache:
                assert cache[k]["messages"]==msgs and cache[k]["params"]==reg["model"]
                raw=cache[k]["raw"]
            elif mode=="dry":
                raw=b.make_reply("need: 填单\nconfidence: 0.5\ndirect: 可以。\nbaseline: 可以。\ndisplace: none","经手人：常鸣")
            else:
                assert mode=="live" and secret
                api=b.openai_compatible(**reg["model"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90)
                try:
                    raw=api(msgs);cl=dict(ok=True,raw=raw,response=api.observations.response)
                except Exception as exc:
                    if not cache or any(w in str(exc) for w in ("401","402","403")):raise RuntimeError(type(exc).__name__) from None
                    raw="";cl=dict(ok=False,raw=raw,error=type(exc).__name__,response=api.observations.response)
                cl.update(id=k,messages=msgs,params=reg["model"])
                with lock:
                    with gzip.open(log,"ab") as fp:fp.write(b.wire(cl)+b"\n")
                    cache[k]=cl
            caught.append((b.digest(msgs),raw));return raw
        with TemporaryDirectory() as tmp:
            st=b.Store(tmp)
            with b.Session(c,st,send,options=b.SessionConfig(memory_mode="frozen",consolidate_on_exit=False,memory_scope=a)) as ss:
                before=st.traits();ans=ss.say(user)
            assert before==st.traits()==c.role.traits_state()==frozen[a]
        assert len(caught)==1
        return k,dict(policies=policies,request=caught[0][0],memory_chars=view["chars"],goal=goal,order=[z["value"] for z in menu],
                      say=ans.say,validity=ans.validity,event=parent.commit(scene,menu,goal,"N",ans.say,caught[0][1]))
    rows=dict([job(plan[0])])
    with ThreadPoolExecutor(max_workers=4) as ex:rows.update(ex.map(job,plan[1:]))
    for a,h,q,m,n in plan:assert rows[key(a,h,q,m,n)]["request"]==rows[key(a,h,q,m,(n+2)%4)]["request"]
    if mode!="dry":assert set(cache)==set(rows)
    save("choices.json",rows);save("results.json",measurement(rows))
    print(mode,48,"checked")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"));study(ap.parse_args().mode)
