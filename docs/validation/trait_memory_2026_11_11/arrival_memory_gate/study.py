"""Three task states x three memory exposures. Only live mode contacts the provider."""
import argparse
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
from tempfile import TemporaryDirectory
from threading import Lock

HERE=Path(__file__).resolve().parent
from gate import admit,check_gate,Gate
from score import summarize
sys.path.insert(0,str(HERE.parent/"assignment_form"))
import forms as form
b,t,prev=form.b,form.t,form.prev
D=b.read(HERE/"design.json")
TRI={"future":True,"present":False,"unknown":None}
PREFIX="\n这次从交接记录取回的条目：\n"
def key(*parts):return ".".join(map(str,parts))
def jobs():
    rng=random.Random(D["seed"]);out=[]
    blocks=list(itertools.product(D["actors"],D["jobs"],TRI,range(4)));rng.shuffle(blocks)
    for a,j,q,n in blocks:
        cells=list(itertools.product(D["histories"],D["arms"]));rng.shuffle(cells)
        out.extend((a,j,q,h,m,n) for h,m in cells)
    return out

def view(a,h,q,m,mem):
    who=b.OTHER[a];calls=[]
    def read(k):
        calls.append(k)
        return mem[key(a,h,k)]
    if m=="Z":
        # An unexposed history label must never trigger retrieval, even in the diagnostic arms.
        outcome=admit(a,who,(a,who,"arrival"),TRI[q],read) if q!="future" else Gate("omit")
        assert not calls and outcome.visible==""
    elif m=="V" and q=="future":
        outcome=admit(a,who,(a,who,"arrival"),True,read)
        assert outcome.action=="use" and calls==["V"]
    else:
        outcome=Gate("diagnostic",read(m)["visible"]);assert calls==[m]
    return outcome.visible,calls

def scene(a,j,q):
    info=D["jobs"][j];post=info["post"];who=b.NAME[b.OTHER[a]]
    return dict(field=post,family="assignment",public=D["states"][q].replace("{台}",info["location"])+D["facts"].replace("{事}",info["work"]),
                target=[who,"由"+who+post],other=["常鸣","由常鸣"+post],line=prev.M["form"])

def enact(job,mem,traits,send):
    a,j,q,h,m,n=job;v,reads=view(a,h,q,m,mem)
    s=scene(a,j,q);menu,goal=t.menu(s,a,n);setting,user=form.wording(s,menu,"N")
    fragment=PREFIX+v if v else ""
    c=b.cfg(a,b.BASE[a],setting,"这张岗位单由你本人签。",D["common"]+fragment)
    old="对面是"+b.NAME[b.OTHER[a]]+"。";assert c.role.identity.count(old)==1
    c.role.identity=c.role.identity.replace(old,"操作单由登记员林穗递交，林穗不是候选人。")
    cap=[]
    def capture(msgs):
        norm=deepcopy(msgs)
        if fragment:
            encoded=json.dumps(fragment,ensure_ascii=False)[1:-1]
            assert norm[0]["content"].count(encoded)==1
            norm[0]["content"]=norm[0]["content"].replace(encoded,"",1)
        for msg in norm:
            assert not any(x in msg["content"] for x in ("05-04","07-22","关系簿：","取回的条目","两次准点、两次未到"))
        raw=send(msgs);cap.append((b.digest(msgs),b.digest(norm),raw));return raw
    with TemporaryDirectory() as tmp:
        st=b.Store(tmp)
        with b.Session(c,st,capture,options=b.SessionConfig(memory_mode="frozen",memory_scope=a,consolidate_on_exit=False)) as ss:
            before=st.traits();r=ss.say(user)
        assert before==st.traits()==c.role.traits_state()==traits[a]
    assert len(cap)==1
    dh,nh,raw=cap[0]
    return dict(request=dh,without_arrival=nh,read_calls=reads,chars=len(v),goal=goal,order=[z["value"] for z in menu],
                say=r.say,validity=r.validity,event=form.commit(s,menu,goal,"N",r.say,raw))

def run(mode,out):
    plan=jobs();assert len(plan)==len(set(plan))==288
    check_gate()
    oldreg=form.HERE/"observed/registration.json";old=b.read(oldreg)
    for fn,digest in old["files"].items():assert hashlib.sha256((b.ROOT/fn).read_bytes()).hexdigest()==digest,fn
    deps=[HERE/p for p in ("design.json","gate.py","study.py","score.py")]
    deps.extend([oldreg,form.OLD/"memory.json",form.OLD/"traits.json",form.HERE/"scope_check/observed/results.json"])
    reg=dict(inputs={p.relative_to(b.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},
             order=[key(*j) for j in plan],model=D["model"])
    assert reg["model"]==old["model"]
    def save(fn,obj):
        p=out/fn
        if mode=="replay" or mode=="live" and p.exists():assert b.read(p)==obj,fn
        elif mode=="live":b.write(p,obj)
    if mode=="live":out.mkdir(parents=True,exist_ok=True)
    save("registration.json",reg)
    mem=b.read(form.OLD/"memory.json");traits=b.read(form.OLD/"traits.json")
    routing={}
    for a,h in itertools.product(D["actors"],D["histories"]):
        who=b.OTHER[a]
        for q,known in TRI.items():
            reads=[]
            def read(m):
                reads.append(m);return mem[key(a,h,m)]
            v=admit(a,who,(a,who,"arrival"),known,read)
            assert (v.action,reads)=={"future":("use",["V"]),"present":("omit",[]),"unknown":("confirm_arrival",[])}[q]
            routing[key(a,h,q)]=dict(action=v.action,reads=list(reads),chars=len(v.visible))
            if known is None:
                for resolved,expect in ((True,"use"),(False,"omit")):
                    reads.clear()
                    z=admit(a,who,(a,who,"arrival"),resolved,read)
                    assert z.action==expect and reads==(["V"] if resolved else [])
    save("routing.json",routing)
    cache={};log=out/"calls.jsonl.gz";lock=Lock()
    if mode!="dry" and log.exists():
        logs=b.lines(log);cache={x["id"]:x for x in logs}
        assert len(logs)==len(cache) and set(cache)<=set(reg["order"])
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")) if mode=="live" and len(cache)<len(plan) else None
    def sample(j):
        k=key(*j)
        def send(msgs):
            if k in cache:
                assert cache[k]["messages"]==msgs and cache[k]["params"]==reg["model"]
                return cache[k]["raw"]
            if mode=="dry":return b.make_reply("need: 填单\nconfidence: 0.5\ndirect: 可以。\nbaseline: 可以。\ndisplace: none","经手人：常鸣")
            if mode!="live" or not secret:raise ValueError("Missing receipt or credential")
            api=b.openai_compatible(**reg["model"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90)
            try:
                raw=api(msgs);rec=dict(ok=True,raw=raw,response=api.observations.response)
            except Exception as exc:
                if not cache or any(w in str(exc) for w in ("401","402","403")):raise RuntimeError(type(exc).__name__) from None
                raw="";rec=dict(ok=False,raw=raw,response=api.observations.response,error=type(exc).__name__)
            rec.update(id=k,messages=msgs,params=reg["model"])
            with lock:
                with gzip.open(log,"ab") as fp:fp.write(b.wire(rec)+b"\n")
                cache[k]=rec
                if len(cache)%64==0:print(len(cache),"archived",flush=True)
            return raw
        return k,enact(j,mem,traits,send)
    rows=dict([sample(plan[0])])
    with ThreadPoolExecutor(max_workers=4) as ex:rows.update(ex.map(sample,plan[1:]))
    for a,j,q,h,m,n in plan:
        r=rows[key(a,j,q,h,m,n)];z=rows[key(a,j,q,"rise","Z",n)]
        assert r["without_arrival"]==z["request"] and (r["request"]==z["request"])==(m=="Z")
        assert r["request"]==rows[key(a,j,q,h,m,(n+2)%4)]["request"]
    if mode=="dry":assert all(r["event"]["chosen"]=="常鸣" for r in rows.values())
    else:assert set(cache)==set(rows)
    save("choices.json",rows);save("results.json",summarize(rows))
    print(mode,len(rows),"checked",flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"))
    ap.add_argument("--out",type=Path,default=HERE/"observed");args=ap.parse_args();run(args.mode,args.out)
