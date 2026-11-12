"""Separate, bounded interface calibration after the first run failed coverage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import gzip
import hashlib
import os
from pathlib import Path
import random
from threading import Lock

import trial as t

def jobs():
    blocks=[(a,q,n) for a in t.base.ROLES for q in t.SCENES if q["family"]!="transfer" for n in range(4)]
    rng=random.Random(45115);rng.shuffle(blocks);js=[]
    for a,q,n in blocks:
        arms=list(t.ARMS);rng.shuffle(arms);js.extend((a,b,q,n) for b in arms)
    return js

def userline(q,opts):
    return (q["line"]+"\n这是交到你本人手里的操作单，不是让我替你拿主意。两项都列在这里：\n"
            +"\n".join(o["ticket"]+"｜"+o["action"] for o in opts)
            +"\n由你现在选一项。交给我的对外回复只需独占一行写「落签：签名」，把签名换成你敲定的那枚。无需解释原因。")

def run(mode,out):
    reg=t.registration();reg["order"]=[t.key(j) for j in jobs()]
    for p in (Path(__file__),t.HERE/"handoff_design.json",t.HERE/"observed/choices.json"):
        reg["files"][p.relative_to(t.base.ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    def save(name,v):
        if mode=="live":t.write(out/name,v)
        if mode=="replay":assert t.read(out/name)==v,name
    with_lock=Lock();seen=set();cache={};token=None
    if mode!="dry":
        if mode=="live":out.mkdir(parents=True,exist_ok=True)
        fp=out/"registration.json";log=out/"calls.jsonl.gz"
        if fp.exists():assert t.read(fp)==reg
        else:
            assert mode=="live" and not any(out.iterdir());save("registration.json",reg)
        if log.exists():
            logs=t.base.lines(log);cache={x["id"]:x for x in logs};assert len(logs)==len(cache)
        if mode=="live":token=os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")
    old=t.read(t.HERE/"observed/choices.json")
    # Identical old foreground messages except for the last public user input.
    originals={r["id"]:r["messages"] for r in t.base.lines(t.HERE/"observed/calls.jsonl.gz")}
    _,traj=t.prior.states();hist={a:t.base.context(t.prior.attr.hist,a) for a in t.base.ROLES}
    def send(k,m):
        with with_lock:
            assert k not in seen;seen.add(k)
            if k in cache:
                assert cache[k]["params"]==reg["foreground"] and m==cache[k]["messages"];return cache[k]["raw"]
        if mode=="dry":return t.base.make_reply("need: 落签\nconfidence: 0.5\ndirect: 可以。\nbaseline: 可以。\ndisplace: none","落签：白签")
        if mode=="replay":raise ValueError("missing "+k)
        api=t.base.openai_compatible(**reg["foreground"],base_url="https://api.deepseek.com/v1",api_key=token,timeout=90)
        try:raw=api(m);r=dict(ok=True,response=api.observations.response,raw=raw)
        except Exception as e:raw="";r=dict(ok=False,error=type(e).__name__,raw=raw)
        r.update(id=k,messages=m,params=reg["foreground"])
        with with_lock:
            with gzip.open(log,"ab") as f:f.write(t.base.wire(r)+b"\n")
            cache[k]=r
        return raw
    def sample(j):
        a,b,q,n=j;k=t.key(j);opts,goal=t.menu(q,a,n);v=traj[a][-1]["growing"] if b=="G" else t.base.BASE[a]
        user=userline(q,opts);raws=[]
        def callback(m):
            original=originals[k]
            assert m[:-1]==original[:-1] and m[-1]["role"]==original[-1]["role"]=="user"
            assert original[-1]["content"]==q["line"] and m[-1]["content"]==user
            raw=send(k,m);raws.append(raw);return raw
        ans,m=t.base.turn(a,v,t.render(q,opts),q["private"],hist[a],user,callback)
        assert len(raws)==1 and v==old[k]["state"] and opts==old[k]["menu"]
        return k,dict(**ans,state=v,menu=opts,goal=goal,request=t.base.digest(m),normalized=t.base.digest(t.base.norm(m,a,v)),
                     event=t.transition(q,opts,goal,ans["say"],raws[0]))
    with ThreadPoolExecutor(max_workers=4) as ex:rows=dict(ex.map(sample,jobs()))
    for a,b,q,n in jobs():
        g,f,s=[rows[t.key((a,arm,q,n))] for arm in t.ARMS]
        assert g["normalized"]==f["normalized"]==s["normalized"] and g["request"]!=f["request"]==s["request"]
    assert len(rows)==len(seen)==120
    if mode!="dry":assert set(cache)==seen
    else:assert all(z["event"]["ticket"]=="白签" for z in rows.values())
    save("choices.json",rows);print("120 matched action forms checked.",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("live","replay","dry"));ap.add_argument("--out",type=Path,default=t.HERE/"handoff")
    args=ap.parse_args();run(args.mode,args.out)
