"""A ballot commits one local story action. Dry/replay do not contact the provider."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from threading import Lock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"reference_growth"))
import experiment as prior
base=prior.base
read,write=base.read,base.write
SCENES=read(HERE/"scenes.json")
ARMS=("G","F","S")
COLORS=("青签","白签")
MARK=re.compile(r"^\s*<<<SAY>>>\s*$")
DIRECTIVE=re.compile(r"落签\s*[：:]\s*(青签|白签)[。.]?")

def menu(q,a,layout):
    spec=q.get(a,q)
    opts={b:spec[b] for b in ("target","other")}
    target_first=layout%2==0;target_blue=layout//2==0
    codes={b:COLORS[0 if (b=="target")==target_blue else 1] for b in opts}
    order=("target","other") if target_first else ("other","target")
    entries=[{"ticket":codes[b],"value":opts[b][0],"action":opts[b][1]} for b in order]
    return entries,opts["target"][0]

def transition(q,entries,goal,say,raw):
    before={q["field"]:"未定"}
    row=dict(before=before,after=dict(before),ticket=None,chosen=None,target=None,status="no_ballot")
    if sum(bool(MARK.fullmatch(x)) for x in raw.splitlines())!=1:
        row["status"]="unmarked_or_ambiguous_say";return row
    ls=[x.strip() for x in say.splitlines() if x.strip()]
    if not ls:return row
    if sum(bool(re.match(r"^落签\s*[：:]",x)) for x in ls)!=1:return row
    match=DIRECTIVE.fullmatch(ls[-1])
    if not match:return row
    ticket=match[1];chosen=next(c["value"] for c in entries if c["ticket"]==ticket)
    hit=chosen==goal
    blocked=q["family"]=="control" and not hit
    row.update(ticket=ticket,chosen=chosen,target=hit,
               status="blocked_by_width" if blocked else "executed",
               after=dict(before) if blocked else {q["field"]:chosen})
    return row

def jobs():
    blocks=[(a,q,n) for a in base.ROLES for q in SCENES for n in range(4)]
    rng=random.Random(151142);rng.shuffle(blocks);out=[]
    for a,q,n in blocks:
        order=list(ARMS);rng.shuffle(order);out.extend((a,b,q,n) for b in order)
    return out

def key(j):
    a,b,q,n=j
    return f'{a}.{b}.{q["id"]}.{n}'

def render(q,entries):
    return (q["public"]+"\n面前的两枚签都由你本人使用，一次只能落一枚。签面说明如下：\n"+
         "\n".join(x["ticket"]+"｜"+x["action"] for x in entries)+
         "\n这是场内实际操作，不是替搭档出主意。装置只读取你对外回应末尾的独立一行：落签：青签 或 落签：白签。"
         "不用解释理由；没有落签就保持未定。")

def registration():
    srcs=[HERE/x for x in ("trial.py","design.json","scenes.json")]
    files=prior.fingerprints()
    files.update({p.relative_to(base.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs})
    return dict(files=files,order=[key(j) for j in jobs()],foreground=base.D["foreground"])

def run(mode,out):
    marks,traj=prior.states()
    for a in base.ROLES:
        for q in SCENES:
            if q["family"]=="transfer":
                z=q[a]["source"]
                assert z.endswith("."+a) and marks[z]["vote"]==1
    assert len({q["field"] for q in SCENES})==7
    reg=registration()
    def save(name,obj):
        if mode=="live":write(out/name,obj)
        elif mode=="replay":assert read(out/name)==obj,name
    lock=Lock();seen=set();cache={};secret=None
    if mode!="dry":
        if mode=="live":out.mkdir(parents=True,exist_ok=True)
        rp=out/"registration.json";log=out/"calls.jsonl.gz"
        if rp.exists():assert read(rp)==reg
        else:
            assert mode=="live" and not any(out.iterdir());save("registration.json",reg)
        if log.exists():
            rs=base.lines(log);cache={z["id"]:z for z in rs};assert len(rs)==len(cache)
        if mode=="live":secret=os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")
    save("states.json",dict(accepted={a:[k for k,v in marks.items() if k.endswith("."+a) and v["vote"]] for a in base.ROLES},trajectories=traj))
    hists={a:base.context(prior.attr.hist,a) for a in base.ROLES}
    def send(k,m):
        with lock:
            assert k not in seen;seen.add(k)
            if k in cache:
                r=cache[k];assert r["messages"]==m and r["params"]==reg["foreground"],k
                return r["raw"]
        if mode=="dry":return base.make_reply("need: 作选择\nconfidence: 0.5\ndirect: 可以。\nbaseline: 可以。\ndisplace: none","落签：青签")
        if mode=="replay":raise ValueError("missing "+k)
        f=base.openai_compatible(**reg["foreground"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90)
        try:
            raw=f(m);rec=dict(ok=True,raw=raw,response=f.observations.response)
        except Exception as exc:
            raw="";rec=dict(ok=False,raw=raw,error=type(exc).__name__)
        rec.update(id=k,messages=m,params=reg["foreground"])
        with lock:
            with gzip.open(log,"ab") as fp:fp.write(base.wire(rec)+b"\n")
            cache[k]=rec
        return raw
    def sample(j):
        a,b,q,n=j;k=key(j);entries,goal=menu(q,a,n);v=traj[a][-1]["growing"] if b=="G" else base.BASE[a]
        caught=[]
        def callback(m):
            raw=send(k,m);caught.append(raw);return raw
        ans,m=base.turn(a,v,render(q,entries),q["private"],hists[a],q["line"],callback)
        assert len(caught)==1
        ans.update(state=v,menu=entries,goal=goal,request=base.digest(m),normalized=base.digest(base.norm(m,a,v)),
                   event=transition(q,entries,goal,ans["say"],caught[0]))
        return k,ans
    with ThreadPoolExecutor(max_workers=4) as pool: rows=dict(pool.map(sample,jobs()))
    for a in base.ROLES:
        for q in SCENES:
            cells=[]
            for n in range(4):
                g,f,s=[rows[key((a,b,q,n))] for b in ARMS]
                assert g["normalized"]==f["normalized"]==s["normalized"] and g["request"]!=f["request"]==s["request"]
                which=next(i for i,x in enumerate(f["menu"]) if x["value"]==f["goal"])
                cells.append((which,f["menu"][which]["ticket"]))
            assert len(set(cells))==4
    assert len(rows)==len(seen)==168
    if mode!="dry":assert seen==set(cache)
    else:assert all(z["event"]["ticket"]=="青签" for z in rows.values())
    save("choices.json",rows)
    print("168 condition/layout requests and story transitions checked.",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"))
    ap.add_argument("--out",type=Path,default=HERE/"observed");args=ap.parse_args();run(args.mode,args.out)
