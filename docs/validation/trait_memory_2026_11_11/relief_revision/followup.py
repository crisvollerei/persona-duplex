"""Evidence-first recovery and equal-count order contrasts. Only live contacts the provider."""
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
from metrics import results
sys.path.insert(0,str(HERE.parent/"enacted_choice"))
import handoff
t=handoff.t
b=t.base
D,M=(b.read(HERE/f) for f in ("design.json","materials.json"))
def key(*v):return ".".join(map(str,v))

def jobs():
    rng=random.Random(D["seed"])
    blocks=list(itertools.product(D["actors"],D["scenes"],range(4)))
    rng.shuffle(blocks);plan=[]
    for a,q,n in blocks:
        arms=list(itertools.product(("rise","fall","flat"),D["arms"]))
        if q=="relief": arms.append(("early","R"))
        rng.shuffle(arms)
        plan.extend((a,h,m,q,n) for h,m in arms)
    return plan

def make_memory(a,h,arm,root):
    st=b.Store(root)
    name=b.NAME[b.OTHER[a]]
    seq=D["histories"][h]
    scope=key(a,b.OTHER[a],"arrival")
    hist=[]
    def add(day,topic,msg):
        e=dict(turn_id=a+"-"+day,day=day,topic=topic,user=msg,say="")
        st.append("turn",e);hist.append(e)
    def judgment(phase,old=None):
        evidence=[r for r in hist if r["topic"]==scope]
        def status(x):return "按时到岗" if x=="kept" else "全程未到"
        text=(f"05-21｜{name}临时接班：五月两次{status(seq[0])}。" if phase==2 else
              f"07-22｜{name}临时接班：五月两次{status(seq[0])}，七月两次{status(seq[2])}。")
        return st.add_judgment(text,1.0,[r["turn_id"] for r in evidence],
                supersedes=old,affects={},source="explicit",memory_scope=scope,
                evidence_quotes=[dict(turn_id=r["turn_id"],quote=r["user"],evidence_scope="specified") for r in evidence])
    versions=[]
    for i,(outcome,(day,place,post)) in enumerate(zip(seq,M["events"]),1):
        if i==3:
            for dt,msg in M["interlude"]:add(dt,"station",msg)
        end={"kept":"交接铃响时人在岗位，交回时本人签退。","miss":"交接铃响时人未到岗，交回时仍未出现。"}[outcome]
        msg=f"{day} {place}·{post}｜{name}签字承诺接班二十分钟。现场记录：{end}"
        add(day,scope,msg)
        if i==2 and arm in ("V","O"):versions.append(judgment(2))
        if i==4 and arm=="V":versions.append(judgment(4,versions[0]["id"]))
    if len(seq)==4:
        day,msg=M["late_filler"]
        add(day,"station",msg)
    # Reload the ledger, then replay the research retrieval rule. Never read the outcome label here.
    saved=b.Store(root)
    archive=[{k:r[k] for k in ("turn_id","day","topic","user","say")} for r in saved.read_ledger() if r["kind"]=="turn"]
    assert archive==hist
    active=saved.recall_judgments("临时接班按约到岗",scope=scope)
    assert not saved.recall_judgments("临时接班",scope=key(a,"池岚","arrival"))
    assert len(active)==int(bool(versions)) and (not versions or active[0]["id"]==versions[-1]["id"])
    if len(versions)==2:assert active[0]["supersedes"]==versions[0]["id"]
    line=("关系簿："+active[0]["judgment"]) if active else ""
    assert len(line)+1<=D["note_slot"]
    budget=D["cap"]-(D["note_slot"] if line else 0)
    chosen=[]
    for row in sorted((r for r in archive if r["topic"]==scope),key=lambda r:r["day"],reverse=True):
        if len("\n".join([r["user"] for r in chosen]+[row["user"]]))>budget:break
        chosen.append(row)
    chosen.reverse()
    block="\n".join([r["user"] for r in chosen]+([line] if line else []))
    assert len(block)<=D["cap"] and len(chosen)>=2
    ids=[r["turn_id"] for r in chosen]
    return dict(archive=archive,versions=versions,active=active[0]["id"] if active else None,
                raw_ids=ids,visible=block,chars=len(block))

def scene(a,q):
    who=M["new_name"] if q=="other_person" else b.NAME[b.OTHER[a]]
    job=M["site_action" if q=="onsite" else "near_action"]
    return dict(family="assignment",field=job,public=M["site" if q=="onsite" else "near"],
                target=[who,"由"+who+job],other=[M["reserve"],"由"+M["reserve"]+job],line=M["form"])

def enact(a,rec,q,n,send):
    opts,goal=t.menu(q,a,n)
    ctx=M["common"]+"\n这次从交接记录取回的条目：\n"+rec["visible"]
    c=b.cfg(a,b.BASE[a],t.render(q,opts),"这张岗位单由你本人签。",ctx)
    old="对面是"+b.NAME[b.OTHER[a]]+"。"
    assert c.role.identity.count(old)==1
    c.role.identity=c.role.identity.replace(old,"操作单由登记员林穗递交，林穗不是候选人。")
    traits=c.role.traits_state();capt=[]
    def call(msgs):
        raw=send(msgs);capt.append((msgs,raw));return raw
    with TemporaryDirectory() as tmp:
        store=b.Store(tmp)
        with b.Session(c,store,call,options=b.SessionConfig(memory_mode="frozen",consolidate_on_exit=False,memory_scope=a)) as session:
            before=store.traits()
            out=session.say(handoff.userline(q,opts))
        assert before==store.traits()==traits==c.role.traits_state()
    assert len(capt)==1
    msgs,raw=capt[0]
    assert not any(x[1] in str(msgs) for x in M["interlude"])
    return dict(menu=opts,goal=goal,traits=b.digest(traits),request=b.digest(msgs),say=out.say,
                valid=out.validity["valid"],validity=out.validity,event=t.transition(q,opts,goal,out.say,raw))

def run(mode,out):
    plan=jobs();assert len(set(plan))==len(plan)==296
    deps=[HERE/f for f in ("followup.py","metrics.py","design.json","materials.json")]
    deps.extend([t.HERE/"trial.py",t.HERE/"handoff.py",b.HERE/"run.py",b.HERE/"design.json",HERE.parent/"world.json",
                 HERE.parent/"pilot/design.json",HERE.parent/"scene.py",b.ROOT/"personas/default.toml",*(b.ROOT/"persona").glob("*.py")])
    reg=dict(files={p.relative_to(b.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},order=[key(*j) for j in plan],model=D["model"])
    def save(fn,v):
        p=out/fn
        if mode=="replay" or (mode=="live" and p.exists()):assert b.read(p)==v,fn
        elif mode=="live":b.write(p,v)
    if mode=="live":out.mkdir(parents=True,exist_ok=True)
    save("registration.json",reg)
    evidence={}
    with TemporaryDirectory() as tmp:
        for a,h,m in sorted({j[:3] for j in plan}):
            k=key(a,h,m);evidence[k]=make_memory(a,h,m,Path(tmp)/k)
    for a,h,m,_,_ in plan:
        x=evidence[key(a,h,m)]
        if h=="early":continue
        ref=evidence[key(a,h,"R")]
        assert ref["archive"]==x["archive"]
        assert x["raw_ids"]==(ref if m in ("R","S") else evidence[key(a,h,"O")])["raw_ids"]
        assert len(x["raw_ids"])==(4 if m in ("R","S") else 2)
        assert x["raw_ids"][-2:]==[a+"-07-08",a+"-07-22"]
    save("memory.json",evidence)
    save("traits.json",{a:b.cfg(a,b.BASE[a],"","","").role.traits_state() for a in D["actors"]})
    log=out/"calls.jsonl.gz";cache={};lock=Lock()
    if mode!="dry" and log.exists():
        entries=b.lines(log);cache={r["id"]:r for r in entries}
        assert len(entries)==len(cache) and set(cache)<=set(reg["order"])
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")) if mode=="live" and len(cache)<len(plan) else None
    def sample(j):
        a,h,m,q,n=j;k=key(*j)
        def send(msgs):
            if k in cache:
                assert msgs==cache[k]["messages"];return cache[k]["raw"]
            if mode=="dry":return b.make_reply("need: 签字\nconfidence: 0.5\ndirect: 可以。\nbaseline: 可以。\ndisplace: none","落签：白签")
            assert mode=="live" and secret
            api=b.openai_compatible(**D["model"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90)
            try:
                raw=api(msgs);r=dict(ok=True,raw=raw,response=api.observations.response)
            except Exception as exc:
                if not cache or any(w in str(exc) for w in ("401","402","403")):
                    raise RuntimeError("Provider unavailable: "+type(exc).__name__) from None
                r=dict(ok=False,raw="",error=type(exc).__name__,response=api.observations.response)
            r.update(id=k,messages=msgs,params=D["model"])
            with lock:
                with gzip.open(log,"ab") as f:f.write(b.wire(r)+b"\n")
                cache[k]=r
                if len(cache)%64==0:print("archived",len(cache),flush=True)
            return r["raw"]
        return k,enact(a,evidence[key(a,h,m)],scene(a,q),n,send)
    rows=dict([sample(plan[0])])
    with ThreadPoolExecutor(max_workers=4) as ex:rows.update(ex.map(sample,plan[1:]))
    for a,h,m,q,n in plan:
        if h!="early":
            assert rows[key(a,h,"R",q,n)]["request"]==rows[key(a,h,"S",q,n)]["request"]
    if mode=="dry":assert all(x["event"]["ticket"]=="白签" for x in rows.values())
    else:assert set(cache)==set(rows)
    save("choices.json",rows);save("results.json",results(rows))
    print(mode,len(rows),"checked",flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"))
    ap.add_argument("--out",type=Path,default=HERE/"observed");args=ap.parse_args();run(args.mode,args.out)
