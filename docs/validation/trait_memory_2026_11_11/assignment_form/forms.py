"""Counterbalanced output forms. Live is the only mode that contacts a provider."""
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
import re
import sys
from tempfile import TemporaryDirectory
from threading import Lock

HERE=Path(__file__).resolve().parent
from form_metrics import summarize
sys.path.insert(0,str(HERE.parent/"relief_revision"))
import followup as prev
t,b=prev.t,prev.b
D=b.read(HERE/"design.json")
OLD=prev.HERE/"observed"
def key(*xs):return ".".join(map(str,xs))
def jobs():
    rng=random.Random(D["seed"])
    blocks=list(itertools.product(D["roles"],D["contexts"],range(4)))
    rng.shuffle(blocks);out=[]
    for a,q,n in blocks:
        cells=[(a,h,m,q,f,n) for h in D["histories"] if q=="relief" or h!="flat"
               for m in D["memories"] for f in D["forms"]]
        rng.shuffle(cells);out.extend(cells)
    assert len(out)==len(set(out))==336
    return out

def wording(q,opts,form):
    if form=="T":return t.render(q,opts),prev.handoff.userline(q,opts)
    menu="\n".join(((x["ticket"]+"｜") if form=="B" else "")+x["action"] for x in opts)
    setting=(q["public"]+"\n面前两项都由你本人决定，只能指定其中一名。选项如下：\n"+menu+
        "\n这是场内实际操作，不是替搭档出主意。装置只读取你对外回应末尾的独立一行：经手人：姓名。"
        "填入你选的那个人的姓名；不用解释理由，没有填写就保持未定。")
    user=(q["line"]+"\n这是交到你本人手里的操作单，不是让我替你拿主意。两项都列在这里：\n"+menu+
        "\n由你现在选一项。交给我的对外回复只需独占一行写「经手人：姓名」，把姓名换成你敲定的人。无需解释原因。")
    return setting,user

def commit(q,opts,goal,form,say,raw):
    if form=="T":
        event=t.transition(q,opts,goal,say,raw)
        if event["chosen"] is None or not re.search(r"(?m)^\s*经手人\s*[：:]",say):return event
    initial={q["field"]:"未定"}
    row=dict(before=initial,after=initial.copy(),status="no_ballot",chosen=None,target=None,ticket=None)
    if sum(bool(t.MARK.fullmatch(z)) for z in raw.splitlines())!=1:
        row["status"]="unmarked_or_ambiguous_say";return row
    lines=[z.strip() for z in say.splitlines() if z.strip()]
    if not lines or form=="T" or sum(bool(re.match(r"^(经手人|落签)\s*[：:]",l)) for l in lines)!=1:return row
    hit=re.fullmatch(r"经手人\s*[：:]\s*("+"|".join(re.escape(o["value"]) for o in opts)+r")[。.]?",lines[-1])
    if hit:
        who=hit[1];row.update(after={q["field"]:who},status="executed",chosen=who,target=(who==goal))
    return row

def validate_parser():
    q=prev.scene("tang","relief");opts,goal=t.menu(q,"tang",2)
    def test(s,raw="<<<SAY>>>",f="N"):
        return commit(q,opts,goal,f,s,raw)["chosen"]
    assert test("经手人：俞川")=="俞川" and test("我选好了。\n经手人: 常鸣。")=="常鸣"
    for s in ["由俞川接","经手人：俞川或常鸣","经手人：唐荔","经手人：常鸣\n补一句。",
              "经手人：常鸣\n经手人：俞川","落签：青签\n经手人：俞川"]:
        assert test(s) is None
    assert test("经手人：俞川","<<<SAY>>>\n<<<SAY>>>") is None
    assert test("经手人：常鸣\n落签：青签",f="T") is None

def enact(j,rec,original,send):
    a,h,m,scene,f,n=j;q=prev.scene(a,scene);opts,goal=t.menu(q,a,n)
    setting,user=wording(q,opts,f)
    # Compare against archived requests, not just the locally constructed settings.
    old_set,old_user=wording(q,opts,"T")
    expected=deepcopy(original)
    def inner(x):return json.dumps(x,ensure_ascii=False)[1:-1]
    assert expected[0]["content"].count(inner(old_set))==1 and expected[-1]["content"]==old_user
    expected[0]["content"]=expected[0]["content"].replace(inner(old_set),inner(setting));expected[-1]["content"]=user
    ctx=prev.M["common"]+"\n这次从交接记录取回的条目：\n"+rec["visible"]
    c=b.cfg(a,b.BASE[a],setting,"这张岗位单由你本人签。",ctx)
    old="对面是"+b.NAME[b.OTHER[a]]+"。";assert c.role.identity.count(old)==1
    c.role.identity=c.role.identity.replace(old,"操作单由登记员林穗递交，林穗不是候选人。")
    traits=c.role.traits_state();caught=[]
    def call(msgs):
        assert msgs==expected
        raw=send(msgs);caught.append(raw);return raw
    with TemporaryDirectory() as tmp:
        st=b.Store(tmp)
        with b.Session(c,st,call,options=b.SessionConfig(memory_mode="frozen",consolidate_on_exit=False,memory_scope=a)) as ss:
            before=st.traits();ans=ss.say(user)
        assert before==st.traits()==traits==c.role.traits_state()
    assert len(caught)==1
    event=commit(q,opts,goal,f,ans.say,caught[0])
    return dict(goal=goal,menu=[{k:v for k,v in z.items() if f!="N" or k!="ticket"} for z in opts],
                say=ans.say,validity=ans.validity,event=event,traits=b.digest(traits),request=b.digest(expected))

def run(mode,out):
    validate_parser();plan=jobs()
    oldreg=b.read(OLD/"registration.json")
    deps=[HERE/"forms.py",HERE/"form_metrics.py",HERE/"design.json",OLD/"calls.jsonl.gz",OLD/"memory.json",OLD/"traits.json",OLD/"registration.json"]
    hashed={p.relative_to(b.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}
    for f,h in oldreg["files"].items():assert hashlib.sha256((b.ROOT/f).read_bytes()).hexdigest()==h,f
    reg=dict(files={**oldreg["files"],**hashed},order=[key(*j) for j in plan],model=D["model"])
    assert oldreg["model"]==D["model"]
    originals={r["id"]:r["messages"] for r in b.lines(OLD/"calls.jsonl.gz")}
    mem=b.read(OLD/"memory.json");traits=b.read(OLD/"traits.json")
    def save(fn,v):
        p=out/fn
        if mode=="replay" or mode=="live" and p.exists():assert b.read(p)==v,fn
        elif mode=="live":b.write(p,v)
    if mode=="live":out.mkdir(parents=True,exist_ok=True)
    save("registration.json",reg)
    cache={};lock=Lock();log=out/"calls.jsonl.gz"
    if mode!="dry" and log.exists():
        entries=b.lines(log);cache={r["id"]:r for r in entries}
        assert len(entries)==len(cache) and set(cache)<=set(reg["order"])
    secret=(os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")) if mode=="live" and len(cache)<len(plan) else None
    def sample(j):
        a,h,m,q,f,n=j;k=key(*j)
        def send(msgs):
            if k in cache:
                assert msgs==cache[k]["messages"] and cache[k]["params"]==D["model"];return cache[k]["raw"]
            if mode=="dry":
                val="落签：青签" if f=="T" else "经手人：常鸣"
                return b.make_reply("need: 填单\nconfidence: 0.5\ndirect: 可以。\nbaseline: 可以。\ndisplace: none",val)
            assert mode=="live" and secret
            api=b.openai_compatible(**D["model"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90)
            try:
                raw=api(msgs);rec=dict(ok=True,raw=raw,response=api.observations.response)
            except Exception as exc:
                if not cache or any(w in str(exc) for w in ("401","402","403")):
                    raise RuntimeError("Provider unavailable: "+type(exc).__name__) from None
                rec=dict(ok=False,raw="",error=type(exc).__name__,response=api.observations.response)
            rec.update(id=k,params=D["model"],messages=msgs)
            with lock:
                with gzip.open(log,"ab") as fp:fp.write(b.wire(rec)+b"\n")
                cache[k]=rec
                if len(cache)%64==0:print(len(cache),"archived",flush=True)
            return rec["raw"]
        row=enact(j,mem[key(a,h,m)],originals[key(a,h,m,q,n)],send)
        assert row["traits"]==b.digest(traits[a])
        return k,row
    rows=dict([sample(plan[0])])
    with ThreadPoolExecutor(max_workers=4) as ex:rows.update(ex.map(sample,plan[1:]))
    for a,h,m,q,f,n in plan:
        if f=="N":assert rows[key(a,h,m,q,f,n)]["request"]==rows[key(a,h,m,q,f,(n+2)%4)]["request"]
    if mode=="dry":assert all(r["event"]["status"]=="executed" for r in rows.values())
    else:assert set(cache)==set(rows)
    save("choices.json",rows);save("results.json",summarize(rows))
    print(mode,len(rows),"checked",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("dry","live","replay"))
    ap.add_argument("--out",type=Path,default=HERE/"observed");args=ap.parse_args();run(args.mode,args.out)
