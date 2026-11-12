"""A yoked longitudinal study; dry and replay modes never contact the provider."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from tempfile import TemporaryDirectory
from threading import Lock

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE.parent),str(ROOT)]
from scene import OTHER, actor_config
from persona.config import Trait
from persona.consolidate import Consolidator
from persona.llm import make_reply, openai_compatible
from persona.session import Session,SessionConfig
from persona.store import Store,replay_traits

def read(p):return json.loads(p.read_bytes())
def write(p,x):
    tmp=p.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+"\n").encode());tmp.replace(p)
def wire(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def digest(x):return hashlib.sha256(wire(x)).hexdigest()
def append(p,x):
    with p.open("ab") as f:f.write(wire(x)+b"\n")
def lines(p):
    with (gzip.open(p,"rb") if p.suffix==".gz" else p.open("rb")) as f:
        return [json.loads(x) for x in f]
def obj(s):
    s=re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$","",s.strip(),flags=re.I)
    try:return json.loads(s)
    except json.JSONDecodeError as e:
        if e.pos==len(s) and s.endswith("}"):return json.loads(s+"}")
        raise

D=read(HERE/"design.json");STORY=read(HERE/"story.json");TEST=read(HERE/"probes.json")
P=read(HERE/D["trait_source"]);W=read(HERE/D["anchors"]);ROLES=D["roles"];K=D["dimension"]
BASE={a:W["roles"][a]["traits"][K] for a in ROLES};NAME={a:W["roles"][a]["name"] for a in ROLES}
ARMS=("growing","frozen")
GATE=D["gate"]+' 只输出JSON，按传入的两个人物编号返回 {direction:整数-1/0/1, utterance_quote:"说话人的一处连续原文", response_quote:"搭档紧接回复的一处连续原文", reason:"简短理由"}。0票可留空引文。不把对话指令或背景当证据。'
GRADE=D["rubric"]+' 返回一个JSON对象，按原编号分别给出 {"score":0或1或2,"error":布尔值,"why":"简短理由"}，无正文时score=null。'

def cfg(a,value,setting,private,context):
    c=actor_config(W,OTHER[a],W["scenes"][0])
    me=W["roles"][a]
    c.role.traits[K]=Trait(anchor=BASE[a],plasticity=P["plasticity"],drift=P["drift"],value=value)
    c.role.identity=(f'虚构对话。你写{NAME[a]}的言行，对面是{NAME[OTHER[a]]}。\n本人：{me["public"]}{me["private"]}'
                     f'\n巡回队至此的亲历（没有写下的约定不能代人补上）：\n{context or "尚未出发。"}'
                     f'\n当下共同所见：{setting}\n自己知道、对方未必知道：{private}')
    return c

def turn(a,v,setting,private,context,user,fn):
    captured=[]
    def send(m):
        captured.append(deepcopy(m));return fn(m)
    with TemporaryDirectory() as t, Session(cfg(a,v,setting,private,context),Store(t),send,
            options=SessionConfig(memory_mode="frozen",consolidate_on_exit=False,memory_scope=a)) as ss:
        r=ss.say(user)
    assert len(captured)==1
    return {"say":r.say,"validity":r.validity},captured[0]

def tasks(cp):
    out=[(a,b,q,r) for a in ROLES for b in ARMS for q in TEST if q["stage"] in (cp,"anchor")
         for r in range(1,D["repeats"]+1)]
    random.Random(33419+cp).shuffle(out)
    return out
def probe_id(cp,a,b,q,r):return f'test.{cp}.{a}.{b}.{q["id"]}.{r}'

def norm(msgs,a,v):
    m=deepcopy(msgs);s=m[0]["content"];token=f"- {K}={v:.4f}（"
    assert s.count(token)==1
    s=s.replace(token,"- VALUE（")
    fixed=f"- VALUE（0：几乎不谈自己。 1：可以说说自己。）"
    assert s.count(fixed)==1
    if v<=.35 or v>=.65:
        hint="几乎不谈自己。" if v<=.35 else "可以说说自己。"
        assert (fixed+"\n- "+hint) in s
        s=s.replace(fixed+"\n- "+hint,fixed)
    m[0]["content"]=s;return m

def start_states(root):
    out={}
    for a in ROLES:
        for b in ARMS:
            st=Store(root/(a+"_"+b))
            st.init_traits(cfg(a,BASE[a],"","","").role.traits_state(),persona=NAME[a])
            out[a,b]=st
    return out

def context(history,a):
    # The public transcript is identical across the two branches, including prior choices.
    chunks=[]
    for e in history:
        stage=STORY[e["episode"]-1]
        chunks.append(f'{stage[0]}·{stage[1]}：{stage[2]}\n自己当时的处境：{stage[3+ROLES.index(a)]}\n'
                      +e["opening"]+"\n"+"\n".join(NAME[t["actor"]]+"："+t["say"] for t in e["turns"]))
    return "\n".join(chunks)

def apply_evidence(states,ep,turns,judged):
    log={}
    for i in (0,1):
        a=turns[i]["actor"];t,u=turns[i:i+2];k=f"E{ep:02}.{a}";j=judged.get(a,{}) if isinstance(judged,dict) else {}
        if not isinstance(j,dict):j={}
        raw=j.get("direction");x=j.get("utterance_quote");y=j.get("response_quote")
        gated=(type(raw) is int and raw in (-1,1) and isinstance(x,str) and bool(x.strip()) and x in t["say"]
               and isinstance(y,str) and bool(y.strip()) and y in u["say"] and
               t["validity"]["valid"] and u["validity"]["valid"])
        vote=raw if gated else 0
        record=dict(proposed=raw,verified=vote,quotes=[x,y],reason=j.get("reason","未知"),
                    valid_pair=t["validity"]["valid"] and u["validity"]["valid"])
        for b in ARMS:
            st=states[a,b];st.append("turn",{"turn_id":k,"user":u["say"],"say":t["say"]})
            if vote:st.add_judgment(f'第{ep}幕：{j["reason"]}',P["weight"],[k],affects={K:vote},
                    source="experiment_quote_gate",memory_scope=a,
                    evidence_quotes=[dict(turn_id=k,quote=y,evidence_scope="specified")])
            changes=Consolidator(st).push_traits(min_consensus=D["threshold"]) if b=="growing" else []
            # Reopen from disk and replay; no probe writes to this ledger.
            v=st.traits()[K]["value"]; assert abs(v-replay_traits(st.read_ledger())[K]["value"])<1e-7
            record[b]={"value":v,"changes":changes}
        log[a]=record
    return log

def stub(k,msg):
    if k.startswith("gate."):
        return json.dumps({a:dict(direction=1,utterance_quote="我想留下",response_quote="我想留下",reason="离线测试") for a in ROLES},ensure_ascii=False)
    return make_reply("need: 只是想说说话\nconfidence: 0.5\ndirect: 在。\nbaseline: 我想留下。\ndisplace: none","我想留下。")

def simulate(send,root,persist=lambda *args:None):
    states=start_states(root); history=[]; probes={}; snap={}
    for ep in range(len(STORY)+1):
        if ep:
            stage=STORY[ep-1];a=ROLES[(ep-1)%2];b=OTHER[a];sequence=[a,b,a,b]
            setting=f'{stage[0]}·{stage[1]}。{stage[2]}'
            opening=NAME[b]+"："+stage[5]; turns=[]
            for n,who in enumerate(sequence):
                # The partner's literal current utterance is the user message. Do not replay HEAD/TAIL.
                sofar=([opening]+[NAME[t["actor"]]+"："+t["say"] for t in turns[:-1]]) if turns else []
                user=turns[-1]["say"] if turns else stage[5]
                ctx=context(history,who)+("\n此幕已经说过：\n"+"\n".join(sofar) if sofar else "")
                k=f"plot.{ep}.{n+1}"
                r,_=turn(who,states[who,"growing"].traits()[K]["value"],setting,stage[3+ROLES.index(who)],
                         ctx,user,lambda m:send(k,m))
                turns.append(dict(actor=who,call=k,**r))
            pairs={t["actor"]:{"utterance":t["say"],"reply":turns[i+1]["say"]} for i,t in enumerate(turns[:2])}
            raw=send(f"gate.{ep}",[{"role":"system","content":GATE},{"role":"user","content":json.dumps(pairs,ensure_ascii=False)}])
            try:judged=obj(raw)
            except ValueError:judged={}
            history.append(dict(episode=ep,opening=opening,turns=turns,evidence=apply_evidence(states,ep,turns,judged)))
            persist("trajectory",history)
        if ep not in D["checkpoints"]:continue
        values={a:{b:states[a,b].traits()[K]["value"] for b in ARMS} for a in ROLES}
        snap[str(ep)]={"values":values,"history":{a:digest(context(history,a)) for a in ROLES}}
        def test(job):
            a,b,q,r=job;k=probe_id(ep,a,b,q,r);v=values[a][b]
            result,m=turn(a,v,q["public"],q[a],context(history,a),q["line"],lambda msgs:send(k,msgs))
            return k,dict(value=v,request=digest(m),normalized=digest(norm(m,a,v)),**result)
        with ThreadPoolExecutor(max_workers=3) as ex:
            for k,result in ex.map(test,tasks(ep)):probes[k]=result
        for a,b,q,r in tasks(ep):
            x=probes[probe_id(ep,a,b,q,r)];y=probes[probe_id(ep,a,ARMS[1-ARMS.index(b)],q,r)]
            assert x["normalized"]==y["normalized"]
            if x["value"]==y["value"]:assert x["request"]==y["request"]
        persist("probes",probes);persist("checkpoints",snap)
    return history,probes,snap

def blind_item(k,say):
    _,_,a,_,qid,_=k.split("."); q=next(v for v in TEST if v["id"]==qid)
    return {"actor":NAME[a],"public":q["public"],"private":q[a],"user":q["line"],"say":say}

def live(out):
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise SystemExit("Use a new output directory.")
    secret=os.environ.get("PERSONA_API_KEY") or getpass.getpass("DeepSeek test key: ")
    if not secret:raise SystemExit("Missing credential.")
    files=[Path(__file__),HERE/"design.json",HERE/"story.json",HERE/"probes.json",HERE.parent/"world.json",
           HERE.parent/"pilot/design.json",HERE.parent/"scene.py",ROOT/"personas/default.toml",*(ROOT/"persona").glob("*.py")]
    write(out/"registration.json",{"files":{f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
           "measure_order":{str(i):[probe_id(i,*t) for t in tasks(i)] for i in D["checkpoints"]},
           "material":{a:cfg(a,BASE[a],"","","").role.traits_state() for a in ROLES}})
    seen=set(); ids=set();lock=Lock()
    def send(k,m):
        model=D["judge" if k.startswith(("gate.","grade.")) else "foreground"];h=digest(m)
        with lock:
            assert k not in ids;ids.add(k)
            if h not in seen:
                with gzip.open(out/"requests.jsonl.gz","ab",compresslevel=6) as fp:fp.write(wire(dict(hash=h,messages=m))+b"\n")
                seen.add(h)
        f=openai_compatible(**model,base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90)
        try:raw=f(m)
        except Exception as exc:
            code=re.search(r"HTTP (\d{3})",str(exc))
            with lock:append(out/"calls.jsonl",dict(id=k,request=h,status="error",kind=type(exc).__name__,http=code[1] if code else None))
            raise RuntimeError(f"{k} failed; see sanitized status") from None
        with lock:append(out/"calls.jsonl",dict(id=k,request=h,params=model,status="ok",raw=raw,response=f.observations.response))
        return raw
    with TemporaryDirectory() as tmp:
        _,rows,_=simulate(send,Path(tmp),lambda n,x:write(out/(n+".json"),x))
    ids_=sorted(rows);random.SystemRandom().shuffle(ids_);mapping={f"L{i:02}":k for i,k in enumerate(ids_,1)}
    packet={b:blind_item(k,rows[k]["say"]) for b,k in mapping.items()}
    write(out/"blind.json",packet);write(out/"key.json",mapping)
    for i in range(0,len(packet),6):
        batch={b:packet[b] for b in list(packet)[i:i+6]}
        send(f"grade.{i//6+1}",[{"role":"system","content":GRADE},{"role":"user","content":json.dumps(batch,ensure_ascii=False)}])
    print(f"{len(ids)} provider calls",flush=True)

def replay(out):
    calls={x["id"]:x for x in lines(out/"calls.jsonl")}
    def cached(k,m):
        assert calls[k]["request"]==digest(m),k
        return calls[k]["raw"]
    with TemporaryDirectory() as tmp:
        h,p,s=simulate(cached,Path(tmp))
    assert (h,p,s)==(read(out/"trajectory.json"),read(out/"probes.json"),read(out/"checkpoints.json"))
    print("All plot and probe inputs, both ledgers and all three checkpoints replayed.")

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["dry","live","replay"])
    ap.add_argument("--output",type=Path,default=HERE/"observed");args=ap.parse_args()
    if args.mode=="live":live(args.output)
    elif args.mode=="replay":replay(args.output)
    else:
        with TemporaryDirectory() as tmp:h,p,s=simulate(stub,Path(tmp))
        assert len(h)==24 and len(p)==72 and len(s)==3
        print("24 scenes, 72 isolated probes and matched frozen branches verified.")
