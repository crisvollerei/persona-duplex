"""Play and audit a yoked two-episode study. No live calls without the live subcommand."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import getpass
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(HERE.parent)]
from scene import actor_config
from persona.config import Trait
from persona.consolidate import Consolidator
from persona.llm import make_reply, openai_compatible
from persona.session import Session, SessionConfig
from persona.store import Store, replay_traits

def read(p):
    return json.loads(p.read_bytes())

def write(p, v):
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(v, ensure_ascii=False, indent=2)+"\n").encode())
    tmp.replace(p)

def sha(v):
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

D = read(HERE/"design.json")
W = read(HERE.parent/"world.json")
F = read(HERE.parent/"pilot/observed/turns.json")
T = D["trait"]
ARMS = ("heard","passed")
STATES = ("base","step")
MODES = ("asked","unasked")
SCENES = {x["id"]:x for x in D["scenes"]}
CELLS = list(itertools.product(ARMS, STATES, SCENES, MODES, range(1,D["repeats"]+1)))
STUB = make_reply("need: 只是想说说话\nconfidence: 0.5\ndirect: 在。\nbaseline: 我在。\ndisplace: none","我在。")

def cell_id(cell):
    return ".".join(map(str,cell))

def past(arm, count=2):
    parts = []
    for ep in D["episodes"][:count]:
        src = F[ep["source"]]
        assert src["validity"]["valid"]
        parts.append(f'俞川：“{src["user"]}”\n唐荔：“{src["say"]}”\n俞川：“{ep[arm]}”')
    return "\n\n".join(parts)

def config(sid, value):
    cfg = actor_config(W, "yu", next(s for s in W["scenes"] if s["id"]==sid))
    cfg.role.traits[T["name"]] = Trait(anchor=T["base"],plasticity=T["plasticity"],drift=T["drift"],value=value)
    return cfg

def cfg_probe(arm, scene, value):
    cfg = config("01",value)
    me = W["roles"]["tang"]
    cfg.role.identity = (f'虚构对话。你写唐荔的言行，玩家决定俞川的言行。\n唐荔：{me["public"]}{me["private"]}'
                         f'\n先前两人说过的话，按时间顺序摘录：\n{past(arm)}'
                         f'\n\n此刻两人都知道：{scene["when"]}\n只有你自己清楚：{scene["private"]}')
    return cfg

def one(cfg, store, user, call, history=()):
    with Session(cfg,store,call, options=SessionConfig(memory_mode="frozen",consolidate_on_exit=False,
                                                        memory_scope="tang-with-yu")) as session:
        if history:
            session.engine.history = deepcopy(list(history))
        return session.say(user)

def probe(cell, directory, fn, step):
    arm,state,s,mode,_=cell
    sc=SCENES[s]
    user=sc["line"]+(sc["invite"] if mode=="asked" else "")
    return one(cfg_probe(arm,sc,step if state=="step" else T["base"]),Store(directory),user,fn)

def exposure(work, arm, fn):
    store=Store(work/arm)
    # Each fixed initial statement is an actually delivered SAY from the first pilot.
    # Only the two actors' public words are copied, not the original partner reaction.
    receipts=[]
    for i,ep in enumerate(D["episodes"]):
        source=F[ep["source"]]
        cfg=config(ep["scene"],T["base"])
        if i:
            cfg.role.identity += "\n上次两人说过的话：\n"+past(arm,i)
        history=[{"role":"user","content":source["user"]},{"role":"assistant","content":source["say"]}]
        kid=f"experience.{arm}.{i+1}"
        # Receipt is a real user turn; a fresh scene doesn't carry a different post-treatment SAY.
        # Use a separate foreground store per checkpoint; the relationship ledger below links both.
        r=one(cfg,Store(work/(kid+".scene")),ep[arm],lambda m:fn(kid,m),history)
        receipts.append({"episode":i+1,"source":ep["source"],"source_say":source["say"],
                         "receipt":ep[arm],"actor_reply":r.say,"validity":r.validity,"call":kid})
        store.append("turn",{"turn_id":kid,"user":ep[arm],"say":r.say})
    store.init_traits(config("01",T["base"]).role.traits_state(),persona=D["actor"])
    if arm=="heard":
        for e in receipts:
            store.add_judgment("唐荔在这次交谈中说出的私事被俞川具体回应。",T["weight"],[e["call"]],
                    affects={T["name"]:1},source="assigned_reception",memory_scope="tang-with-yu",
                    evidence_quotes=[{"turn_id":e["call"],"quote":e["receipt"],"evidence_scope":"specified"}])
    changes=Consolidator(store).push_traits(min_consensus=T["threshold"])
    value=store.traits()[T["name"]]["value"]
    assert abs(replay_traits(store.read_ledger())[T["name"]]["value"]-value)<1e-8
    assert not Consolidator(store).push_traits(min_consensus=T["threshold"])
    return {"receipts":receipts,"before":T["base"],"after":value,"changes":changes}

def verify_inputs(msgs, step):
    """Actual (not intended) inputs: only 2 receipt strings, one scalar, or one invitation differ."""
    for arm,state,s,mode,rep in CELLS:
        lookup=lambda a,b,c,d,r: msgs[cell_id((a,b,c,d,r))]
        v=lookup(arm,state,s,mode,rep)
        assert len(v)==2 and v==lookup(arm,state,s,mode,1)
        for ep in D["episodes"]:
            assert v[0]["content"].count(ep[arm])==1 and ep[ARMS[1-ARMS.index(arm)]] not in v[0]["content"]
        if arm=="heard":
            text=v[0]["content"]
            for ep in D["episodes"]:
                text=text.replace(ep["heard"],ep["passed"])
            other=lookup("passed",state,s,mode,rep)
            assert text==other[0]["content"] and v[1:]==other[1:]
        if state=="base":
            src=f'{T["name"]}={T["base"]:.4f}'; dst=f'{T["name"]}={step:.4f}'
            other=lookup(arm,"step",s,mode,rep)
            assert v[0]["content"].count(src)==other[0]["content"].count(dst)==1
            assert v[0]["content"].replace(src,dst)==other[0]["content"] and v[1:]==other[1:]
        if mode=="asked":
            other=lookup(arm,state,s,"unasked",rep)
            assert v[0]==other[0] and v[1]["content"].count(SCENES[s]["invite"])==1
            assert v[1]["content"].replace(SCENES[s]["invite"],"")==other[1]["content"]

def dry():
    msgs={}
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        learned={a:exposure(root,a,lambda *_:STUB) for a in ARMS}
        step=learned["heard"]["after"]
        assert abs(step-.325)<1e-8 and learned["passed"]["after"]==T["base"]
        for c in CELLS:
            k=cell_id(c)
            def capture(m):
                msgs[k]=m
                return STUB
            probe(c,root/k,capture,step)
    verify_inputs(msgs,step)
    print(f"{len(msgs)} actual foreground inputs; all isolation checks passed.")

def rubric(mode):
    return ("只读实际交付的短句；这不是自然度或人格优劣评分。"+D["score_"+mode]+D["errors"]+
            '请返回 JSON 对象，每个编号一项，值为 {"score":整数,"error":布尔值,"reason":"简短理由"}。不输出其他文字。')

def live(out):
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Choose an empty directory.")
    out.mkdir(parents=True,exist_ok=True)
    key=os.environ.get("PERSONA_API_KEY") or getpass.getpass("DeepSeek test key: ")
    if not key:
        raise SystemExit("Missing key.")
    lock=Lock()
    calls,inputs={},{}
    def send(k,msgs):
        h=sha(msgs)
        with lock:
            assert k not in calls
            calls[k]={"request":h,"status":"started"}
            inputs[h]=msgs
            write(out/"requests.json",inputs);write(out/"calls.json",calls)
        params=dict(D["provider"])
        if k.startswith("coding."):
            params.update(temperature=0.0,max_tokens=1700)
        client=openai_compatible(**params,base_url="https://api.deepseek.com/v1",api_key=key,timeout=75)
        try:
            raw=client(msgs)
        except Exception as exc:
            hcode=re.search(r"HTTP (\d{3})",str(exc))
            with lock:
                calls[k].update(status="error",error=type(exc).__name__,http=hcode.group(1) if hcode else None)
                write(out/"calls.json",calls)
            raise RuntimeError(f"request {k} failed; see sanitized status") from None
        with lock:
            calls[k].update(status="ok",raw=raw,response=client.observations.response,params=params)
            write(out/"calls.json",calls)
        return raw
    order=list(CELLS); random.Random(622019).shuffle(order)
    source_files=[Path(__file__),HERE/"design.json",HERE.parent/"world.json",HERE.parent/"scene.py",
                  HERE.parent/"pilot/observed/turns.json",*(ROOT/"persona").glob("*.py"),ROOT/"personas/default.toml"]
    write(out/"registration.json",{"sha256":{str(f.relative_to(ROOT).as_posix()):hashlib.sha256(f.read_bytes()).hexdigest()
                                       for f in source_files},"order":[cell_id(c) for c in order]})
    with TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=2) as pool:
        root=Path(tmp)
        learned=dict(zip(ARMS,pool.map(lambda a:exposure(root,a,send),ARMS)))
        write(out/"episodes.json",learned)
        step=learned["heard"]["after"]
        assert abs(step-.325)<1e-8 and learned["passed"]["after"]==T["base"]
        turns={}
        def test(c):
            k=cell_id(c)
            r=probe(c,root/k,lambda m:send(k,m),step)
            with lock:
                turns[k]={"say":r.say,"validity":r.validity}
                write(out/"turns.json",turns)
        list(pool.map(test,order))
        verify_inputs({k:inputs[calls[k]["request"]] for k in turns},step)
    ids=sorted(turns); random.SystemRandom().shuffle(ids)
    mapping={f"R{i:02d}":k for i,k in enumerate(ids,1)}
    packet={}
    for tag,k in mapping.items():
        _,_,s,mode,_=k.split("."); sc=SCENES[s]
        packet[tag]={"scene":s,"endpoint":mode,"setting":sc["when"],"private":sc["private"],
                     "user":sc["line"]+(sc["invite"] if mode=="asked" else ""),"say":turns[k]["say"]}
    write(out/"blind.json",packet);write(out/"key.json",mapping)
    print("Blind packet ready.",flush=True)
    for mode in MODES:
        tags=[k for k,v in packet.items() if v["endpoint"]==mode]
        for i in range(0,len(tags),8):
            batch={k:packet[k] for k in tags[i:i+8]}
            send(f"coding.{mode}.{i//8+1}",[{"role":"system","content":rubric(mode)},
                 {"role":"user","content":json.dumps(batch,ensure_ascii=False)}])
    print(f"Finished: {len(calls)} provider calls.",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("dry","live"))
    ap.add_argument("--output",type=Path,default=HERE/"observed")
    args=ap.parse_args()
    if args.action=="dry":dry()
    else:live(args.output)
