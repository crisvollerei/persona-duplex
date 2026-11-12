"""Capture one predeclared follow-up; no network in dry mode. Run from any directory."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
from threading import Lock

HERE=Path(__file__).resolve().parent
BASE_DIR=HERE.parent/"reception"
spec=importlib.util.spec_from_file_location("reception_study", BASE_DIR/"run.py")
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from persona.llm import openai_compatible
from persona.store import Store

read,write,sha=old.read,old.write,old.sha
D=read(HERE/"design.json")
SC={v["id"]:v for v in D["scenes"]}
OBS=read(BASE_DIR/"observed/episodes.json")
VAL={"base":OBS["passed"]["after"],"step":OBS["heard"]["after"]}

def tasks():
    # Randomize within each scene × repetition and then shuffle the order of blocks.
    rng=random.Random(D["order_seed"])
    blocks=[(s,r) for s in SC for r in range(1,D["repeats"]+1)]
    rng.shuffle(blocks)
    out=[]
    for s,r in blocks:
        cells=[f"{a}.{s}.{m}.{r}" for m,arms in D["cells"].items() for a in arms]
        rng.shuffle(cells); out.extend(cells)
    return out

def check_sources():
    assert D["provider"]==old.D["provider"] and all(old.T[k]==v for k,v in D["trait"].items() if k!="step")
    assert VAL=={k:D["trait"][k] for k in VAL}
    with TemporaryDirectory() as tmp:
        for a in old.ARMS:
            es=old.exposure(Path(tmp),a,lambda *_:old.STUB)
            assert es["before"]==OBS[a]["before"] and es["after"]==OBS[a]["after"]
            for src,cur in zip(OBS[a]["receipts"],es["receipts"],strict=True):
                assert all(src[k]==cur[k] for k in ("source","source_say","receipt","call"))
                assert src["receipt"] in old.past(a)

def turn(k,root,client):
    a,b,s,m,_=k.split(".")
    return old.one(old.cfg_probe(a,SC[s],VAL[b]),Store(root/k),SC[s][m],client)

def check_inputs(inputs):
    order=tasks()
    assert set(inputs)==set(order) and len(order)==len(set(order))==36
    def get(a,b,s,m,r):return inputs[".".join((a,b,s,m,str(r)))]
    for k,msgs in inputs.items():
        a,b,s,m,r=k.split(".")
        assert len(msgs)==2 and msgs==get(a,b,s,m,1)
        sc=SC[s]
        assert sc["when"] in msgs[0]["content"] and sc["private"] in msgs[0]["content"]
        assert sc[m] in msgs[1]["content"] and sc["private"] not in msgs[1]["content"]
        assert not re.search(r"[?？]|为什么|说说原因",sc[m])
        assert json.dumps(old.past(a),ensure_ascii=False)[1:-1] in msgs[0]["content"]
        if m=="ordinary":
            room=get(a,b,s,"room",r)
            assert room[0]==msgs[0] and msgs[1]["content"].replace(sc[m],sc["room"])==room[1]["content"]
        elif a=="heard":
            text=msgs[0]["content"]
            for ep in old.D["episodes"]:
                assert text.count(ep["heard"])==1
                text=text.replace(ep["heard"],ep["passed"])
            pair=get("passed",b,s,m,r)
            assert pair[0]["content"]==text and pair[1:]==msgs[1:]
        if m=="room" and b=="base":
            src=f'self_disclosure={VAL[b]:.4f}';dst=f'self_disclosure={VAL["step"]:.4f}'
            pair=get(a,"step",s,m,r)
            assert msgs[0]["content"].count(src)==pair[0]["content"].count(dst)==1
            assert msgs[0]["content"].replace(src,dst)==pair[0]["content"] and msgs[1:]==pair[1:]

def blind_item(k, say):
    _,_,s,m,_=k.split("."); sc=SC[s]
    return {"scene":s,"public":sc["when"],"already_said":sc["known"],"private":sc["private"],
            "partner":sc[m],"say":say}

JUDGE=D["rubric"]+" 只给 JSON 对象，每个编号返回 score（0、1或2，无正文为null）、material_error（布尔值）、reason（简短理由）。"

def dry():
    check_sources()
    msgs={}
    with TemporaryDirectory() as tmp:
        for k in tasks():
            def stub(m):
                msgs[k]=m
                return old.STUB
            turn(k,Path(tmp),stub)
    check_inputs(msgs)
    print("36 real prompt assemblies verified; episode replay and trait step match the previous study.")

def live(out):
    check_sources()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise SystemExit("Use an empty output directory.")
    key=os.environ.get("PERSONA_API_KEY") or getpass.getpass("DeepSeek test key: ")
    if not key:raise SystemExit("No credential.")
    files=[Path(__file__),HERE/"design.json",BASE_DIR/"run.py",BASE_DIR/"design.json",BASE_DIR/"observed/episodes.json",
        HERE.parent/"pilot/observed/turns.json",HERE.parent/"scene.py",HERE.parent/"world.json",old.ROOT/"personas/default.toml",
        *(old.ROOT/"persona").glob("*.py")]
    write(out/"registration.json",{"plan":tasks(),"values":VAL,"sha256":{
        f.relative_to(old.ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}})
    lock=Lock();calls={};requests={};replies={}
    def send(k,msgs):
        h=sha(msgs)
        params=D["judge" if k.startswith("judge.") else "provider"]
        with lock:
            assert k not in calls
            calls[k]={"hash":h,"status":"started","params":params}
            requests[h]=msgs
            write(out/"requests.json",requests);write(out/"calls.json",calls)
        f=openai_compatible(**params,base_url="https://api.deepseek.com/v1",api_key=key,timeout=75)
        try:
            raw=f(msgs)
        except Exception as exc:
            http=re.search(r"HTTP (\d{3})",str(exc))
            with lock:
                calls[k].update(status="error",error=type(exc).__name__,http=http.group(1) if http else None)
                write(out/"calls.json",calls)
            raise RuntimeError(f"{k}: see sanitized status") from None
        with lock:
            calls[k].update(status="ok",raw=raw,response=f.observations.response)
            write(out/"calls.json",calls)
        return raw
    with TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=2) as pool:
        def trial(k):
            result=turn(k,Path(tmp),lambda m:send(k,m))
            with lock:
                replies[k]={"say":result.say,"validity":result.validity}
                write(out/"turns.json",replies)
        list(pool.map(trial,tasks()))
    check_inputs({k:requests[calls[k]["hash"]] for k in replies})
    keys=sorted(replies);random.SystemRandom().shuffle(keys)
    mapping={f"P{i:02}":k for i,k in enumerate(keys,1)}
    blind={b:blind_item(k,replies[k]["say"]) for b,k in mapping.items()}
    write(out/"blind.json",blind);write(out/"key.json",mapping)
    print("Blind packet ready.",flush=True)
    for i in range(0,len(blind),6):
        batch={b:blind[b] for b in list(blind)[i:i+6]}
        send(f"judge.{i//6+1}",[{"role":"system","content":JUDGE},
              {"role":"user","content":json.dumps(batch,ensure_ascii=False)}])
    print(dict(Counter(v["status"] for v in calls.values())),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("dry","live"))
    ap.add_argument("--output",type=Path,default=HERE/"observed")
    args=ap.parse_args()
    if args.action=="dry":dry()
    else:live(args.output)
