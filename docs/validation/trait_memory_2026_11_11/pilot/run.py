"""A staged live pilot. Only explicitly invoked; exported requests do not include HTTP headers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[3])]
from scene import OTHER, actor_config
from persona.config import Trait
from persona.consolidate import Consolidator
from persona.llm import openai_compatible
from persona.session import Session, SessionConfig
from persona.store import Store, replay_traits

def read(p):
    return json.loads(p.read_bytes())

def save(p, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())
    tmp.replace(p)

D = read(HERE / "design.json")
W = read(HERE.parent / "world.json")
SC = {s["id"]: s for s in W["scenes"]}
TRAIT = D["trait"]
BASE = {k: v["traits"][TRAIT] for k,v in W["roles"].items()}
LOCK = threading.Lock()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path, help="empty output directory for this run")
    ap.add_argument("--feedback", type=Path, required=True, help="directory for two operator-authored feedback files")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        ap.error("output directory must be empty")
    key = os.environ.get("PERSONA_API_KEY") or getpass.getpass("DeepSeek test key: ")
    if not key:
        ap.error("no key supplied")
    records, turns, learned, histories = {}, {}, {}, {}
    save(out/"design_snapshot.json", {"design":digest(D), "world":digest(W),
         "runner":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def call(k, messages):
        with LOCK:
            if k in records:
                raise RuntimeError("repeated call id")
            records[k] = {"messages":messages, "request_hash":digest(messages), "status":"started"}
            save(out/"calls.json", records)
        client = openai_compatible(**D["provider"], base_url="https://api.deepseek.com/v1", api_key=key, timeout=65)
        try:
            raw = client(messages)
        except Exception as exc:
            import re
            http = re.search(r"HTTP (\d{3})", str(exc))
            with LOCK:
                records[k].update(status="failed", error_type=type(exc).__name__,
                                  http_status=http.group(1) if http else None)
                save(out/"calls.json", records)
            raise RuntimeError(f"{k} failed, see sanitized request status") from None
        with LOCK:
            records[k].update(status="ok", raw=raw, response=client.observations.response)
            save(out/"calls.json", records)
        return raw

    def turn(k, cfg, store, who, user, *, memory=None, history=()):
        opts = SessionConfig(memory_mode="frozen", consolidate_on_exit=False, memory_scope=who)
        with Session(cfg, store, lambda msgs: call(k,msgs), options=opts) as ss:
            if memory is not None:
                ss.engine.memory_snapshot = deepcopy(memory)
            if history:
                ss.engine.history = deepcopy(list(history))
            r = ss.say(user)
        with LOCK:
            turns[k] = {"turn_id":r.turn_id, "user":user, "say":r.say, "validity":r.validity}
            save(out/"turns.json", turns)
        return r

    def cfg0(actor, sid="01"):
        cfg = actor_config(W, OTHER[actor], SC[sid])
        cfg.role.traits[TRAIT] = Trait(anchor=BASE[actor], plasticity=D["plasticity"], drift=D["drift"])
        return cfg

    def later_cfg(actor, setting, value, familiar=True):
        cfg = cfg0(actor)
        me, other = W["roles"][actor], W["roles"][OTHER[actor]]
        partner = other["name"] if familiar else "陈岑"
        cfg.role.identity = (f'虚构对话。你写{me["name"]}的言行，玩家决定{partner}的言行。'
                             f'\n本人：{me["public"]}\n个人背景：{me["private"]}\n当下：{setting}')
        cfg.role.traits[TRAIT].value = value
        return cfg

    with TemporaryDirectory() as work, ThreadPoolExecutor(max_workers=2) as pool:
        stores = {a:Store(Path(work)/a) for a in OTHER}
        eligible = {a:[] for a in OTHER}
        for i in range(2):
            def cue(a):
                tr = D["training"][a][i]
                return turn(f"{a}.{i}.cue",cfg0(a,tr["scene"]),stores[a],a,tr["cue"])
            list(pool.map(cue, OTHER))
            print(f"CUES {i+1} READY",flush=True)
            feedback_path = args.feedback / f"feedback_{i+1}.json"
            deadline = time.monotonic() + 1200
            while not feedback_path.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("missing operator feedback")
                time.sleep(1)
            feedback = read(feedback_path)
            save(out/feedback_path.name, feedback)
            def respond(a):
                tr = D["training"][a][i]
                c = turns[f"{a}.{i}.cue"]
                f = feedback[a]
                if f["quote"] and f["quote"] not in c["say"]:
                    raise ValueError("unmatched quote for " + a)
                u = f["reply"] + tr["receipt"]
                r = turn(f"{a}.{i}.reply", cfg0(a,tr["scene"]),stores[a],a,u)
                if f["eligible"] and f["quote"] and c["validity"]["valid"] and r.validity["valid"]:
                    eligible[a].append((c["turn_id"],r.turn_id,f["quote"],f["reply"]))
                n = 0
                def mem(prompt):
                    nonlocal n
                    n += 1
                    return call(f"{a}.{i}.memory{n}",[{"role":"user","content":prompt}])
                result = Consolidator(stores[a], llm=mem, memory_scope=a).run_once()
                save(out/f"{a}_memory_{i+1}.json", {"result":result,"active":stores[a].user_judgments(None)})
                histories[a] = [{"role":"user","content":u},{"role":"assistant","content":r.say}]
            list(pool.map(respond, OTHER))
        snapshots = {a:stores[a].user_judgments(None) for a in OTHER}
        for a in OTHER:
            st = stores[a]
            for cue_id, response_id, quote, response in eligible[a]:
                st.add_judgment("该次互动中，自己的具体需要得到了对方具体回应。",D["weight"],
                                [cue_id,response_id], affects={TRAIT:1}, memory_scope=a, source="pilot_label",
                                evidence_quotes=[{"turn_id":response_id,"quote":response,"evidence_scope":"specified"}])
            changes = Consolidator(st).push_traits()
            value = st.traits()[TRAIT]["value"]
            assert abs(replay_traits(st.read_ledger())[TRAIT]["value"] - value) < 1e-5
            assert not Consolidator(st).push_traits()
            learned[a] = {"eligible":eligible[a],"before":BASE[a],"after":value,"changes":changes}
        save(out/"learned.json", learned)

        def probe(task):
            k,a,setting,user,val,mem,hist = task
            turn(k,later_cfg(a,setting,val),Store(Path(work)/"probes"/k),a,user,memory=mem,history=hist)
        tasks = []
        for a in OTHER:
            for arm in ("empty","memory"):
                tasks.append((f"{a}.fact.{arm}",a,"冬天，在柜台前遇到一起整理过照片的旧相识。",
                              D["fact_probe"],BASE[a],snapshots[a] if arm=="memory" else [],[]))
            for p in D["probes"]:
                for rep in range(D["repeats"]):
                    arms = [("fixed",BASE[a]),("changed",learned[a]["after"])]
                    if arms[0][1] == arms[1][1]:
                        arms = arms[:1]
                    for arm,val in arms:
                        tasks.append((f"{a}.{p['id']}.{rep}.{arm}",a,p["setting"],p["user"],val,
                                      snapshots[a],histories[a]))
        random.Random(83161).shuffle(tasks)
        list(pool.map(probe,tasks))
        for a in OTHER:
            from persona.prompt import build_system_prompt
            baseline = build_system_prompt(later_cfg(a,D["stranger"],BASE[a],False))
            # The research router stores per-pair values; no value for this new person.
            rel_values = {W["roles"][OTHER[a]]["name"]:learned[a]["after"]}
            assert build_system_prompt(later_cfg(a,D["stranger"],rel_values.get("陈岑",BASE[a]),False)) == baseline
        blind = [{"id":k,"actor":W["roles"][k.split(".")[0]]["name"],"scene":k.split(".")[1],
                  "say":r["say"]} for k,r in turns.items() if k.split(".")[1] in ("lamp","byline")]
        random.Random(31913).shuffle(blind)
        mapping = {f"B{i:02}":x.pop("id") for i,x in enumerate(blind,1)}
        save(out/"blind.json", {j:x for j,x in zip(mapping,blind)})
        save(out/"key.json", mapping)
        print(f"DONE {len(records)} calls",flush=True)

if __name__ == "__main__":
    main()
