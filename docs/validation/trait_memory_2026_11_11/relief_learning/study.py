"""Frozen actor, scoped ledger note and an enacted, externally scored assignment."""
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

HERE = Path(__file__).resolve().parent
from analyze import summarize
sys.path.insert(0, str(HERE.parent / "enacted_choice"))
import handoff
t = handoff.t
b = t.base
D, M = (b.read(HERE / f) for f in ("design.json", "materials.json"))
O = HERE / "observed"
LABEL = "\n由上面三张交接条形成的关系判断："
HISTORY = {a: b.context(t.prior.attr.hist, a) for a in D["roles"]}


def jobs():
    blocks = list(itertools.product(D["roles"], D["candidates"], M["tasks"], D["layouts"]))
    rng = random.Random(D["seed"])
    rng.shuffle(blocks)
    result = []
    for a, p, q, n in blocks:
        arms = list(itertools.product(D["outcomes"], D["memory"]))
        rng.shuffle(arms)
        result.extend((a, h, m, p, q, n) for h, m in arms)
    return result


def key(job):
    return ".".join(map(str, job))


def evidence(a, h, with_note, root):
    partner = b.NAME[b.OTHER[a]]
    scope = f"{a}/{b.OTHER[a]}/relief-arrival"
    store = b.Store(root)
    rows = []
    for i, (place, task, start, end) in enumerate(M["events"], 1):
        kw = dict(person=partner, actor=b.NAME[a], place=place, job=task, start=start, end=end)
        row = dict(turn_id=f"{a}-{i}", user=M["promise"].format(**kw), say=M[h].format(**kw))
        store.append("turn", row)
        rows.append(row)
    if with_note:
        store.add_judgment(M["judgments"][h].format(person=partner), 1.0,
                           [x["turn_id"] for x in rows], source="explicit", affects={},
                           memory_scope=scope, evidence_quotes=[
                               dict(turn_id=x["turn_id"], quote=x["say"], evidence_scope="specified") for x in rows])
    saved = b.Store(root)
    turns = [{k: r[k] for k in ("turn_id", "user", "say")} for r in saved.read_ledger() if r["kind"]=="turn"]
    notes = saved.recall_judgments("短时接班按约到岗", scope=scope)
    assert turns == rows and len(notes) == int(with_note)
    assert not saved.recall_judgments("接班", scope=f"{a}/池岚/relief-arrival")
    note = None
    if notes:
        n = notes[0]
        note = {k:n[k] for k in ("judgment","evidence_ids","evidence_quotes","memory_scope","affects")}
        assert note["evidence_ids"] == [x["turn_id"] for x in turns] and not note["affects"]
        assert [q["quote"] for q in note["evidence_quotes"]] == [x["say"] for x in turns]
    return dict(turns=turns, note=note)


def scene(a, person, task):
    who = b.NAME[b.OTHER[a]] if person=="old" else M["new_person"]
    reserve = M["reserve"]
    spec = M["tasks"][task]
    background = M["crop_common" if task=="crop" else "near_common"]
    # Names only appear in the counterbalanced menu; the reserve's reference record stays fixed.
    return dict(field=spec["job"], family="assignment", public=spec["public"]+background,
                target=[who, "由"+who+spec["job"]], other=[reserve, "由"+reserve+spec["job"]],
                line="这项交给谁，由你来签。登记后我照签排人。")


def request(a, record, q, layout, send):
    opts, goal = t.menu(q, a, layout)
    turns = "\n".join(f"{i}. {x['user']}{x['say']}" for i,x in enumerate(record["turns"],1))
    note = LABEL+record["note"]["judgment"] if record["note"] else ""
    ctx = HISTORY[a]+"\n巡回结束后，你收好的三张临时交接条及亲历：\n"+turns+note
    cfg = b.cfg(a,b.BASE[a],t.render(q,opts),"这张岗位单由你本人签。",ctx)
    old = "对面是"+b.NAME[b.OTHER[a]]+"。"
    assert cfg.role.identity.count(old)==1
    cfg.role.identity = cfg.role.identity.replace(old,"现在递单给你的是站方登记员"+M["clerk"]+"，登记员不参与候选。",1)
    traits = cfg.role.traits_state()
    caught = []
    def capture(m):
        raw = send(m)
        caught.append((deepcopy(m), raw))
        return raw
    with TemporaryDirectory() as tmp:
        store = b.Store(tmp)
        with b.Session(cfg,store,capture,options=b.SessionConfig(memory_mode="frozen",consolidate_on_exit=False,memory_scope=a)) as session:
            before = store.traits()
            ans = session.say(handoff.userline(q,opts))
        assert before == store.traits() == traits and cfg.role.traits_state()==traits
    assert len(caught)==1
    messages, raw = caught[0]
    normalized = deepcopy(messages)
    assert sum(x["content"].count(LABEL.strip()) for x in messages)==int(bool(note))
    if note:
        encoded = json.dumps(note,ensure_ascii=False)[1:-1]
        assert normalized[0]["content"].count(encoded)==1
        normalized[0]["content"] = normalized[0]["content"].replace(encoded,"")
    return dict(menu=opts, candidate=goal, say=ans.say, validity=ans.validity,
                event=t.transition(q,opts,goal,ans.say,raw), request=b.digest(messages),
                without_note=b.digest(normalized), traits=b.digest(traits))


def registration():
    deps = {p for p in (b.ROOT/"persona").glob("*.py")}
    deps.update([HERE/f for f in ("design.json","materials.json","study.py","analyze.py")])
    deps.update([b.HERE/"observed/trajectory.json",b.HERE/"design.json",b.HERE/"story.json",
                 b.HERE/"run.py",HERE.parent/"pilot/design.json",HERE.parent/"world.json",
                 HERE.parent/"scene.py",t.HERE/"trial.py",t.HERE/"handoff.py",b.ROOT/"personas/default.toml"])
    return dict(order=list(map(key,jobs())),foreground=D["foreground"],inputs={
        p.relative_to(b.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(deps)})


def run(mode):
    reg = registration()
    def save(name,obj):
        fp = O/name
        if mode=="replay" or (mode=="live" and fp.exists()):
            assert b.read(fp)==obj, name
        elif mode=="live":
            b.write(fp,obj)
    cache = {}
    log = O/"calls.jsonl.gz"
    if mode!="dry":
        if mode=="live": O.mkdir(exist_ok=True)
        if not (O/"registration.json").exists(): assert not any(O.iterdir())
        save("registration.json",reg)
        if log.exists():
            entries = b.lines(log)
            cache = {r["id"]:r for r in entries}
            assert len(entries)==len(cache)
    records = {}
    with TemporaryDirectory() as tmp:
        for a,h,m in itertools.product(D["roles"],D["outcomes"],D["memory"]):
            records[key((a,h,m))] = evidence(a,h,m=="J",Path(tmp)/key((a,h,m)))
            if m!="R":
                assert records[key((a,h,m))]["turns"]==records[key((a,h,"R"))]["turns"]
    save("evidence.json",records)
    traits = {a:b.cfg(a,b.BASE[a],"","","").role.traits_state() for a in D["roles"]}
    save("traits.json",traits)
    secret = (os.getenv("PERSONA_API_KEY") or getpass.getpass("Provider credential: ")) if mode=="live" and len(cache)<len(jobs()) else None
    if mode=="live" and len(cache)<len(jobs()) and not secret: raise ValueError("Missing credential.")
    lock = Lock()
    def sample(j):
        a,h,m,p,q,n = j
        k = key(j)
        def send(messages):
            if k in cache:
                assert cache[k]["messages"]==messages
                return cache[k]["raw"]
            if mode=="dry":
                return b.make_reply("need: 落签\nconfidence: 0.5\ndirect: 可以。\nbaseline: 可以。\ndisplace: none","落签：青签")
            if mode!="live": raise ValueError("Missing "+k)
            api = b.openai_compatible(**D["foreground"],base_url="https://api.deepseek.com/v1",api_key=secret,timeout=90)
            try:
                raw = api(messages)
                rec = dict(ok=True,raw=raw,response=api.observations.response)
            except Exception as e:
                if not cache or any(x in str(e) for x in ("HTTP 401","HTTP 403","HTTP 402")):
                    raise RuntimeError("Provider unavailable: "+type(e).__name__) from None
                rec = dict(ok=False,raw="",error=type(e).__name__,response=api.observations.response)
            rec.update(id=k,messages=messages,params=D["foreground"])
            with lock:
                with gzip.open(log,"ab") as f: f.write(b.wire(rec)+b"\n")
                cache[k]=rec
                if len(cache)%48==0: print(len(cache),"archived",flush=True)
            return rec["raw"]
        row = request(a,records[key((a,h,m))],scene(a,p,q),n,send)
        assert row["traits"]==b.digest(traits[a])
        return k,row
    plan=jobs()
    first=sample(plan[0])
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=dict([first,*pool.map(sample,plan[1:])])
    assert len(rows)==len(set(reg["order"]))==288
    for a,h,_,p,q,n in plan:
        r,j,s=[rows[key((a,h,m,p,q,n))] for m in D["memory"]]
        assert r["request"]==s["request"]==j["without_note"]!=j["request"]
    if mode=="dry": assert all(r["event"]["ticket"]=="青签" for r in rows.values())
    else: assert set(cache)==set(rows)
    save("choices.json",rows)
    save("results.json",summarize(rows))
    print(mode,"288 actions checked",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("dry","live","replay"))
    run(ap.parse_args().mode)
