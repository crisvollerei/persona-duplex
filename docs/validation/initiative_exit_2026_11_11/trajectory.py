"""Replay authored exchanges through Session; freeze the input evidence, not model answers."""
import random

from study import HERE, Session, SessionConfig, Store, baseline, config, digest, load, trial, write
from persona.protocol import visible_dialogue


def build(path, *, register=False):
    path.mkdir(exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("fixture directory must be empty")
    design = load(HERE/"trajectory.json")
    ids = {}
    labels = {}
    snapshots = []
    history = []
    for block in design["sessions"]:
        store = Store(path)
        cfg = SessionConfig(memory_mode="frozen", generation_path="direct", consolidate_on_exit=False)
        reply = ""
        with Session(config(), store, lambda messages: reply, options=cfg) as s:
            assert visible_dialogue(s.engine.history) == history[-12:]
            for user, reply, *name in block:
                t = s.say(user)
                if t.say != reply:
                    raise ValueError("authored turn not delivered")
                history.extend([dict(role="user",content=user),dict(role="assistant",content=reply)])
                if name:
                    label = name[0]
                    extra = ({"memory_scope":"沙盘"} if label=="foreign" else
                             {"expires_at":"2002-01-01T00:00:00+00:00"} if label=="expired" else {})
                    previous = {"title-new":"title-old","style-new":"style-old"}.get(label)
                    record = store.add_judgment(user,1.,[t.turn_id],
                        supersedes=ids[previous] if previous else None,
                        evidence_quotes=[{"turn_id":t.turn_id,"quote":user,"evidence_scope":"specified"}], **extra)
                    if previous and record["supersedes"] != ids[previous]:
                        raise ValueError("supersession missing")
                    ids[label] = record["id"]
                    labels[record["id"]] = label
        snapshots.append(len(history)//2)
    store=Store(path)
    probes=[]
    jobs=[]
    for p in design["probes"]:
        q=p["user"]
        refs=store.recall_judgments(q)
        refs_labels=[labels[r["id"]] for r in refs]
        full,_,_=trial(q,"full",lambda _:"预备占位",path)
        strong,_,_=trial(q,"strong",lambda _:"预备占位",path)
        if strong!=baseline(config(),full) or strong[1:-1]!=history[-12:]:
            raise ValueError("visible histories or state differ")
        context=full[0]["content"].rsplit("\n\n[你和这个人之间]\n",1)[1]
        leaked=sorted(set(("title-old","style-old","expired","foreign")) & set(refs_labels))
        # Check the actual generation input as well as the selected IDs.
        leaked_text=[text for text in ("砂糖河滩","白塔唱针","晴窗广播","暂时别投票") if text in str(full)]
        expected=p["source"]
        sent = None if expected is None else any(r["judgment"] in context for r in refs if labels[r["id"]]==expected)
        probes.append(dict(case=p["id"], memory_ids=refs_labels, expected_source=expected,
            expected_source_in_prompt=sent, obsolete_or_external_ids=leaked, obsolete_or_external_text=leaked_text,
            visible_history_messages=len(strong[1:-1]), shared_context_hash=digest(context)))
        for route in ("full","strong"):
            for rep in (1,2):
                jobs.append(dict(id=f"{p['id']}-{route}-{rep}",case=p["id"],mode=p["mode"],
                                 route=route,rep=rep,user=q))
    stats=dict(authored_session_end_turns=snapshots,ledger_memory_versions=len(ids),probes=probes,
               observation="curated source-linked writes; frozen extraction; real Session restore and recall")
    if register:
        write(HERE/"long_preflight.json",stats)
    elif stats!=load(HERE/"long_preflight.json"):
        raise ValueError("trajectory reconstruction differs")
    random.Random(771024).shuffle(jobs)
    return jobs
