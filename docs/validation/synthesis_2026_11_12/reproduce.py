"""Recompute the main paper's tables from the published ballots, choices and coded SAY."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
VALIDATION=HERE.parent
SOURCES={}


def read(path):
    p=VALIDATION/path
    data=p.read_bytes()
    SOURCES[path]=hashlib.sha256(data).hexdigest()
    return json.loads(data)


def role_votes():
    b="character_continuity_2026_11_11/"
    raw=read(b+"observed/ballots.json")
    key=read(b+"observed/registration.json")["codebook"]
    result={}
    for group,origin in (("human","human"),("ai_reference","ai_reference")):
        votes=[r for r in raw["ballots"] if r["origin"]==origin]
        per_scene={};diffs=[]
        for sid in ("S1","S2","S3"):
            per_scene[sid]={}
            for field in ("fit","continue_chat"):
                counts=Counter(key[sid].get(r["ratings"][sid][field],r["ratings"][sid][field]) for r in votes)
                per_scene[sid][field]=dict(counts)
            diffs.extend([r["id"],sid] for r in votes if r["ratings"][sid]["fit"]!=r["ratings"][sid]["continue_chat"])
        result[group]=dict(readers=len(votes),per_scene=per_scene,different_questions=diffs)
    return result


def slow_growth():
    b="trait_memory_2026_11_11/"
    z=read(b+"reference_growth/observed/states.json")
    key=read(b+"reference_growth/observed/key.json")
    coding=read(b+"reference_growth/observed/reading.json")
    values={key[i]:r["score"] for i,r in coding.items()}
    result={}
    for actor in ("tang","yu"):
        arms={arm:mean(v for k,v in values.items() if k.split(".")[:2]==[actor,arm]) for arm in ("G","F","S")}
        trajectory=z["trajectories"][actor]
        result[actor]=dict(initial=trajectory[0]["frozen"],final=trajectory[-1]["growing"],
            growth_minus_frozen=round(arms["G"]-arms["F"],6),resample_minus_frozen=round(arms["S"]-arms["F"],6))
    choices=read(b+"enacted_choice/handoff/choices.json")
    for actor in result:
        per_scene={}
        for sid in ("rest","sequence","proof","windows"):
            complete=[]
            for layout in range(4):
                k={arm:f"{actor}.{arm}.{sid}.{layout}" for arm in ("G","F","S")}
                if all(name in choices for name in k.values()):
                    group={arm:choices[name]["event"] for arm,name in k.items()}
                    if all(e["ticket"] is not None for e in group.values()):
                        complete.append({arm:int(e["target"]) for arm,e in group.items()})
            if complete:per_scene[sid]=dict(n=len(complete),rates={arm:mean(r[arm] for r in complete) for arm in ("G","F","S")})
        assert len(per_scene)==4
        result[actor]["action"]=dict(paired_layouts=sum(s["n"] for s in per_scene.values()),
            rates={arm:mean(s["rates"][arm] for s in per_scene.values()) for arm in ("G","F","S")},
            scenes_above_both_frozen=sum(s["rates"]["G"]>max(s["rates"]["F"],s["rates"]["S"]) for s in per_scene.values()))
    return result


def relationship():
    b="trait_memory_2026_11_11/"
    learning=read(b+"relief_learning/observed/results.json")
    gate=read(b+"arrival_memory_gate/observed/results.json")
    return dict(consistent_history={a:{m:z["paired"][m]["groups"]["old_near"]
                for m in ("R","J","S")} for a,z in learning["actors"].items()},
                gate={a:{position:{m:{"n":v["n"],"left":v["left"],"right":v["right"]}
                         for m,v in row.items() if m in ("V","R","Z")}
                         for position,row in z["cells"]["pooled"].items()} for a,z in gate["actors"].items()})


def speech():
    b="trait_memory_2026_11_11/arrival_speech_act/"
    gold=read(b+"observed/human.json")["labels"]
    pred=read(b+"observed/predictions.json")
    out={}
    for method in ("direct","scoped"):
        rows={i:pred[f"{i}.{method}.A"] for i in gold}
        out[method]=dict(n=len(gold),act_matches=sum(r["act"]==gold[i]["act"] for i,r in rows.items()),
            state_matches=sum(r["state"]==gold[i]["state"] for i,r in rows.items()),
            state_available=sum(r["state"] is not None for r in rows.values()),
            draft_matches=sum(r["draft"]==gold[i]["state"] for i,r in rows.items()),
            false_present=sum(r["state"]=="present" and gold[i]["state"]!="present" for i,r in rows.items()))
    return out


def viewpoint():
    b="shorekeeper_viewpoint_2026_11_12/"
    raw=read(b+"observed/turns.json")
    key=read(b+"observed/semantic_key.json")
    codes={c:{tuple(key[r["id"]][x] for x in ("variant","condition","replicate","act")):r
              for r in read(b+f"observed/coding_{c}.json")} for c in ("C1","C2")}
    result={}
    for method in ("scoped","full_notes"):
        for act in (1,2,3):
            rows=[r for r in raw if r["condition"]==method and r["act"]==act]
            result[f"{method}.{act}"]={c:dict(n=len(rows),
                insufficient=sum(coding[r["variant"],method,r["replicate"],act]["knowledge"]=="insufficient" for r in rows),
                shared=sum(coding[r["variant"],method,r["replicate"],act]["shared"]=="recognized" for r in rows),
                active=sum(coding[r["variant"],method,r["replicate"],act]["pact"]=="active" for r in rows))
                for c,coding in codes.items()}
    return result


def fixed_state():
    b="shorekeeper_viewpoint_2026_11_12/current_state/"
    rows={r["id"]:r for r in read(b+"observed/turns.json")}
    key=read(b+"observed/key.json")
    variants={v["id"]:v for v in read("shorekeeper_viewpoint_2026_11_12/scenes.json")["variants"]}
    out={}
    for coder in ("C1","C2"):
        coding={key[r["id"]]:r for r in read(b+f"observed/coding_{coder}.json")}
        stats={}
        flags={}
        for i,r in rows.items():
            z=coding[i];v=variants[r["variant"]]
            leak=any(t in r["say"] for t in (v["source"],*v["aliases"]))
            flags[i]=(z["knowledge"]=="claimed_known" and not leak and z["identity"] in ("none","hint_only")
                      if r["act"]==2 else z["shared"]=="recognized" and z["pact"]!="active")
        for act in (2,3):
            for method in ("control","state_card"):
                rr=[r for r in rows.values() if r["condition"]==method and r["act"]==act]
                stats[f"{method}.{act}"]=dict(n=len(rr),met=sum(flags[r["id"]] for r in rr),
                    by_sample={str(s):sum(flags[r["id"]] for r in rr if r["sample"]==s) for s in (1,2)})
        out[coder]=stats
    return out


def compute():
    out=dict(scope="After commit 245c20a; repeats and different questions are never pooled as independent readers.",
             human=role_votes(),growth=slow_growth(),relationship=relationship(),speech=speech(),
             viewpoint=viewpoint(),fixed_state=fixed_state(),
             supplemental=dict(attribution=read("trait_memory_2026_11_11/attribution/results.json")["gates"],
                 listener=read("trait_memory_2026_11_11/listener/first_case.json")["novel"],
                 reference_growth=read("trait_memory_2026_11_11/reference_growth/observed/results.json")["expand"],
                 initiative=read("initiative_exit_2026_11_11/results.json")))
    out["source_sha256"]=SOURCES
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--write",action="store_true");args=p.parse_args()
    data=compute();dest=HERE/"tables.json"
    if args.write:
        dest.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+"\n").encode())
    else:
        assert json.loads(dest.read_bytes())==data,"published tables differ"
    print("Main paper tables reproduced from",len(SOURCES),"source files.")
