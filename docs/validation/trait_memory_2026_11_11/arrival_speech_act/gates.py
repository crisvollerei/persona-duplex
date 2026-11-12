"""Evaluate the registered thresholds and the same-call scoped draft/final ablation, offline."""
import argparse
import hashlib
import json
from pathlib import Path
from score import calculate, load, ratio, require

HERE=Path(__file__).resolve().parent


def ablate(ids, gold, predictions, replicate):
    def between(which):
        return {i: predictions[f"{i}.scoped.{replicate}"].get(which) for i in ids}
    draft,final=between("draft"),between("state")
    ref={i:gold[i]["state"] for i in ids}
    nonpresent=[i for i in ids if ref[i]!="present"]
    metrics={}
    for name,row in (("draft",draft),("final",final)):
        fp=[i for i in nonpresent if row[i]=="present"]
        metrics[name]={"correct":ratio(sum(row[i]==ref[i] for i in ids),len(ids)),
                       "valid":ratio(sum(row[i] is not None for i in ids),len(ids)),
                       "false_present":dict(**ratio(len(fp),len(nonpresent)),cases=fp)}
    fixed=[i for i in ids if draft[i]!=ref[i] and final[i]==ref[i]]
    lost=[i for i in ids if draft[i]==ref[i] and final[i]!=ref[i]]
    before,after=(set(metrics[k]["false_present"]["cases"]) for k in ("draft","final"))
    changes=[dict(case=i,human=ref[i],draft=draft[i],final=final[i]) for i in ids if draft[i]!=final[i]]
    return {"of":len(ids),**metrics,"fixed":fixed,"added_errors":lost,"net":len(fixed)-len(lost),
            "false_present_removed":sorted(before-after),"false_present_added":sorted(after-before),
            "changed":changes}


def judge(cases,preds,human,sha,design):
    require(design["states"]==["present","future","away","unknown"] and
            all(k in design for k in ("pass_acts","pass_state")), "unrecognized registration")
    r=calculate(cases,preds,human,sha)
    for key,row in preds.items():
        if key.split(".")[1]=="scoped":
            require("draft" in row and row["draft"] in (*design["states"], None),
                    f"missing/invalid scoped draft: {key}")
    out={"registered_wording":{k:design[k] for k in ("pass_acts","pass_state")},"reference":r["reference"],
         "annotated":r["annotated"],"planned":r["planned"],"eligible":False,"act_gate":None,
         "scoped_state_gate":None,"scoped_draft_vs_final":None}
    if human is None:return out
    ids=[c["id"] for c in cases if c["id"] in human["labels"] and human["labels"][c["id"]] is not None]
    gold=human["labels"]
    eligible=len(ids)==len(cases)==16 and all(sum(c["family"]==f for c in cases)==8 for f in ("handoff","sight"))
    out["eligible"]=eligible
    by_method=r["against_human"];out["act_gate"]={}
    for m in ("direct","scoped"):
        row=by_method[m+".A"]
        single={f:row[f]["correct"]["act"] for f in ("handoff","sight")}
        paired={f:row[f]["pairs"]["act"]["both_ends_correct"] for f in ("handoff","sight")}
        risky={k:v for k,v in row["all"]["risk"].items() if k!="false_present"}
        conditions={"at_least_7_of_8_in_each_family": all(z["n"]>=7 and z["of"]==8 for z in single.values()),
                    "at_least_3_of_4_exact_pairs_in_each_family": all(z["n"]>=3 and z["of"]==4 for z in paired.values()),
                    "zero_dangerous_swaps_min_3_opportunities_each": all(z["n"]==0 and z["of"]>=3 for z in risky.values())}
        out["act_gate"][m]={"pass":all(conditions.values()) if eligible else None,"criteria":conditions,
                            "family_act_correct":single,"family_pairs_both_correct":paired,"risk":risky}
    row=by_method["scoped.A"];rep=by_method["scoped.A2"]["all"]
    all_state=row["all"]["correct"]["state"];fp=row["all"]["risk"]["false_present"]
    risky_rep={k:v for k,v in rep["risk"].items() if k!="false_present"}
    per_family={f:row[f]["correct"]["state"] for f in ("handoff","sight")}
    delta=r["scoped_minus_direct"]["A"]["state"]
    conditions={"passed_act_gate":out["act_gate"]["scoped"]["pass"],
                "at_least_14_of_16_state_correct":all_state["n"]>=14 and all_state["of"]==16,
                "at_least_6_of_8_states_each":all(z["n"]>=6 and z["of"]==8 for z in per_family.values()),
                "zero_false_present_min_4_opportunities":fp["n"]==0 and fp["of"]>=4,
                "net_state_fixes_strictly_positive":delta["net"]>0,
                "no_dangerous_act_errors_on_repeat":all(z["n"]==0 for z in risky_rep.values())}
    out["scoped_state_gate"]={"pass":all(conditions.values()) if eligible else None,"criteria":conditions,
                              "overall_state_correct":all_state,"family_state_correct":per_family,"false_present":fp,
                              "state_delta_vs_direct":delta,"repeat_act_risk":risky_rep,
                              "repeat_act_valid":rep["coverage"]["act"]}
    group=lambda f:[c["id"] for c in cases if c["id"] in ids and (not f or c["family"]==f)]
    out["scoped_draft_vs_final"]={run:{f or "all":ablate(group(f),gold,preds,run)
                                       for f in ("","handoff","sight")} for run in ("A","A2")}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human",type=Path,help="Defaults to observed/human.json if present.")
    p.add_argument("--predictions",type=Path,default=HERE/"observed/predictions.json")
    p.add_argument("--cases",type=Path,default=HERE/"cases.json")
    p.add_argument("--design",type=Path,default=HERE/"design.json")
    p.add_argument("--out",type=Path,default=HERE/"observed/gates.json")
    p.add_argument("--check",action="store_true")
    a=p.parse_args();h=a.human if a.human is not None else HERE/"observed/human.json"
    human=load(h) if a.human is not None or h.exists() else None
    z=judge(load(a.cases),load(a.predictions),human,hashlib.sha256(a.cases.read_bytes()).hexdigest(),load(a.design))
    if a.check:require(load(a.out)==z,"stored gates differ from recomputation")
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.out.with_suffix(".tmp");tmp.write_bytes((json.dumps(z,ensure_ascii=False,indent=2)+"\n").encode());tmp.replace(a.out)
    print("act gates:",{k:v["pass"] for k,v in z["act_gate"].items()} if z["act_gate"] else None,
          "scoped state gate:",(z["scoped_state_gate"] or {}).get("pass"))


if __name__=="__main__":main()
