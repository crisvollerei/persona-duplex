"""Evaluate the registered continuation rule; describe act 3 by the archived prior reply."""
import argparse
import hashlib
import json
from pathlib import Path
from semantic import features

HERE=Path(__file__).resolve().parent
OUT=HERE/"observed"; PARENT=HERE.parent
def read(p):return json.loads(p.read_bytes())

def compute():
    key=read(OUT/"key.json"); rows={t["id"]:t for t in read(OUT/"turns.json")}
    stats=read(OUT/"semantic_summary.json")["by_coder"]
    variants={v["id"]:v for v in read(PARENT/"scenes.json")["variants"]}
    previous=PARENT/"observed"
    old_key=read(previous/"semantic_key.json")
    prev={}
    for label in ("C1","C2"):
        for x in read(previous/f"coding_{label}.json"):
            r=old_key[x["id"]]
            if r["act"]==2 and r["condition"]=="scoped":
                prev.setdefault((r["variant"],r["replicate"]),set()).add(x["knowledge"])
    if len(prev)!=6 or any(len(z)!=1 for z in prev.values()):
        raise ValueError("Parent second-act labels missing or disputed; do not stratify")
    out={"rule_source":"protocol.json:decision","previous_act2_strata":{
        v+"."+h:next(iter(zz)) for (v,h),zz in sorted(prev.items())},"by_coder":{}}
    for coder, summary in stats.items():
        coding={key[z["id"]]:z for z in read(OUT/f"coding_{coder}.json")}
        nets={k:v["net"] for k,v in summary["card_vs_control_same_sample"].items() if not k.endswith("all")}
        new_hazards=[]; persistent_hazards=[]; removed_hazards=[]; missing=[]
        strata={}
        for ident,r in rows.items():
            if r["condition"]!="state_card":continue
            c=ident.replace(".state_card.",".control.")
            if ident not in coding or c not in coding:missing.append(ident);continue
            def flags(k):
                f=features(rows[k],coding[k],variants[r["variant"]])
                return ((f["literal_own_name"] or f["confirmed_identity"] or f["conflicting_identity"])
                        if r["act"]==2 else f["pact_active"]), f["primary"]
            bad_new, pos_new=flags(ident);bad_old,pos_old=flags(c)
            if bad_new and not bad_old:new_hazards.append(ident)
            if bad_new and bad_old:persistent_hazards.append(ident)
            if bad_old and not bad_new:removed_hazards.append(ident)
            if r["act"]==3:
                label=next(iter(prev[r["variant"],r["history"]]))
                cell=strata.setdefault(label,dict(paired_samples=0,control_met=0,state_card_met=0))
                cell["paired_samples"]+=1;cell["control_met"]+=int(pos_old);cell["state_card_met"]+=int(pos_new)
        out["by_coder"][coder]=dict(net_by_act_and_sample=nets,new_hazard_slots=new_hazards,
            persistent_hazard_slots=persistent_hazards,removed_hazard_slots=removed_hazards,missing=missing,
            positive_both_acts_both_samples=all(v>0 for v in nets.values()),
            passes=not missing and not new_hazards and all(v>0 for v in nets.values()),
            exploratory_act3_by_frozen_act2_knowledge=strata)
    out["both_acts_reproduced"]=all(v["passes"] for v in out["by_coder"].values())
    out["source_hashes"]={p.relative_to(PARENT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [HERE/"protocol.json",OUT/"semantic_summary.json",previous/"semantic_key.json",
                  *[previous/f"coding_{c}.json" for c in ("C1","C2")]]}
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--check",action="store_true");args=p.parse_args()
    r=compute();dest=OUT/"conclusions.json"
    if args.check or dest.exists():
        if read(dest)!=r:raise ValueError("Stored conclusion differs from recomputation")
    else:dest.write_bytes((json.dumps(r,ensure_ascii=False,indent=2)+"\n").encode())
    print("Both-act criterion:",r["both_acts_reproduced"])
