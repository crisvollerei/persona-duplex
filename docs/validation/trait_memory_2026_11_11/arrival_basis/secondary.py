"""Keep development-set human comparisons separate from generated update controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def compare(rows,truth,people,method,with_basis=False):
    c=Counter();miss=[];bad_basis=[]
    for i,g in truth.items():
        r=rows[f"{i}.{method}.A"]["people"]
        matched=0
        for p in people:
            v=r[p]["state"];c["person_n"]+=1;matched+=int(v==g[p])
            if v!=g[p]:miss.append(dict(case=i,person=p,reference=g[p],observer=v))
            if with_basis and method=="typed" and g[p]!="unknown":
                c["explicit_state_n"]+=1
                c["explicit_state_correct"]+=int(v==g[p] and r[p]["basis"]=="explicit")
                if v!=g[p] or r[p]["basis"]!="explicit":bad_basis.append(dict(case=i,person=p,state=v,basis=r[p]["basis"]))
        c["people_correct"]+=matched;c["cases"]+=1;c["cases_correct"]+=matched==len(people)
    return dict(counts=dict(c),differences=miss,control_explicit_disagreements=bad_basis)
def summarize():
    d=load(HERE/"design.json");o=HERE/"observed"
    h1=load(HERE.parent/"arrival_shadow/observed/human.json")["labels"];cases=load(HERE/"cases.json")
    old=load(o/"calibration.json");control=load(o/"controls.json")
    constructed={c["id"]:dict(zip(d["people"],d["updates"][(int(c["id"])-1)%3]["states"])) for c in cases}
    ret={f:compare(old,h1,["俞川","常鸣"],f) for f in ("legacy","typed")}
    checks={f:compare(control,constructed,d["people"],f,True) for f in ("legacy","typed")}
    return dict(calibration_known_old_human=ret,controlled_latest_turn_not_human=checks)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--check",action="store_true");args=ap.parse_args()
    data=summarize();file=HERE/"observed/secondary.json"
    if args.check:assert load(file)==data
    else:file.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for k,methods in data.items():
        print(k,[(m,r["counts"]) for m,r in methods.items()])
