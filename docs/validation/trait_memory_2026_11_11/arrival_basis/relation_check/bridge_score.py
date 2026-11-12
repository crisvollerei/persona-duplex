"""Post-hoc bridge to an existing human ballot; only A is compared, never independent confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import score as base


def relation_predictions(raw,ids):
    allowed=[{i:i+suffix for i in ids} for suffix in (".relation.A",".relation","")]
    names=next((mapping for mapping in allowed if set(mapping.values())==set(raw)),None)
    base.require(names is not None,"missing / extra / mixed-format bridge positions")
    rows={}
    for i,k in names.items():
        row=raw[k]["people"];base.require(set(row)==set(base.PEOPLE),"wrong participants")
        rows[i]={}
        for p in base.PEOPLE:
            state,basis=base.state_basis(row[p])
            base.require(state is None or state in base.STATES,"invalid state")
            base.require(basis is None or basis in base.BASES,"invalid basis")
            rows[i][p]={"state":state,"basis":basis}
    return rows


def transitions(gold,old,new,ids,fields):
    """Invalid/missing predictions stay wrong; no listwise deletion."""
    people=[];cases=[];gains=[];losses=[]
    for i in ids:
        flags=[]
        for p in base.PEOPLE:
            encode=lambda r: tuple(r[i][p][f] for f in fields)
            g,o,n=encode(gold),encode(old),encode(new)
            before,after=o==g,n==g;flags.append((before,after))
            if before!=after:
                (gains if after else losses).append({"case":i,"person":p,"human":dict(zip(fields,g)),
                                                    "previous":dict(zip(fields,o)),"relation":dict(zip(fields,n))})
        people.extend(flags);cases.append((all(x for x,y in flags),all(y for x,y in flags)))
    def count(v):
        gain=sum(not x and y for x,y in v);loss=sum(x and not y for x,y in v)
        return {"gain":gain,"loss":loss,"net":gain-loss,"both_right":sum(x and y for x,y in v),
                "both_wrong":sum(not x and not y for x,y in v),"n":len(v)}
    return {"people":count(people),"cases":count(cases),"corrections":gains,"new_errors":losses}


def compare(cases,sha,ballot,old_raw,new_raw):
    ids=[c["id"] for c in cases];base.require(len(ids)==len(set(ids)),"duplicate corpus ID")
    gold=base.check_human(ballot,ids,sha);active=[i for i in ids if i in gold]
    orig=base.read_predictions(old_raw,ids)
    names={"legacy":orig["legacy.A"],"typed":orig["typed.A"],"relation":relation_predictions(new_raw,ids)}
    methods={name:base.against(cases,gold,rows,name!="legacy") for name,rows in names.items()}
    changes={}
    for origin in ("legacy","typed"):
        dims={"state":("state",)}
        if origin=="typed":dims.update(basis=("basis",),joint=("state","basis"))
        changes["relation_vs_"+origin]={k:transitions(gold,names[origin],names["relation"],active,fields) for k,fields in dims.items()}
        assert base.state_delta(gold,names[origin],names["relation"],active)=={k:changes["relation_vs_"+origin]["state"][k] for k in ("people","cases")}
    return {"scope":"post_hoc_single_reviewer_existing_cases","corpus_sha256":sha,"reviewer":ballot["reviewer"],
            "planned_cases":len(ids),"annotated_cases":len(active),"skipped":[i for i in ids if i not in gold],
            "methods":methods,"paired_changes":changes,"independent_confirmation":False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human",type=Path,default=ROOT/"observed/human_H01.json")
    ap.add_argument("--cases",type=Path,default=ROOT/"cases.json")
    ap.add_argument("--primary",type=Path,default=ROOT/"observed/predictions.json")
    ap.add_argument("--relation",type=Path,default=HERE/"bridge_observed/predictions.json")
    ap.add_argument("--out",type=Path,default=HERE/"bridge_observed/comparison.json")
    ap.add_argument("--check",action="store_true")
    a=ap.parse_args()
    r=compare(base.load(a.cases),hashlib.sha256(a.cases.read_bytes()).hexdigest(),base.load(a.human),
              base.load(a.primary),base.load(a.relation))
    if a.check:base.require(base.load(a.out)==r,"stored comparison is not reproducible")
    else:
        tmp=a.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(a.out)
    print(r["scope"],r["annotated_cases"],"cases")

if __name__=="__main__":main()
