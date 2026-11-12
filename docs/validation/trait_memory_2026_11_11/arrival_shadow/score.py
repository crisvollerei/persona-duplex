"""Compare independent human labels with model outputs. No labels are inferred here."""
import argparse
from collections import Counter
import json
from pathlib import Path
from observer import group,PEOPLE,STATES
HERE=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def dump(p,v):
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+"\n").encode());tmp.replace(p)
def evaluate(human,rows,design,cases):
    labels=human["labels"];ids=[c["id"] for c in cases]
    assert human["source"]=="human_self_report" and set(labels)<=set(ids)
    assert all(v is None or (isinstance(v,dict) and set(v)==set(PEOPLE) and all(x in STATES for x in v.values())) for v in labels.values())
    valid=[i for i in ids if labels.get(i) is not None]
    if not valid:raise ValueError("No human annotations to compare")
    methods={}
    for mode in design["forms"]:
        for order in design["orders"]:
            runs={i:{p:r["state"] for p,r in rows[f"{i}.{mode}.{order}"]["people"].items()} for i in valid}
            def tally(subset):
                c=Counter()
                for i in subset:
                    gold=labels[i];pred=runs[i];hg=group(gold.values());pg=group(pred.values())
                    c["cases"]+=1;c["both_people_correct"]+=pred==gold;c["delivered"]+=pg!="invalid"
                    c["person_correct"]+=sum(pred[p]==gold[p] for p in PEOPLE)
                    c["person_invalid"]+=sum(pred[p] is None for p in PEOPLE)
                    c["coarse_correct"]+=("unknown" if hg=="mixed" else hg)==("unknown" if pg=="mixed" else pg)
                    c["fine_group_correct"]+=hg==pg
                    c["not_both_present"]+=hg!="present";c["false_both_present"]+=hg!="present" and pg=="present"
                return dict(c)
            confusion={h:{s:0 for s in (*STATES,"invalid")} for h in STATES}
            diffs=[]
            for i in valid:
                for p in PEOPLE:
                    confusion[labels[i][p]][runs[i][p] or "invalid"]+=1
                    if labels[i][p]!=runs[i][p]:diffs.append(dict(case=i,person=p,human=labels[i][p],observer=runs[i][p]))
            risks={}
            for name,pairs in design["targeted_persons"].items():
                checked=[(i,p) for i,p in pairs if i in valid and labels[i][p]!="present"]
                risks[name]=dict(n=len(checked),false_present=sum(runs[i][p]=="present" for i,p in checked))
            methods[f"{mode}.{order}"]=dict(overall=tally(valid),confusion=confusion,disagreements=diffs,risks=risks,
                topics={k:tally([i for i in v["items"] if i in valid]) for k,v in design["topics"].items()})
    primary=methods["evidence.YC"];rev=methods["evidence.CY"]
    stable={f:sum(all(rows[f"{i}.{f}.YC"]["people"][p]["state"]==rows[f"{i}.{f}.CY"]["people"][p]["state"] for p in PEOPLE) for i in valid) for f in design["forms"]}
    gate=len(valid)==len(ids) and primary["overall"]["delivered"]==16 and primary["overall"]["both_people_correct"]>=14 and all(
        r["overall"]["false_both_present"]==0 and all(item["false_present"]==0 for item in r["risks"].values()) for r in (primary,rev))
    return dict(annotated=len(valid),missing_or_skipped=[i for i in ids if i not in valid],methods=methods,order_agreement=stable,expand_shadow_only=gate)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--human",type=Path,default=HERE/"observed/human.json")
    ap.add_argument("--out",type=Path,default=HERE/"observed/results.json");ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    data=evaluate(load(args.human),load(HERE/"observed/predictions.json"),load(HERE/"design.json"),load(HERE/"cases.json"))
    if args.check:assert data==load(args.out)
    else:dump(args.out,data)
    print("Human cases:",data["annotated"],"expand:",data["expand_shadow_only"])
