"""Score human labels and repeat stability separately; no provider or inferred reference labels."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTS = ("promise", "on_duty", "not_seen", "seen_leave", "unclear")
STATES = ("present", "future", "away", "unknown")
METHODS = ("direct", "scoped")
RUNS = ("A", "A2")
FIELDS = {"act": ("act",), "state": ("state",), "joint": ("act", "state")}


def ratio(n, denominator):
    return {"n": n, "of": denominator, "rate": n / denominator if denominator else None}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_bytes())


def observed(row, fields):
    return all(row.get(k) is not None for k in fields)


def vector(row, fields):
    return tuple(row.get(k) for k in fields)


def agrees(a, b, fields):
    return observed(a, fields) and observed(b, fields) and vector(a, fields) == vector(b, fields)


def comparison(ids, left, right):
    out = {}
    for field, cols in FIELDS.items():
        valid = [i for i in ids if observed(left[i], cols) and observed(right[i], cols)]
        same = [i for i in valid if agrees(left[i], right[i], cols)]
        out[field] = {"equal_among_valid":ratio(len(same), len(valid)), "valid_pairs":ratio(len(valid),len(ids)),
                      "different": [i for i in valid if i not in same], "missing": [i for i in ids if i not in valid]}
    return out


def pairs_score(pairs, gold, model):
    complete = {k: ii for k,ii in pairs.items() if all(i in gold for i in ii)}
    out = {}
    for field, cols in FIELDS.items():
        details = {}
        for pair, (i,j) in complete.items():
            truth_change = vector(gold[i],cols) != vector(gold[j],cols)
            valid = observed(model[i], cols) and observed(model[j], cols)
            predicted_change = (vector(model[i],cols) != vector(model[j],cols)) if valid else None
            details[pair] = {"cases":[i,j], "human_changed":truth_change, "observer_changed":predicted_change,
                             "both_ends_correct":agrees(gold[i],model[i],cols) and agrees(gold[j],model[j],cols),
                             "change_agrees":valid and truth_change == predicted_change}
        out[field] = {"both_ends_correct":ratio(sum(z["both_ends_correct"] for z in details.values()),len(complete)),
                      "changed_or_unchanged_agrees":ratio(sum(z["change_agrees"] for z in details.values()),len(complete)),
                      "valid_pair_count":sum(z["observer_changed"] is not None for z in details.values()),
                      "per_pair":details}
    return out


def against(ids, pairs, gold, model):
    correct = {k:ratio(sum(agrees(gold[i],model[i],f) for i in ids),len(ids)) for k,f in FIELDS.items()}
    known = {k:ratio(sum(observed(model[i],f) for i in ids),len(ids)) for k,f in FIELDS.items()}
    confusion = {}
    for k,vs in (("act",ACTS),("state",STATES)):
        mat = {h:{p:0 for p in (*vs,"invalid")} for h in vs}
        for i in ids:
            mat[gold[i][k]][model[i][k] or "invalid"] += 1
        confusion[k] = mat
    risk = {}
    for human,pred in (("promise","on_duty"),("not_seen","seen_leave")):
        opportunities = [i for i in ids if gold[i]["act"]==human]
        errors = [i for i in opportunities if model[i]["act"]==pred]
        risk[human+"_as_"+pred] = dict(**ratio(len(errors),len(opportunities)), cases=errors)
    opportunities = [i for i in ids if gold[i]["state"]!="present"]
    errors = [i for i in opportunities if model[i]["state"]=="present"]
    risk["false_present"] = dict(**ratio(len(errors),len(opportunities)),cases=errors)
    diffs = [{"case":i, "human":gold[i], "observer":model[i],
              "act_matches":agrees(gold[i],model[i],FIELDS["act"]),
              "state_matches":agrees(gold[i],model[i],FIELDS["state"])}
             for i in ids if not agrees(gold[i],model[i],FIELDS["joint"])]
    return {"items":len(ids),"correct":correct,"coverage":known,"risk":risk,"confusion":confusion,
            "pairs":pairs_score(pairs,gold,model),"errors":diffs}


def delta(ids, gold, base, new):
    result = {}
    for name, fs in FIELDS.items():
        fixed, lost, both_correct, both_wrong = [],[],[],[]
        for i in ids:
            l,r = agrees(gold[i],base[i],fs),agrees(gold[i],new[i],fs)
            (both_correct if l and r else fixed if r else lost if l else both_wrong).append(i)
        result[name] = {"fixed":fixed, "added_errors":lost, "net":len(fixed)-len(lost),
                        "both_correct":len(both_correct), "both_wrong":len(both_wrong), "of":len(ids)}
    return result


def calculate(cases, preds, human=None, corpus_sha256=None):
    ids = [c["id"] for c in cases]
    require(len(set(ids))==len(ids) and ids, "duplicate or empty cases")
    families = {c["id"]:c["family"] for c in cases}
    pairs = {}
    for c in cases:
        require(c["family"] in ("handoff","sight"), "invalid family")
        pairs.setdefault(c["pair"],[]).append(c["id"])
    require(all(len(ii)==2 and families[ii[0]]==families[ii[1]] for ii in pairs.values()),"unmatched pair")
    require(set(preds)=={f"{i}.{m}.{r}" for i in ids for m in METHODS for r in RUNS},"missing/extra prediction slots")
    matrix = {}
    for m in METHODS:
        for r in RUNS:
            cells = {}
            for i in ids:
                z = preds[f"{i}.{m}.{r}"]
                require(isinstance(z,dict) and "act" in z and "state" in z and z["act"] in (*ACTS,None)
                        and z["state"] in (*STATES,None),f"invalid prediction {i}.{m}.{r}")
                cells[i]={field:z[field] for field in ("act","state")}
            matrix[f"{m}.{r}"]=cells
    out={"reference":"none","annotated":0,"planned":len(ids),"corpus_sha256":corpus_sha256,
         "repeat":{m:comparison(ids,matrix[m+".A"],matrix[m+".A2"]) for m in METHODS},
         "between_methods":comparison(ids,matrix["direct.A"],matrix["scoped.A"]),
         "against_human":None, "scoped_minus_direct":None}
    if human is None:return out
    require(human["source"]=="human_self_report" and human["corpus_sha256"]==corpus_sha256
            and bool(corpus_sha256),"human reference lacks provenance or uses another corpus")
    labels=human["labels"]
    require(isinstance(labels,dict) and set(labels)<=set(ids),"unknown human item IDs")
    gold={}
    for i,x in labels.items():
        if x is None:continue
        require(isinstance(x,dict) and x.get("act") in ACTS and x.get("state") in STATES,f"incomplete human label {i}")
        gold[i] = {k:x[k] for k in ("act","state")}
    selected = [i for i in ids if i in gold]
    out.update(reference="single_independent_human",annotated=len(selected),skipped=[i for i in ids if i not in gold],
               against_human={},scoped_minus_direct={})
    for k,model in matrix.items():
        scopes={"all":(selected,pairs)}
        for family in ("handoff","sight"):
            scopes[family]=([i for i in selected if families[i]==family],
                            {p:v for p,v in pairs.items() if families[v[0]]==family})
        out["against_human"][k]={name:against(ii,pp,gold,model) for name,(ii,pp) in scopes.items()}
    for r in RUNS:
        out["scoped_minus_direct"][r]=delta(selected,gold,matrix["direct."+r],matrix["scoped."+r])
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human",type=Path,help="Only pass an actual independent human ballot.")
    ap.add_argument("--cases",type=Path,default=HERE/"cases.json")
    ap.add_argument("--predictions",type=Path,default=HERE/"observed/predictions.json")
    ap.add_argument("--out",type=Path,default=HERE/"observed/results.json");ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    result=calculate(load(args.cases),load(args.predictions),load(args.human) if args.human else None,
                     hashlib.sha256(args.cases.read_bytes()).hexdigest())
    if args.check:require(load(args.out)==result,"saved results differ")
    else:
        args.out.parent.mkdir(exist_ok=True,parents=True)
        tmp=args.out.with_suffix(".tmp");tmp.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode());tmp.replace(args.out)
    print(result["reference"], result["annotated"], "/",result["planned"])


if __name__=="__main__":main()
