"""Score a registered arrival observer without promoting model agreement to ground truth."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PEOPLE = ("沈遥", "许原")
STATES = ("present", "future", "unknown")
BASES = ("explicit", "pragmatic", "none")
METHODS = ("legacy", "typed")
ORDERS = ("A", "B")
INVALID = "invalid"


def require(test, message):
    if not test:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def ratio(n, d):
    return {"n": n, "of": d, "rate": n/d if d else None}


def state_basis(obj):
    return (obj["state"], obj.get("basis"))


def check_human(human, ids, corpus_sha):
    require(human.get("source") == "human_self_report", "not an independent human annotation")
    require(isinstance(human.get("reviewer"), str) and human["reviewer"].strip(), "reviewer ID required")
    require(human.get("corpus_sha256") == corpus_sha, "annotations use another corpus")
    labels = human.get("labels")
    require(isinstance(labels, dict) and set(labels) <= set(ids), "unknown case IDs")
    for i, row in labels.items():
        if row is None:
            continue
        require(isinstance(row, dict) and set(row) == set(PEOPLE), f"incomplete labels for {i}")
        for p, z in row.items():
            require(isinstance(z, dict) and z.get("state") in STATES and z.get("basis") in BASES,
                    f"invalid annotation {i}/{p}")
    return {i: labels[i] for i in ids if labels.get(i) is not None}


def read_predictions(raw, ids):
    orders = (*ORDERS,"A2") if any(k.endswith(".A2") for k in raw) else ORDERS
    require(set(raw) == {f"{i}.{m}.{o}" for i in ids for m in METHODS for o in orders},
            "missing or extra observation positions")
    clean = {}
    for m in METHODS:
        for o in orders:
            cells = {}
            for i in ids:
                item = raw[f"{i}.{m}.{o}"]["people"]
                require(set(item) == set(PEOPLE), f"invalid people in {i}.{m}.{o}")
                cells[i] = {}
                for p, response in item.items():
                    state, basis = state_basis(response)
                    require(state is None or state in STATES, "invalid prediction state")
                    require(m == "legacy" or basis is None or basis in BASES, "invalid prediction basis")
                    cells[i][p] = {"state":state, "basis":basis if m == "typed" else None}
            clean[m+"."+o] = cells
    return clean


def agreement(left, right, ids, typed):
    """No standard of correctness: count exact agreement only when both outputs are valid."""
    out = {}
    for fields in ([("state",), ("basis",), ("state","basis")] if typed else [("state",)]):
        key = "joint" if len(fields)==2 else fields[0]
        denominator = equal = exact_cases = complete_cases = 0
        differences = []
        for i in ids:
            complete = exact = True
            for p in PEOPLE:
                l = tuple(left[i][p][k] for k in fields)
                r = tuple(right[i][p][k] for k in fields)
                if None in l+r:
                    complete = exact = False
                    continue
                denominator += 1
                if l == r:
                    equal += 1
                else:
                    exact = False
                    differences.append({"case":i,"person":p,"left":list(l),"right":list(r)})
            complete_cases += int(complete)
            exact_cases += int(complete and exact)
        out[key] = {"people_agree":ratio(equal,denominator), "cases_agree":ratio(exact_cases,complete_cases),
                    "people_missing":2*len(ids)-denominator, "cases_missing":len(ids)-complete_cases,
                    "differences":differences}
    return out


def against(cases, gold, pred, typed):
    ids = [c["id"] for c in cases if c["id"] in gold]
    state_cm = {s:{p:0 for p in (*STATES,INVALID)} for s in STATES}
    basis_cm = {s:{p:0 for p in (*BASES,INVALID)} for s in BASES} if typed else None
    diffs = []
    case_scores = {}
    total = Counter()
    for i in ids:
        result = Counter()
        for p in PEOPLE:
            hs,hb = state_basis(gold[i][p])
            ps,pb = state_basis(pred[i][p])
            e = ps == hs
            be = typed and pb == hb
            state_cm[hs][ps or INVALID] += 1
            if typed: basis_cm[hb][pb or INVALID] += 1
            result["state_correct"] += int(e)
            result["state_valid"] += int(ps in STATES)
            result["basis_correct"] += int(be)
            result["basis_valid"] += int(typed and pb in BASES)
            result["joint_correct"] += int(e and be)
            total["false_present"] += int(ps=="present" and hs!="present")
            total["false_explicit_present"] += int(typed and ps=="present" and pb=="explicit" and hs!="present")
            total["false_present_opportunities"] += int(hs!="present")
            total["false_explicit"] += int(typed and pb=="explicit" and hb!="explicit")
            total["false_explicit_opportunities"] += int(typed and hb!="explicit")
            total["given_decisive"] += int(ps in ("present","future"))
            total["correct_decisive"] += int(hs in ("present","future") and ps==hs)
            total["human_decisive"] += int(hs in ("present","future"))
            total["gold_unknown"] += int(hs=="unknown")
            if not e or (typed and not be):
                diffs.append({"case":i,"person":p,"human":gold[i][p],
                              "observed":{"state":ps,"basis":pb} if typed else {"state":ps}})
        total.update(result)
        targets_are_both_present = all(gold[i][p]["state"]=="present" for p in PEOPLE)
        model_says_both_present = all(pred[i][p]["state"]=="present" for p in PEOPLE)
        total["false_both_present"] += int(not targets_are_both_present and model_says_both_present)
        total["not_both_present"] += int(not targets_are_both_present)
        case_scores[i] = {k:result[k+"_correct"]==2 for k in (("state","basis","joint") if typed else ("state",))}
    group_ids = {}
    for c in cases:
        group_ids.setdefault(c["pair"],[]).append(c["id"])
    whole_pairs = {g:v for g,v in group_ids.items() if all(i in gold for i in v)}
    both = {g:{k:all(case_scores[i][k] for i in indexes) for k in (("state","basis","joint") if typed else ("state",))}
            for g,indexes in whole_pairs.items()}
    summary = {}
    np = len(ids)*2
    for k in ("state","basis","joint") if typed else ("state",):
        summary[k]={"people_correct":ratio(total[k+"_correct"],np),
                    "cases_correct":ratio(sum(v[k] for v in case_scores.values()),len(ids)),
                    "pairs_all_correct":ratio(sum(v[k] for v in both.values()),len(whole_pairs))}
    return {"scores":summary,"case_scores":case_scores,"paired_cases":both,
            "skipped_pairs":[g for g in group_ids if g not in whole_pairs],
            "confusion":{"state":state_cm,"basis":basis_cm},"mismatches":diffs,
            "coverage":{"state_valid":ratio(total["state_valid"],np),
                        "basis_valid":ratio(total["basis_valid"],np) if typed else None,
                        "decisive_fraction":ratio(total["given_decisive"],np),
                        "decisive_state_recall":ratio(total["correct_decisive"],total["human_decisive"]),
                        "always_unknown_state_accuracy":ratio(total["gold_unknown"],np),
                        "per_state_recall":{k:ratio(state_cm[k][k],sum(state_cm[k].values())) for k in STATES}},
            "risk":{"false_person_present":ratio(total["false_present"],total["false_present_opportunities"]),
                    "false_explicit_present":ratio(total["false_explicit_present"],total["false_present_opportunities"]) if typed else None,
                    "false_both_present":ratio(total["false_both_present"],total["not_both_present"]),
                    "unsupported_explicit":ratio(total["false_explicit"],total["false_explicit_opportunities"]) if typed else None}}


def state_delta(gold, old, new, ids):
    def delta(bits):
        diff = Counter("gain" if b and not a else "loss" if a and not b else "both_right" if a else "both_wrong" for a,b in bits)
        return {k:diff[k] for k in ("gain","loss","both_right","both_wrong")} | {"net":diff["gain"]-diff["loss"],"n":len(bits)}
    person, full = [], []
    for i in ids:
        z = [(old[i][p]["state"]==gold[i][p]["state"],new[i][p]["state"]==gold[i][p]["state"]) for p in PEOPLE]
        person.extend(z); full.append((all(u for u,v in z),all(v for u,v in z)))
    return {"people":delta(person),"cases":delta(full)}


def score(cases,predictions,humans,corpus_sha):
    ids=[c["id"] for c in cases]
    require(len(set(ids))==len(ids),"duplicate case IDs")
    require(all(n==2 for n in Counter(c["pair"] for c in cases).values()),"each stimulus pair must have two cases")
    require(len(humans)<=2,"at most two independent reviewers")
    pred=read_predictions(predictions,ids)
    annotations={}
    for human in humans:
        labels=check_human(human,ids,corpus_sha)
        who=human["reviewer"].strip()
        require(who not in annotations,"the same reviewer cannot be counted twice")
        annotations[who]=labels
    hcount=len(annotations)
    result={"corpus_sha256":corpus_sha,"cases":len(ids),"reviewer_count":hcount,
            "reference_status":("no_human_reference","single_reviewer_only","two_reviewer_files")[hcount],
            "diagnostics":{"between_methods":{o:agreement(pred["legacy."+o],pred["typed."+o],ids,False) for o in ORDERS},
                           "between_orders":{m:agreement(pred[m+".A"],pred[m+".B"],ids,m=="typed") for m in METHODS}},
            "per_reviewer":None,"between_reviewers":None,"two_full_ballots":False,
            "two_reviewer_expand":None,"two_reviewer_state_improved":None}
    if "typed.A2" in pred:
        result["diagnostics"]["identical_input_repeats"]={m:agreement(pred[m+".A"],pred[m+".A2"],ids,m=="typed") for m in METHODS}
    if annotations:
        result["per_reviewer"]={}
    for who,gold in annotations.items():
        chosen=[i for i in ids if i in gold]
        methods={k:against(cases,gold,v,k.startswith("typed.")) for k,v in pred.items() if not k.endswith(".A2")}
        z=methods["typed.A"];risk=z["risk"];test_eligible=len(chosen)==len(ids)==12
        result["per_reviewer"][who]={"annotated_cases":len(chosen),"missing":[i for i in ids if i not in gold],
            "methods":methods,"primary_pass":((z["scores"]["state"]["cases_correct"]["n"]>=10
                and z["scores"]["joint"]["people_correct"]["n"]>=19 and risk["unsupported_explicit"]["of"]>=4
                and risk["unsupported_explicit"]["n"]<=1 and risk["false_explicit_present"]["n"]==0) if test_eligible else None),
            "typed_minus_legacy_state":{o:state_delta(gold,pred["legacy."+o],pred["typed."+o],chosen) for o in ORDERS}}
    if hcount==2:
        l,r=annotations;shared=[i for i in ids if i in annotations[l] and i in annotations[r]]
        result["between_reviewers"]={"left":l,"right":r,"common_cases":len(shared),
                                    "agreement":agreement(annotations[l],annotations[r],shared,True)}
        result["two_full_ballots"]=len(shared)==len(ids)
        if result["two_full_ballots"] and len(ids)==12:
            result["two_reviewer_expand"]=all(r["primary_pass"] for r in result["per_reviewer"].values())
            result["two_reviewer_state_improved"]=all(r["typed_minus_legacy_state"]["A"]["people"]["net"]>0 for r in result["per_reviewer"].values())
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human",nargs="+",type=Path,default=[],help="One or two human ballots; omitted for diagnostics only.")
    ap.add_argument("--predictions",type=Path,default=HERE/"observed/predictions.json")
    ap.add_argument("--repeats",type=Path,help="Optional A2-only file; uses repeats.json beside predictions if present.")
    ap.add_argument("--cases",type=Path,default=HERE/"cases.json")
    ap.add_argument("--out",type=Path,default=HERE/"observed/results.json")
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    predictions=load(args.predictions)
    repeats=args.repeats or args.predictions.with_name("repeats.json")
    if args.repeats is not None or repeats.exists():
        extra=load(repeats)
        require(bool(extra) and all(k.endswith(".A2") for k in extra) and not (extra.keys() & predictions.keys()),
                "malformed repeat records")
        predictions.update(extra)
    data=score(load(args.cases),predictions,[load(f) for f in args.human],hashlib.sha256(args.cases.read_bytes()).hexdigest())
    if args.check:
        require(load(args.out)==data,"stored results do not match recomputation")
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        tmp.replace(args.out)
    print(data["reference_status"],"cases",data["cases"],"reviewers",data["reviewer_count"])


if __name__=="__main__":
    main()
