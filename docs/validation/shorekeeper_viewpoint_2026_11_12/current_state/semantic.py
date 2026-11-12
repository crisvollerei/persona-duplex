"""Freeze SAY-only reading packets; score paired fixed-history arms offline."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
from random import Random
import json


HERE = Path(__file__).resolve().parent
OUT = HERE / "observed"
SCENES = HERE.parent / "scenes.json"
RUBRIC = HERE.parent / "rubric.md"
METHODS = ("control", "state_card")
CODERS = ("C1", "C2")
FIELDS = {
    "knowledge": ("insufficient", "claimed_known", "withheld", "unspecified", "mixed"),
    "identity": ("none", "correct", "incorrect", "hint_only", "mixed"),
    "pact": ("active", "released", "historical", "unmentioned", "ambiguous"),
    "shared": ("recognized", "denied", "unmentioned", "ambiguous"),
}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def load(path):
    def unique(items):
        obj = dict(items)
        require(len(obj) == len(items), f"duplicate JSON key: {path}")
        return obj
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def emit(path, obj, check):
    if check or path.exists():
        require(load(path) == obj, f"frozen artifact mismatch: {path.name}")
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode())
        tmp.replace(path)


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def slot(v, h, a, c, s):
    return f"{v}.{h}.{a}.{c}.{s}"


def prepare(data, archive):
    variants = {v["id"]: v for v in data["variants"]}
    require(set(variants) == {"V1", "V2", "V3"} and len(data["variants"]) == 3, "unexpected variants")
    turns = {r["id"]: r for r in archive}
    expected = {slot(*k) for k in product(variants, ("A", "B"), (2, 3), METHODS, (1, 2))}
    require(set(turns) == expected and len(archive) == len(turns), "missing, duplicate or unexpected slot")
    for k, r in turns.items():
        require(k == slot(r["variant"], r["history"], r["act"], r["condition"], r["sample"]), "slot mislabel")
        require(r["status"] in ("delivered", "empty", "error") and isinstance(r["say"], str)
                and (r["status"] == "delivered") == bool(r["say"].strip()), "invalid visible status")
    order = sorted(k for k,r in turns.items() if r["status"] == "delivered")
    Random(8721063).shuffle(order)
    key = {f"K{i:02d}": k for i,k in enumerate(order, 1)}
    packet = []
    for blind_id, k in key.items():
        r = turns[k]; v = variants[r["variant"]]; act = r["act"]
        packet.append(dict(id=blind_id, act=act, question=data["question"], reference_name=v["source"],
                           authorized_facts=[{f:e[f] for f in ("shorekeeper","traveler")} for e in v["events"][:act]],
                           say=r["say"]))
    return turns, variants, packet, key


def from_coder(rows, key, turns):
    require(isinstance(rows, list), "expected an array of independently coded SAY")
    mapping = {}
    for r in rows:
        require(isinstance(r, dict) and set(r) == {"id", "quote", *FIELDS}, "bad coding fields")
        ident = r["id"]
        require(ident in key and key[ident] not in mapping, "bad or duplicate blind ID")
        require(all(r[f] in domain for f, domain in FIELDS.items()), f"bad label: {ident}")
        require(isinstance(r["quote"], str) and r["quote"].strip() and r["quote"] in turns[key[ident]]["say"],
                f"missing original-text quotation: {ident}")
        mapping[key[ident]] = r
    require(set(mapping) == set(key.values()), "not all delivered SAY coded")
    return mapping


def features(r, coding, v):
    if coding is None:
        return None
    k, i, p, s = (coding[f] for f in FIELDS)
    name = any(t in r["say"] for t in (v["source"], *v["aliases"]) if t)
    known_and_kept = k == "claimed_known" and not name and i in ("none", "hint_only")
    recognizes = s == "recognized" and p != "active"
    return dict(primary=known_and_kept if r["act"] == 2 else recognizes,
                strict_no_hint=known_and_kept and i == "none" if r["act"] == 2 else None,
                claimed_known=k == "claimed_known", says_does_not_know=k == "insufficient",
                withheld_without_knowledge_claim=k == "withheld", knowledge_unspecified=k == "unspecified",
                literal_own_name=name, confirmed_identity=i == "correct", incorrect_identity=i == "incorrect",
                conflicting_identity=i == "mixed", hint_only=i == "hint_only",
                pact_active=p == "active", pact_ambiguous=p == "ambiguous", pact_released=p == "released",
                shared_recognized=s == "recognized", shared_denied=s == "denied")


def cell(slots, turns, coding, values):
    rows = [values[k] for k in slots if values[k] is not None]
    common = {"planned":len(slots), "visible":len(rows),
              "valid_protocol":sum((turns[k]["validity"] or {}).get("valid") is True for k in slots)}
    return {**common, "measures":{f:sum(r[f] is True for r in rows)
                                 for f in (rows[0] if rows else {}) if any(r[f] is not None for r in rows)},
            "label_distributions":{f:dict(sorted(Counter(coding[k][f] for k in slots if k in coding).items()))
                                   for f in FIELDS}}


def paired(pairs, values):
    bins = {b:[] for b in ("gain", "loss", "both_met", "neither_met", "missing")}
    for label, x, y in pairs:
        a,b=values[x],values[y]
        category = ("missing" if a is None or b is None else "both_met" if a["primary"] and b["primary"]
                    else "gain" if b["primary"] else "loss" if a["primary"] else "neither_met")
        bins[category].append(label)
    return {"counts":{k:len(v) for k,v in bins.items()}, "net":len(bins["gain"])-len(bins["loss"]),
            "comparable":len(pairs)-len(bins["missing"]),
            "gain_ids":bins["gain"], "loss_ids":bins["loss"], "missing_ids":bins["missing"]}


def score(turns, variants, key, coders):
    by_coder = {}
    code_values = {}
    vh = list(product(sorted(variants), ("A","B")))
    for c, coding in coders.items():
        v = {k:features(r,coding.get(k),variants[r["variant"]]) for k,r in turns.items()}
        code_values[c] = v
        cells, matched, resample, histories = {}, {}, {}, {}
        for act, method in product((2,3), METHODS):
            for sample in (1,2,"all"):
                slots=[k for k,r in turns.items() if r["act"]==act and r["condition"]==method and sample in ("all",r["sample"])]
                cells[f"{method}.{act}.{sample}"] = cell(slots,turns,coding,v)
            pairs=[(f"{a}.{h}",slot(a,h,act,method,1),slot(a,h,act,method,2)) for a,h in vh]
            resample[f"{method}.{act}"]=paired(pairs,v)
        for act in (2,3):
            for sample in (1,2,"all"):
                ss=(1,2) if sample=="all" else (sample,)
                pairs=[(f"{a}.{h}.{s}",slot(a,h,act,"control",s),slot(a,h,act,"state_card",s))
                       for (a,h),s in product(vh,ss)]
                matched[f"{act}.{sample}"]=paired(pairs,v)
            for a,h in vh:
                histories[f"{a}.{h}.{act}"] = {m:[v[k]["primary"] if v[k] is not None else None
                                                  for k in (slot(a,h,act,m,s) for s in (1,2))] for m in METHODS}
        by_coder[c]=dict(by_condition_act_sample=cells, card_vs_control_same_sample=matched,
                         same_request_sample2_vs_sample1=resample, per_history_primary=histories)
    diffs={}
    cols=[*FIELDS,"primary"]
    for ident,k in key.items():
        ds=[f for f in FIELDS if coders["C1"][k][f]!=coders["C2"][k][f]]
        if code_values["C1"][k]["primary"]!=code_values["C2"][k]["primary"]: ds.append("primary")
        if ds: diffs[ident]=ds
    return dict(planned=len(turns),visible=len(key),coders="two independent method-blind AI codings; no human ratings",
                agreement={f:{"equal":len(key)-sum(f in fs for fs in diffs.values()),"of":len(key)} for f in cols},
                disagreements=diffs,by_coder=by_coder,
                interpretation="Denominators: planned and delivered both reported; missing never counts as passing or paired gain/loss. "
                "Act 2: recognizes own knowledge and neither confirms any identity nor contains the exact source or alias; indirect hint allowed, with stricter no-hint variant reported. "
                "Act 3: recognizes listener knowledge and does not treat old pact as active. Naming is descriptive. "
                "Same-number samples match fixed history, not random seeds or independent scenes.")


def main():
    p=ArgumentParser(description=__doc__); p.add_argument("phase",choices=("prepare","score"));p.add_argument("--check",action="store_true")
    a=p.parse_args()
    turns,variants,packet,key=prepare(load(SCENES),load(OUT/"turns.json"))
    emit(OUT/"blind.json",packet,a.check or a.phase=="score");emit(OUT/"key.json",key,a.check or a.phase=="score")
    if a.phase=="score":
        coders={c:from_coder(load(OUT/f"coding_{c}.json"),key,turns) for c in CODERS}
        result=score(turns,variants,key,coders)
        result["fingerprints"]={p.name:fingerprint(p) for p in [SCENES,RUBRIC,OUT/"turns.json",OUT/"blind.json",OUT/"key.json",
                                                               *(OUT/f"coding_{c}.json" for c in CODERS)]}
        emit(OUT/"semantic_summary.json",result,a.check)
    print(f"{a.phase}: {len(packet)} visible / {len(turns)} planned")


if __name__=="__main__":
    main()
