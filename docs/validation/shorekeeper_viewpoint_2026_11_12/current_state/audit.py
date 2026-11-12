"""Independent, offline comparison of actual request envelopes, frozen histories and returned SAY."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from persona import generation
from persona.config import PersonaConfig
from persona.parse import plain_text, say_wire
from persona.prompt import build_history_entry
from statecard import render_card


def check(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with gzip.open(path, "rb") as f:
        return [json.loads(line) for line in f]


def by_id(records, identify=lambda r: r["id"]):
    indexed = {identify(r): r for r in records}
    check(len(indexed) == len(records), "duplicate record key")
    return indexed


def fingerprint(obj):
    return sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def delivery(raw, cfg):
    p = generation.parse_output(raw, "full")
    return (p.say or plain_text(raw),
            generation.validity(p, "full", protocol="text", allowed=cfg.can_displace(p.head.displace) if p.head else True), p)


def audit(folder):
    reg = read(folder / "registration.json")
    for name, checksum in reg["sha256"].items():
        p = (ROOT / name).resolve()
        check(p.is_relative_to(ROOT) and sha256(p.read_bytes()).hexdigest() == checksum, f"source changed: {name}")
    base = PARENT / "observed"
    plan = read(HERE / "protocol.json")
    for name, checksum in plan["parents"].items():
        p = (PARENT / name).resolve()
        check(p.is_relative_to(PARENT) and sha256(p.read_bytes()).hexdigest() == checksum, "changed frozen parent: "+name)
    original = by_id(rows(base / "calls.jsonl.gz"))
    prior = by_id(read(base / "turns.json"), lambda r: (r["variant"], r["condition"], r["replicate"], r["act"]))
    scene = read(PARENT / "scenes.json")
    variants = by_id(scene["variants"])
    q = scene["question"]
    check(q == "你知道这段回音是谁留下的吗？", "unexpected fixed question")
    config = PersonaConfig.load(ROOT / "personas/shorekeeper.toml")
    previous_ids = {r["response"]["id"] for r in original.values() if r["ok"]}
    observed = by_id(rows(folder / "calls.jsonl.gz"))
    turns = by_id(read(folder / "turns.json"))
    conditions = ("control", "state_card")
    slots = list(itertools.product(("V1", "V2", "V3"), ("A", "B"), (2, 3), conditions, (1, 2)))
    planned = {".".join(map(str, slot)) for slot in slots}
    check(set(observed) == set(turns) == planned, "missing or extra request/outcome slots (including failures)")
    check(len(reg["order"])==len(slots) and set(reg["order"])==set(reg["request_sha256"])==planned,
          "registered allocation does not cover requested design")
    results, groups, receipts, service = {}, {}, set(), Counter()
    digests, first_by_pair, problems, errors = {}, {}, [], []
    for variant, history, act, method, sample in slots:
        parentid = f"{variant}.scoped.{history}.{act}"
        ident = f"{variant}.{history}.{act}.{method}.{sample}"
        ds, trial, origin = observed[ident], turns[ident], original[parentid]
        check((trial["variant"],trial["history"],trial["act"],trial["condition"],trial["sample"]) ==
              (variant, history, act, method, sample), ident+": slot metadata")
        check(origin["ok"] and ds["params"] == origin["params"] == reg["model"] == plan["provider"], ident+": provider settings")
        check(ds["parent_id"]==trial["parent_id"]==parentid, ident+": parent linkage")
        past = []
        for n in range(1,act):
            ot = prior[(variant,"scoped",history,n)]
            earlier = original[f"{variant}.scoped.{history}.{n}"]
            check(earlier["ok"] and ot["status"]=="delivered" and ot["user"]==q, ident+": missing ancestor")
            real_say, val, parsed = delivery(earlier["raw"], config)
            check((real_say,val)==(ot["say"],ot["validity"]), ident+": prior delivered text/validity")
            rendered = build_history_entry(real_say, parsed.head, parsed.tail) if val["valid"] else say_wire(real_say)
            past.extend([dict(role="user",content=q),dict(role="assistant",content=rendered)])
        baseline = origin["messages"]
        check(len(baseline)==2*act and baseline[0]["role"]=="system" and
              baseline[1:]==[*past,dict(role="user", content=q)], parentid+": old history or question changed")
        expected = deepcopy(baseline)
        card = render_card(act)
        check(bool(card.strip()) and card not in baseline[0]["content"], ident+": preexisting card")
        check(not any(term in card for v in variants.values() for term in [v["source"],*v["aliases"],v["private_nonce"]]),
              ident+": card adds name or private identifier")
        if method == "state_card":
            expected[0]["content"] += "\n\n"+card
        check(ds["messages"] == expected, ident+": actual wire messages differ from intended one-field experiment")
        key = f"{parentid}.{method}"
        request_digest = fingerprint(ds["messages"])
        check(request_digest==reg["request_sha256"][ident], ident+": request hash differs from pre-registration")
        check(first_by_pair.setdefault(key, request_digest)==request_digest, ident+": repeated request differs")
        digests[parentid] = dict(messages=fingerprint(baseline), history=fingerprint(past), prior_turns=act-1)
        stage = groups.setdefault(f"{method}.{act}", Counter())
        stage["planned"] += 1
        if not ds["ok"]:
            check(trial["status"]=="error" and not trial["say"] and not ds["raw"], ident+": error discarded")
            errors.append(ident); stage["error"] += 1
            continue
        receipt = ds["response"]; rid=receipt["id"]
        check(isinstance(rid,str) and rid and rid not in receipts and rid not in previous_ids, ident+": reused service response")
        receipts.add(rid); service[receipt["model"]] += 1; stage["fresh_service_responses"] += 1
        say, val, _ = delivery(ds["raw"], config)
        check(trial["validity"]==val and trial["say"]==say and trial["status"]==("delivered" if say else "empty"),
              ident+": actual SAY/validity")
        stage["delivered"] += bool(say); stage["empty"] += not bool(say); stage["valid_protocol"] += val["valid"]
        if not val["valid"]: problems.append(dict(id=ident,reasons=val["reasons"]))
        name = any(t in say for t in [variants[variant]["source"],*variants[variant]["aliases"]])
        if say:
            stage["literal_source_or_alias_in_SAY"] += name
            stage["nonce_in_SAY"] += any(v["private_nonce"] in say for v in variants.values())
        for tok in ("prompt_tokens","completion_tokens","total_tokens"): stage[tok]+=receipt["usage"][tok]
    results.update(registered_file_hashes_verified=len(reg["sha256"]), plots=1, variants=len(variants),
                   selected_parent_trajectories=6, frozen_history_points=len(digests), planned_response_slots=len(slots),
                   fresh_unique_provider_receipts=len(receipts), provider_models=dict(service),
                   independent_same_input_repeats=len(first_by_pair), parent_fingerprints=digests,
                   by_condition_and_act={k:dict(v) for k,v in groups.items()},
                   failed=errors, protocol_issues=problems,
                   note="12 个冻结历史点各有2条件、2复采；三幕变体共用一个剧情。第二幕的实名字面命中是披露风险，第三幕只描述是否点名；本审计不编码语义。")
    return results


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", type=Path, default=HERE/"observed")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true")
    args=ap.parse_args()
    result=audit(args.folder)
    target=args.out or args.folder/"audit.json"
    if args.check: check(read(target)==result, "saved audit differs from recomputation")
    else:
        check(not target.exists() or target.resolve()==(args.folder/"audit.json").resolve() or read(target)==result,
              "cannot overwrite an existing unrelated file")
        tmp=target.with_suffix(".tmp")
        tmp.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode()); tmp.replace(target)
    print(f"{result['fresh_unique_provider_receipts']} fresh service responses; {result['frozen_history_points']} frozen points")
