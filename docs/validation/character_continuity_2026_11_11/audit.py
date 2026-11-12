"""Offline archive audit. This module does not open ballots or grade role performance."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona.parse import parse, plain_text, say_wire
from persona.prompt import build_history_entry


def require(value, why):
    if not value:
        raise ValueError(why)


def load(p):
    return json.loads(p.read_bytes())


def audit(out):
    reg, tracks, blind = [load(out / name) for name in
                          ("registration.json", "trajectories.json", "blind.json")]
    protocol = load(HERE / "protocol.json")
    scenes = {s["id"]: s for s in load(HERE / "materials.json")["scenes"]}
    methods = ("framework", "examples")
    branches = {s + "." + m for s in scenes for m in methods}
    require(len(scenes) == 3 and len(reg["order"]) == len(branches)
            and {".".join(b) for b in reg["order"]} == set(tracks) == branches, "trajectory coverage")
    require(protocol["provider"] == reg["provider"], "provider settings")
    cited = list((ROOT / "persona").glob("*.py")) + [ROOT / protocol["role"]]
    cited += [HERE / n for n in ("protocol.json", "materials.json", "baseline.py", "study.py")]
    cited += [HERE.parent / "noninferiority_pilot_2026_11_10" / n for n in ("selection.json","prompts.json")]
    require({p.relative_to(ROOT).as_posix() for p in cited} == set(reg["hashes"]), "source coverage")
    for p in cited:
        require(sha256(p.read_bytes()).hexdigest() == reg["hashes"][p.relative_to(ROOT).as_posix()], p.name)
    with gzip.open(out / "calls.jsonl.gz", "rb") as fp:
        raw = [json.loads(line) for line in fp]
    rec = {r["id"]: r for r in raw}
    require(len(rec) == len(raw), "duplicate foreground slot")
    stats = {m:Counter() for m in methods}
    models, receipts, used = Counter(), set(), set()
    problems, full_histories, visible_histories = [], 0, 0
    for branch in sorted(branches):
        s,m = branch.split(".")
        c,rows = scenes[s],tracks[branch]
        require(len(rows) == len(c["turns"]) == 4, branch)
        seen, prior_wire, failed = [], [], False
        for j,t in enumerate(rows,1):
            key = branch + "." + str(j)
            user = c["turns"][j-1]
            require(t["user"] == user, f"{key}: script")
            st = stats[m]
            st["planned"] += 1
            if failed:
                require(key not in rec and t == dict(user=user,say="",validity=None,status="upstream_missing"), key)
                st["upstream_missing"] += 1
                continue
            r = rec[key]
            used.add(key)
            require(r["params"] == reg["provider"], key + ": parameters")
            st["requests"] += 1
            msgs = r["messages"]
            require([z["role"] for z in msgs] == ["system", *(["user","assistant"]*(j-1)), "user"], key + ": roles")
            require([z["content"] for z in msgs if z["role"] == "user"] == c["turns"][:j], key + ": user chronology")
            require([z["content"] for z in msgs if z["role"] == "assistant"] == prior_wire, key + ": own history")
            # A different branch's *identical* visible text cannot prove leakage: only
            # search for disjoint full replies and future/noncurrent full user turns.
            disallowed = {x for k,v in scenes.items() for x in v["turns"] if k != s}
            disallowed.update(c["turns"][j:])
            disallowed.update(v["focus"] for v in scenes.values())
            allowed = set(seen)
            disallowed.update(t2["say"] for b2,ts in tracks.items() if b2!=branch for t2 in ts
                              if t2["say"] and t2["say"] not in allowed and len(t2["say"])>=10)
            require(not any(text in z["content"] for text in disallowed for z in msgs), key + ": unexpected full line")
            if not r["ok"]:
                require(r["raw"] == "" and t["status"] == "call_failed" and t["validity"] is None
                        and t["say"] == "", key + ": provider failure")
                failed = True
                st["call_failed"] += 1
                continue
            response = r["response"]
            require(response["id"] and response["id"] not in receipts, key + ": receipt")
            receipts.add(response["id"])
            models[response["model"]] += 1
            st["provider_success"] += 1
            u = response["usage"]
            require(u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], key + ": usage")
            for field in ("prompt_tokens","completion_tokens","total_tokens"):
                st[field] += u[field]
            if m == "framework":
                parsed = parse(r["raw"])
                visible = parsed.say or plain_text(r["raw"])
                validity = parsed.validity()
                require(validity == t["validity"], key + ": protocol")
                if validity["valid"]:
                    st["valid_protocol"] += 1
                    wire = build_history_entry(visible, parsed.head, parsed.tail)
                    full_histories += (4-j)
                else:
                    problems.append(dict(slot=key, reasons=validity["reasons"], visible=bool(visible)))
                    st["invalid_protocol"] += 1
                    wire = say_wire(visible)
                    visible_histories += (4-j)
            else:
                require(t["validity"] is None, key + ": unexpected framework condition")
                visible = wire = r["raw"].strip()
            require(t["say"] == visible and t["status"] == ("delivered" if visible else "empty"),
                    key + ": visible response")
            st[t["status"]] += 1
            seen.append(visible)
            prior_wire.append(wire)
    require(set(rec) == used, "orphan receipt")
    require([s["id"] for s in blind["scenes"]] == list(scenes), "blind scene order")
    for z in blind["scenes"]:
        s = z["id"]
        sides = reg["codebook"][s]
        require(set(z) == {"id", "lines"} and set(z["lines"]) == set(sides) == {"A", "B"}
                and set(sides.values()) == set(methods), "blind codebook")
        for side,m in sides.items():
            require(z["lines"][side] == [t["say"] for t in tracks[s+"."+m]], s + side + ": blind text")
    return dict(source_files=len(cited), scenes=len(scenes), trajectories=len(branches),
                provider_receipts=len(receipts), provider_models=dict(models),
                by_method={m:dict(st) for m,st in stats.items()},
                nonconforming_framework_turns=problems,
                framework_history_entries_with_metadata=full_histories,
                framework_history_entries_visible_only=visible_histories,
                scope="Offline check of raw requests, visible text, local protocol, costs and blind packet; no semantic or human ratings.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"observed")
    p.add_argument("--check", action="store_true")
    args=p.parse_args()
    result=audit(args.out)
    dest=args.out/"audit.json"
    if args.check:
        require(load(dest) == result,"stored audit differs")
    else:
        tmp=dest.with_suffix(".tmp")
        tmp.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode())
        tmp.replace(dest)
    print(json.dumps(result,ensure_ascii=False))
