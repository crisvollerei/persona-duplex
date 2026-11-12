"""Observable delivery and literal name/canary exposure; no semantic judge, API or reference reply."""
from collections import Counter
import argparse
import hashlib
import itertools
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONDITIONS = ("scoped", "full_notes")
REPLICATES = ("A", "B")
ACTS = (1, 2, 3)
STATUS = ("delivered", "empty", "error", "upstream_missing")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def names(variant):
    values = [variant["source"], *variant["aliases"]]
    require(all(isinstance(s, str) and s.strip() for s in values), "empty or malformed source/alias")
    return tuple(dict.fromkeys(values))


def metric_rows(scenes, turns):
    variants = {v["id"]: v for v in scenes["variants"]}
    require(len(variants) == len(scenes["variants"]) and bool(variants), "duplicate or absent variant")
    for v in variants.values():
        require(isinstance(v["aliases"], list) and isinstance(v["private_nonce"], str)
                and v["private_nonce"].strip(), "malformed source or canary metadata")
        names(v)
    indexed = {}
    for rec in turns:
        key = (rec["variant"], rec["condition"], rec["replicate"], rec["act"])
        require(key not in indexed and key[0] in variants and key[1] in CONDITIONS
                and key[2] in REPLICATES and type(key[3]) is int and key[3] in ACTS,
                f"unexpected/duplicate output slot {key}")
        require(rec["status"] in STATUS, f"unrecognized delivery status at {key}")
        require(isinstance(rec["say"], str), f"malformed visible SAY at {key}")
        indexed[key] = rec

    rows = []
    for key in itertools.product(variants, CONDITIONS, REPLICATES, ACTS):
        v, condition, rep, act = key
        r = indexed.get(key, {})
        say = r.get("say", "")
        visible = bool(say.strip())
        val = (r.get("validity") or {}).get("valid")
        require(val is None or type(val) is bool, f"non-boolean protocol validity at {key}")
        own = variants[v]
        own_names = [n for n in names(own) if n in say]
        other_names = {other: [n for n in names(x) if n in say] for other, x in variants.items() if other != v}
        canaries = [other for other, x in variants.items() if x["private_nonce"] in say]
        rows.append({"variant": v, "condition": condition, "replicate": rep, "act": act,
                     "status": r.get("status", "unrecorded"), "protocol_valid": val, "visible": visible,
                     "canonical_hit": own["source"] in say if visible else None,
                     "source_or_alias_hit": bool(own_names) if visible else None,
                     "matched_terms": own_names, "canary_source_ids": canaries,
                     "other_variant_terms": {other: ns for other, ns in other_names.items() if ns}})
    return rows


def counts(rows):
    visible = [r for r in rows if r["visible"]]
    return {
        "planned": len(rows), "status": dict(Counter(r["status"] for r in rows)),
        "visible": len(visible), "no_visible": len(rows) - len(visible),
        "protocol_valid": sum(r["protocol_valid"] is True for r in rows),
        "protocol_invalid": sum(r["protocol_valid"] is False for r in rows),
        "protocol_unavailable": sum(r["protocol_valid"] is None for r in rows),
        "among_visible": {"of": len(visible), "canonical_hit": sum(r["canonical_hit"] for r in visible),
                          "source_or_alias_hit": sum(r["source_or_alias_hit"] for r in visible),
                          "any_authors_canary_hit": sum(bool(r["canary_source_ids"]) for r in visible),
                          "other_variant_source_or_alias_hit": sum(bool(r["other_variant_terms"]) for r in visible)},
    }


def analyze(scenes, turns):
    require(isinstance(turns, list), "expected a list of archived turn records")
    rows = metric_rows(scenes, turns)
    by_cell, by_condition_act = {}, {}
    for c in CONDITIONS:
        for a in ACTS:
            rr = [r for r in rows if r["condition"] == c and r["act"] == a]
            by_condition_act[f"{c}.{a}"] = counts(rr)
            for rep in REPLICATES:
                by_cell[f"{c}.{rep}.{a}"] = counts([r for r in rr if r["replicate"] == rep])
    return {"scope": "Exact substrings in delivered SAY only; any mention (even conjecture or negation) "
                     "counts as exposure. Missing SAY is not a secrecy success. No semantic scoring. "
                     "Act 3 naming is descriptive, not required. Replicates are not distinct stories.",
            "overall": counts(rows), "by_condition_act": by_condition_act,
            "by_condition_replicate_act": by_cell, "rows": rows}


def selfcheck():
    # Synthetic fixtures exercise measurement only. No provider, no files, no real study output.
    fixture = {"variants": [{"id": "x", "source": "墨盒甲", "aliases": ["盒甲"], "private_nonce": "X:Y:Z"},
                            {"id": "y", "source": "墨盒乙", "aliases": [], "private_nonce": "Z:Y:X"}]}
    def row(act, say, valid):
        return dict(variant="x", condition="scoped", replicate="A", act=act, say=say,
                    status="delivered" if say else "error", validity={"valid": valid} if valid is not None else None)
    a = analyze(fixture, [row(1, "我没说这是墨盒甲。", False), row(2, "", None),
                          row(3, "听见墨盒乙了。Z:Y:X", True)])
    assert a["overall"]["planned"] == 24 and a["overall"]["visible"] == 2
    z = a["by_condition_replicate_act"]["scoped.A.1"]
    assert z["among_visible"]["source_or_alias_hit"] == 1 and z["protocol_invalid"] == 1
    assert a["by_condition_replicate_act"]["scoped.A.2"]["among_visible"]["of"] == 0
    assert a["overall"]["among_visible"]["other_variant_source_or_alias_hit"] == 1
    assert a["overall"]["among_visible"]["any_authors_canary_hit"] == 1
    try:
        analyze(fixture, [row(1, "", None), row(1, "", None)])
    except ValueError:
        return
    raise AssertionError("duplicate output accepted")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=Path, default=HERE / "scenes.json")
    p.add_argument("--turns", type=Path, default=HERE / "observed/turns.json")
    p.add_argument("--out", type=Path, default=HERE / "observed/metrics.json")
    p.add_argument("--check", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    selfcheck()
    if a.self_test:
        print("Synthetic measurement checks passed.")
        return
    data = [path.read_bytes() for path in (a.scenes, a.turns)]
    result = analyze(*map(json.loads, data))
    result["input_sha256"] = dict(zip(("scenes", "turns"), [hashlib.sha256(b).hexdigest() for b in data]))
    if a.check:
        require(json.loads(a.out.read_bytes()) == result, "Stored metrics differ from recomputation")
    else:
        require(a.out.resolve() not in {p.resolve() for p in (a.scenes, a.turns, Path(__file__))},
                "Refusing to overwrite an input")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    print(result["overall"]["visible"], "/", result["overall"]["planned"], "slots have visible text")


if __name__ == "__main__":
    main()
