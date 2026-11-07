"""Deterministic method-hidden material preparation and fixed candidate gate."""
import argparse
import random
from benchmark import HERE, load, save, sha, context

DATA = HERE


def pools():
    if (DATA / "review_key.json").exists():
        raise ValueError("Review key already exists; use a new study data directory")
    cases = {c["id"]: c for c in load(HERE / "cases.json")}
    data = load(DATA / "pools/results.json")
    texts = [(r["case"], c) for r in data["records"] for c in r["candidates"]]
    random.Random(20261107191).shuffle(texts)
    key, inputs = [], []
    for i, (case_id, candidate) in enumerate(texts, 1):
        id = f"T{i:03}"
        key.append(dict(id=id, case=case_id, text=candidate["text"], candidate_ids=[candidate["id"]], output_ids=[]))
        inputs.append(dict(id=id, **context(cases[case_id]), say=candidate["text"]))
    ordered = list(cases)
    random.Random(20261107192).shuffle(ordered)
    groups, group_key = [], []
    for i, case_id in enumerate(ordered, 1):
        id = f"G{i:02}"
        group_key.append(dict(id=id, case=case_id))
        group = [dict(id=r["id"], text=r["text"]) for r in key if r["case"] == case_id]
        groups.append(dict(scene_id=id, **context(cases[case_id]), candidates=group))
    save(DATA / "reviews/pool_inputs.json", inputs)
    save(DATA / "reviews/pool_groups.json", groups)
    save(DATA / "review_key.json", key)
    save(DATA / "pool_group_key.json", group_key)
    print(f"Prepared {len(inputs)} unique candidate texts and {len(groups)} anonymous groups")


def gate():
    groups = load(DATA / "reviews/pool_groups.json")
    records = load(DATA / "pools/results.json")["records"]
    expected = {g["scene_id"]: {c["id"] for c in g["candidates"]} for g in groups}
    measures = []
    for coder in (1, 2):
        rows = load(DATA / f"reviews/gate_ratings_{coder}.json")
        assert len(rows) == len(expected) and {r["scene_id"] for r in rows} == set(expected)
        for row in rows:
            ids = expected[row["scene_id"]]
            assert len(row["candidates"]) == len(ids) and {c["id"] for c in row["candidates"]} == ids
            assert all(type(c["semantic_usable"]) is bool for c in row["candidates"])
            assert type(row["at_least_two_distinct_usable"]) is bool
            if row["at_least_two_distinct_usable"]:
                pair = row["distinct_pair"]
                assert len(set(pair)) == 2 and set(pair).issubset({c["id"] for c in row["candidates"] if c["semantic_usable"]})
        usable = sum(c["semantic_usable"] for r in rows for c in r["candidates"])
        distinct = sum(r["at_least_two_distinct_usable"] for r in rows)
        measures.append(dict(coder=coder, usable=usable, planned_candidates=72, distinct_pools=distinct,
                             passed=usable >= 65 and distinct >= 10,
                             ratings_sha256=sha(DATA / f"reviews/gate_ratings_{coder}.json")))
    result = dict(pools_sha256=sha(DATA / "pools/results.json"), schema_ok=sum(r["status"] == "ok" for r in records),
                  coders=measures, proceed=len(records) == 12 and all(r["status"] == "ok" for r in records) and all(m["passed"] for m in measures),
                  policy="Whole batch only; no candidate removal, regeneration or replacement")
    save(DATA / "gate.json", result)
    print(result)


def direct():
    if (DATA / "reviews/direct_inputs.json").exists():
        raise ValueError("Direct review input already exists")
    cases = {c["id"]: c for c in load(HERE / "cases.json")}
    key = load(DATA / "review_key.json")
    for r in key:
        r["output_ids"] = []
    existing = {(r["case"], r["text"]): r for r in key}
    rows = load(DATA / "main/results.json")["records"]
    new = []
    for row in rows:
        if row["status"] != "ok":
            continue
        index = row["case"], row["say"]
        if index not in existing:
            item = dict(id=None, case=row["case"], text=row["say"], candidate_ids=[], output_ids=[])
            existing[index] = item
            new.append(item)
        existing[index]["output_ids"].append(row["id"])
    random.Random(20261107193).shuffle(new)
    inputs = []
    for i, row in enumerate(new, 1):
        row["id"] = f"E{i:03}"
        inputs.append(dict(id=row["id"], **context(cases[row["case"]]), say=row["text"]))
    key.extend(new)
    save(DATA / "review_key.json", key)
    save(DATA / "reviews/direct_inputs.json", inputs)
    print(f"Prepared {len(new)} additional unique texts; {len(key)} unique texts total")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("phase", choices=("pools", "gate", "direct")); p.add_argument("--data", type=__import__("pathlib").Path, default=HERE); a = p.parse_args()
    DATA = a.data.resolve()
    {"pools": pools, "gate": gate, "direct": direct}[a.phase]()
