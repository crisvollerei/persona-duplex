"""Method-hidden, persona-visible review inputs. No API calls."""
import random
from benchmark import HERE, load, save, context

def visible_profile(profile):
    return {k: profile[k] for k in ("name", "spec")}

def prepare(data):
    cases = {c["id"]: c for c in load(HERE / "cases.json")}
    profiles = {p["id"]: p for p in load(HERE / "profiles.json")}
    unique = {}
    for row in data["records"]:
        candidates = row["candidates"] if row["method"] == "coverage" else ([{"id": "direct", "text": row["say"]}] if row["say"] else [])
        for candidate in candidates:
            key = (row["case"], row["profile"], candidate["text"])
            if key not in unique:
                unique[key] = {"kind": "online", "case": row["case"], "profile": row["profile"],
                    "say": candidate["text"], "references": [], "visible": {**context(cases[row["case"]]),
                    "persona": visible_profile(profiles[row["profile"]]), "say": candidate["text"]}}
            unique[key]["references"].append({"record_id": row["id"], "candidate_id": candidate["id"]})
    rows = list(unique.values())
    for fixture in load(HERE / "fixtures.json"):
        rows.append({"kind": "fixture", "fixture_id": fixture["id"], "profile": fixture["profile"],
            "visible": {**context(fixture), "persona": visible_profile(profiles[fixture["profile"]]), "say": fixture["say"]}})
    random.Random(20261107311).shuffle(rows)
    inputs, key = [], []
    for i, row in enumerate(rows, 1):
        id = f"R{i:03}"; inputs.append({"id": id, **row.pop("visible")}); key.append({"id": id, **row})
    return inputs, key

if __name__ == "__main__":
    if (HERE / "review_key.json").exists(): raise SystemExit("Review inputs already exist")
    inputs, key = prepare(load(HERE / "run/results.json"))
    save(HERE / "reviews/inputs.json", inputs); save(HERE / "review_key.json", key)
    print(f"Prepared {len(inputs)} records; method hidden, intended persona visible")
