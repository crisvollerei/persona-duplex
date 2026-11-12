"""Release counts for the first completed form only. Never export the listener's text."""
import argparse
from collections import Counter
import json
from pathlib import Path

import listener as l

DEST = l.HERE / "first_case.json"
FIELDS = {"free": ("heard", "unclear", "missing"),
          "action": ("target", "foil", "unclear", "other", "missing"),
          "motive": ("target", "foil", "unclear", "other", "missing")}


def aggregate(rows):
    return {"n": len(rows), **{stage: {key: sum(row[stage] == key for row in rows) for key in keys}
                               for stage, keys in FIELDS.items()}}


def prior_scores():
    folder = l.PREV / "observed"
    ids = json.loads((folder / "key.json").read_bytes())
    sampled = {item["source"] for item in l.form(1)}
    primary = {s for s in ids.values() if s.split(".")[1] in ("G", "F", "S")}
    scores = {}
    for label, filename in (("research", "reading.json"), ("model", "model_coding.json")):
        data = json.loads((folder / filename).read_bytes())
        lookup = {s: data[b]["score"] for b, s in ids.items()}
        def counts(keys):
            c = Counter(lookup[s] for s in keys)
            assert sum(c.values()) == len(keys) and set(c) <= {0, 1, 2}
            return [c[i] for i in (0, 1, 2)]
        scores[label] = {"first_form": counts(sampled), "eligible_archive": counts(primary)}
    return scores


def summarize(root):
    d = l.load(root, 1); answers = d["answers"]
    if len(answers) != 18 or not d.get("receipt") or not d.get("submitted_at"):
        raise ValueError("第一卷尚无完整提交")
    items = l.form(1)
    rows = []
    for i, item in enumerate(items):
        row = {"arm": item["arm"], "role": item["role"], "valid": item["valid"]}
        for j, phase in enumerate(l.PHASES):
            step = 6*j+i
            a = answers[step]
            assert (a["step"], a["phase"], a["position"]) == (step, phase, i)
            choice = a["choice"]
            if phase == "free":
                assert type(a["familiar"]) is bool and choice in ("heard", "unclear")
                assert (choice == "heard") == bool(a["note"].strip())
                row["familiar"] = a["familiar"]
            else:
                assert choice in l.CHOICES
                if choice != "unclear":
                    assert a["quote"].strip() and a["quote"] in item["say"]
                if choice in ("A", "B"):
                    choice = "target" if choice == item[phase]["target"] else "foil"
            row[phase] = choice
        rows.append(row)
    novel = [r for r in rows if not r["familiar"]]
    return {"scope": "form_01_only", "completed_forms": 1, "answered_prompts": 18,
            "novel": aggregate(novel), "familiar": aggregate([r for r in rows if r["familiar"]]),
            "novel_protocol_valid": aggregate([r for r in novel if r["valid"]]),
            "role_arm": {role+"."+arm: aggregate([r for r in novel if r["role"] == role and r["arm"] == arm])
                         for role in l.NAMES for arm in ("G", "F", "S")},
            "prior_scores_0_1_2": prior_scores()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=l.PRIVATE)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    assert l.register() == json.loads((l.HERE / "registration.json").read_bytes())
    result = summarize(args.root)
    if args.write:
        l.write(DEST, result)
    else:
        assert result == json.loads(DEST.read_bytes())
    print("First completed form: aggregate counts verified.")
