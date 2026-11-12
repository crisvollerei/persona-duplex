"""Offline descriptive counts for the anonymous ballots; never pool human and AI choices."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from score import CHOICES, FIELDS, METHODS, STUDY, read, require

OUT = Path(__file__).resolve().parent / "observed"
ORIGINS = {"human": ("humans", r"H[0-9]+"), "ai_reference": ("ai_reference", r"M[0-9]+")}


def count(rows):
    """Each row is one respondent in one scene, with two separate answers."""
    result = {}
    for field in FIELDS:
        counts = {v: 0 for v in (*METHODS, "both", "neither", "uncertain")}
        for row in rows:
            counts[row[field]] += 1
        result[field] = dict(choices=counts, selected_alone_or_together={
            method: counts[method] + counts["both"] for method in METHODS})
    same = sum(row["fit"] == row["continue_chat"] for row in rows)
    return dict(respondent_scene_pairs=len(rows), by_question=result,
                same_option=same, different_option=len(rows)-same)


def aggregate(data, blind, reg, packet_hash):
    require(data.get("study") == STUDY and data.get("packet_sha256") == packet_hash,
            "Study or blind-packet fingerprint mismatch.")
    ids = [s["id"] for s in blind["scenes"]]
    codebook = reg["codebook"]
    require(len(ids) == len(set(ids)) == 3 and set(ids) == set(codebook), "Invalid scene IDs.")
    for i in ids:
        require(set(codebook[i]) == {"A", "B"} and set(codebook[i].values()) == set(METHODS),
                f"{i}: expected a one-to-one A/B mapping.")
    sources = data["sources"]
    require(isinstance(sources, dict) and set(sources) == set(ORIGINS), "Invalid source manifest.")
    provenance = {}
    for origin, (_, pattern) in ORIGINS.items():
        require(isinstance(sources[origin], list), "Expected a list of anonymous IDs.")
        for name in sources[origin]:
            require(isinstance(name, str) and re.fullmatch(pattern, name) and name not in provenance,
                    "Duplicate anonymous ID or ID from the wrong source.")
            provenance[name] = origin

    decoded = {}
    require(isinstance(data["ballots"], list), "Expected a list of ballots.")
    for ballot in data["ballots"]:
        name, origin, ratings = ballot["id"], ballot["origin"], ballot["ratings"]
        require(name in provenance and name not in decoded and origin == provenance[name],
                "Missing, reused or contradictory ballot provenance.")
        require(isinstance(ratings, dict) and set(ratings) == set(ids), "Incomplete or unknown scene.")
        decoded[name] = {}
        for i in ids:
            item = ratings[i]
            require(isinstance(item, dict) and all(isinstance(item.get(f), str) and item[f] in CHOICES
                    for f in FIELDS), f"{name}/{i}: missing or invalid answer.")
            decoded[name][i] = {f: codebook[i].get(item[f], item[f]) for f in FIELDS}
    require(set(decoded) == set(provenance), "A listed respondent has no ballot.")

    report = dict(study=STUDY, packet_sha256=packet_hash, scenes=ids)
    for origin, (group, _) in ORIGINS.items():
        names = sources[origin]
        report[group] = dict(respondents=len(names), by_respondent={
            n: {"decoded":decoded[n], **count(list(decoded[n].values()))} for n in names},
            by_scene={i:count([decoded[n][i] for n in names]) for i in ids},
            overall=count([decoded[n][i] for n in names for i in ids]))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ballots", type=Path, default=OUT / "ballots.json")
    p.add_argument("--blind", type=Path, default=OUT / "blind.json")
    p.add_argument("--registration", type=Path, default=OUT / "registration.json")
    p.add_argument("--out", type=Path, default=OUT / "ratings_summary.json")
    p.add_argument("--check", action="store_true", help="Verify the saved summary without changing it.")
    a = p.parse_args()
    require(a.out.resolve() not in {p.resolve() for p in (a.ballots, a.blind, a.registration, Path(__file__))},
            "Output must not be an input.")
    summary = aggregate(read(a.ballots), read(a.blind), read(a.registration),
                        hashlib.sha256(a.blind.read_bytes()).hexdigest())
    if a.check:
        require(read(a.out) == summary, "Saved results differ from recomputation.")
    else:
        a.out.write_bytes((json.dumps(summary, ensure_ascii=False, indent=2)+"\n").encode())
    print(f"{summary['humans']['respondents']} human ballots; {summary['ai_reference']['respondents']} AI "
          f"reference ballot(s), counted separately.")


if __name__ == "__main__":
    main()
