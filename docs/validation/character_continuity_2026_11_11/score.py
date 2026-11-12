"""Unblind one completed human ballot locally. Free-text evidence is never exported."""
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
STUDY = "character_continuity_2026_11_11"
METHODS = ("framework", "examples")
FIELDS = ("fit", "continue_chat")
CHOICES = {"A", "B", "both", "neither", "uncertain"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        d = dict(pairs)
        require(len(d) == len(pairs), "Repeated JSON key; original input must be unambiguous.")
        return d
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def score(ballot_path=None, blind_path=None, registration_path=None):
    if ballot_path is None:
        return dict(study=STUDY, status="no_human_ballot", respondents=0)
    h = read(ballot_path)
    require(isinstance(h, dict) and h.get("study") == STUDY and h.get("source") == "human_self_report",
            "Expected a self-reported human ballot for this study.")
    digest = hashlib.sha256(blind_path.read_bytes()).hexdigest()
    require(h.get("packet_sha256") == digest, "Ballot does not refer to the supplied blind packet.")
    scenes = read(blind_path)["scenes"]
    ids = [s["id"] for s in scenes]
    cb = read(registration_path)["codebook"]
    require(len(ids) == len(set(ids)) == 3 and set(cb) == set(ids), "Packet/registration scene IDs do not match.")
    ratings = h.get("ratings")
    require(isinstance(ratings, dict) and set(ratings) == set(ids),
            "All three scene IDs are required; incomplete ballots are not scored.")
    per_scene = {}
    counts = {f: dict(exclusive={m: 0 for m in METHODS}, both=0, neither=0, uncertain=0,
                      selected_alone_or_together={m: 0 for m in METHODS}) for f in FIELDS}
    for i in ids:
        require(isinstance(cb[i], dict) and set(cb[i]) == {"A", "B"} and set(cb[i].values()) == set(METHODS),
                "Every scene needs a valid one-to-one A/B mapping.")
        row = ratings[i]
        require(isinstance(row, dict) and isinstance(row.get("evidence", ""), str), "Invalid ballot row.")
        per_scene[i] = {}
        for f in FIELDS:
            value = row.get(f)
            require(isinstance(value, str) and value in CHOICES, f"{i}: missing or invalid {f}.")
            chosen = cb[i].get(value, value)
            per_scene[i][f] = dict(blind_choice=value, decoded=chosen)
            if chosen in METHODS:
                counts[f]["exclusive"][chosen] += 1
                counts[f]["selected_alone_or_together"][chosen] += 1
            else:
                counts[f][chosen] += 1
                if chosen == "both":
                    for method in METHODS:
                        counts[f]["selected_alone_or_together"][method] += 1
    return dict(study=STUDY, status="one_complete_human_ballot", respondents=1, scenes=len(ids),
                packet_sha256=digest, by_scene=per_scene, by_question=counts,
                interpretation="每项各有三个场景。单选表示本题更倾向一段，不表示另一段一定不符合；双选给两路各记一次入选。"
                               "两题分别统计。无法判断、都不符合或都不愿意各自保留，不计为方法胜负。")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ballot", type=Path, help="Only pass an actual human response, never a model rating.")
    p.add_argument("--blind", type=Path, default=HERE/"observed/blind.json")
    p.add_argument("--registration", type=Path, default=HERE/"observed/registration.json")
    p.add_argument("--out", type=Path, help="Optional path for anonymous summary; source files cannot be overwritten.")
    args = p.parse_args()
    summary = score(args.ballot, args.blind, args.registration)
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        inputs = [Path(__file__), args.blind, args.registration]
        if args.ballot is not None:
            inputs.append(args.ballot)
        require(args.out.resolve() not in {f.resolve() for f in inputs}, "Refusing to overwrite an input.")
        if args.out.exists():
            require(args.out.read_bytes() == text.encode(), "Refusing to replace a different existing artifact.")
        else:
            args.out.write_bytes(text.encode())
    print(text, end="")


if __name__ == "__main__":
    main()
