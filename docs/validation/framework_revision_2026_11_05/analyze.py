"""Recompute focused verification and historical stream replay without API calls."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona.stream import StreamParser


def main():
    data = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    if not data["manifest"].get("completed_at"):
        raise SystemExit("Verification has not completed")
    facts = [r for r in data["records"] if r["experiment"] == "facts"]
    memories = [r for r in data["records"] if r["experiment"] == "memory"]
    groups = {}
    for condition in sorted({r["condition"] for r in facts}):
        selected = [r for r in facts if r["condition"] == condition]
        groups[condition] = {"matches": sum(r.get("corrected_strict_match") is True for r in selected),
                             "n": len(selected)}
    turns = [t for r in facts for t in r.get("turns", [])]
    turns += [t for r in memories for t in r.get("training", [])]
    turns += [p["turn"] for r in memories for p in r.get("probes", [])]
    probes = [p for r in memories for p in r.get("probes", [])]
    learned = [p for p in probes if p["condition"] == "learned"]
    empty = [p for p in probes if p["condition"] == "empty"]
    historical = ROOT / "docs/validation/followup_2026_11_05/raw_calls.json.gz"
    replay = []
    for row in json.loads(gzip.decompress(historical.read_bytes())):
        if row["route"] != "stream" or row["trial_id"].startswith("facts_vanilla"):
            continue
        parser = StreamParser(expression_only=row["trial_id"].startswith("facts_persona"))
        chunks = []
        for piece in row["chunks"]:
            chunks.extend(parser.feed(piece))
        parsed = parser.finish()
        visible = "".join(chunks)
        if parsed.say.startswith(visible):
            visible += parsed.say[len(visible):]
        replay.append({"trial_id": row["trial_id"], "matches": visible == parsed.say})
    result = {
        "scope": "Focused engineering verification; limited synthetic tasks, not a randomized version superiority study",
        "input_sha256": hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest(),
        "status": dict(Counter(r["status"] for r in data["records"])),
        "format": groups,
        "memory": {"replicates": len(memories),
                   "learned_strict_recall": sum(p["evaluation"]["recall_both"] for p in learned),
                   "learned_n": len(learned), "empty_unknown": sum(p["evaluation"]["unknown"] for p in empty),
                   "empty_n": len(empty), "quote_counts": [sum(len(j.get("evidence_quotes", [])) for j in r.get("judgments", [])) for r in memories],
                   "quotes_survive_rebuild": sum(r.get("quotes_survive_rebuild") is True for r in memories)},
        "online_turns": len(turns), "valid_turns": sum(t["valid"] is True for t in turns),
        "stream_equal": sum(t["stream_match"] for t in turns), "calls": data["manifest"]["calls_count"],
        "historical_replay": {"n": len(replay), "matches": sum(r["matches"] for r in replay), "records": replay},
        "offline_replay_sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "persona").glob("*.py"))}}
    (HERE / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("offline_replay_sources", "historical_replay")}, ensure_ascii=False))
    print("Historical stream replay:", result["historical_replay"]["matches"], "/", len(replay))


if __name__ == "__main__": main()
