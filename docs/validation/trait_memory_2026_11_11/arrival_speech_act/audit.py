"""Independent read-only audit of the observer archive. Does not open human ballots."""
from collections import Counter, defaultdict
from hashlib import sha256
import gzip
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / "observed"


def load(file):
    return json.loads(file.read_bytes())


def check(value, context):
    if not value:
        raise ValueError(f"Archive audit failed: {context}")


def digest(obj):
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256(data).hexdigest()


def raw_quotes(value):
    if isinstance(value, dict):
        if type(value.get("turn")) is int and isinstance(value.get("quote"), str):
            yield (value["turn"], value["quote"])
        for child in value.values():
            yield from raw_quotes(child)
    elif isinstance(value, list):
        for child in value:
            yield from raw_quotes(child)


def audit():
    cases = load(HERE / "cases.json")
    indexed = {c["id"]: c for c in cases}
    pairs = defaultdict(list)
    for case in cases:
        pairs[case["pair"]].append(case)
    check(len(indexed) == len(cases) == 16 and len(pairs) == 8, "case/pair counts")
    for pair, group in pairs.items():
        check(len(group) == 2, pair)
        first, second = group
        check(first["target"] == second["target"] and first["focus_turn"] == second["focus_turn"], pair)
        before, after = first["turns"], second["turns"]
        check([t["speaker"] for t in before] == [t["speaker"] for t in after], pair)
        check([n for n, (u, v) in enumerate(zip(before, after), 1) if u != v] == [first["focus_turn"]], pair)

    reg, pred = load(OUT / "registration.json"), load(OUT / "predictions.json")
    check(bool(reg["hashes"]), "empty source fingerprint list")
    for name, checksum in reg["hashes"].items():
        path = (ROOT / name).resolve()
        check(path.is_relative_to(ROOT) and "human" not in path.name.lower(), "unsafe source path")
        check(sha256(path.read_bytes()).hexdigest() == checksum, name)
    with gzip.open(OUT / "calls.jsonl.gz", "rb") as fp:
        rows = [json.loads(line) for line in fp]
    calls = {c["id"]: c for c in rows}
    methods, samples = ("direct", "scoped"), ("A", "A2")
    expected = {".".join(keys) for keys in itertools.product(indexed, methods, samples)}
    check(len(rows) == len(calls) == len(reg["order"]) == 64, "duplicate or missing slot")
    check(set(reg["order"]) == set(pred) == set(calls) == expected, "slot coverage")
    receipts, systems = set(), defaultdict(set)
    models, failures, act_ok, state_ok, quotes = (Counter() for _ in range(5))
    for key in expected:
        i, method, sample = key.split(".")
        record, parsed, case = calls[key], pred[key], indexed[i]
        messages = record["messages"]
        check(len(messages) == 2 and [msg["role"] for msg in messages] == ["system", "user"], key)
        check(record["params"] == reg["model"] and parsed["request"] == digest(messages), key)
        systems[method].add(messages[0]["content"])
        observed = json.loads(messages[1]["content"])
        desired = dict(location="城北书展的寄件台", target=case["target"], focus_turn=case["focus_turn"],
                       dialogue=[dict(turn=n, speaker=t["speaker"], text=t["text"])
                                 for n, t in enumerate(case["turns"], 1)])
        check(observed == desired, f"{key} model-facing fields")
        check(calls[f"{i}.{method}.A"]["messages"] == calls[f"{i}.{method}.A2"]["messages"], f"{key} repeat")
        check(calls[f"{i}.direct.{sample}"]["messages"][1] == calls[f"{i}.scoped.{sample}"]["messages"][1],
              f"{key} matched user payload")
        if record["ok"]:
            response = record["response"]
            rid = response["id"]
            check(isinstance(rid, str) and rid and rid not in receipts, f"{key} provider ID")
            receipts.add(rid)
            models[response["model"]] += 1
        else:
            failures[record.get("error", "unknown")] += 1
            check(not record["raw"], f"{key} missing response has completion content")
        if parsed.get("act") is None and parsed.get("state") is None:
            continue
        check(record["ok"], f"{key} accepted without provider result")
        raw = json.loads(record["raw"])
        source = set(raw_quotes(raw))
        clips = parsed["evidence"]
        check(isinstance(clips, list), f"{key} evidence must be a list")
        for clip in clips:
            n, span = clip["turn"], clip["quote"]
            check(type(n) is int and 1 <= n <= len(case["turns"]) and isinstance(span, str) and span.strip()
                  and span in case["turns"][n - 1]["text"] and (n, span) in source, f"{key} unsupported quotation")
        act_ok[method] += int(parsed.get("act") is not None)
        state_ok[method] += int(parsed.get("state") is not None)
        quotes[method] += len(clips)
    check(all(len(systems[m]) == 1 for m in methods), "system text changed between items")
    return {"archive_slots": len(rows), "distinct_dialogues": len(indexed), "minimal_pairs": len(pairs),
            "provider_ids": len(receipts), "service_models": dict(models), "failed_calls": dict(failures),
            "valid_act_outputs": dict(act_ok), "valid_state_outputs": dict(state_ok),
            "verified_spans": dict(quotes),
            "identical_input_repeats": len(cases) * len(methods),
            "matched_user_messages_across_methods": len(cases) * len(samples),
            "registered_source_files": len(reg["hashes"])}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
