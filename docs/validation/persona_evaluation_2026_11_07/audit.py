"""Offline material checks. A passing audit does not imply good model behavior."""
import gzip
import json
from pathlib import Path
import re
from benchmark import HERE, load, sha, save, verify, body_for, generation_messages, label_messages
from prepare_reviews import prepare
from mechanism import choose

def strings(obj):
    if isinstance(obj, str): yield obj
    elif isinstance(obj, dict):
        for value in obj.values(): yield from strings(value)
    elif isinstance(obj, list):
        for value in obj: yield from strings(value)

def main():
    checks = 0
    def check(value, message):
        nonlocal checks
        if not value: raise AssertionError(message)
        checks += 1
    protocol = verify(); check(True, "freeze")
    instruction_manifest = load(HERE / "review_instruction_manifest.json")
    for name, digest in instruction_manifest["files"].items():
        check(sha(HERE / name) == digest, "fixed common review instruction")
    data = load(HERE / "run/results.json")
    calls = json.loads(gzip.decompress((HERE / "run/raw_calls.json.gz").read_bytes()))
    check(data["protocol_sha256"] == sha(HERE / "protocol.json"), "protocol reference")
    check(len(calls) == data["actual_calls"] <= 36, "call count")
    by_call = {r["id"]: r for r in calls}
    check(len(by_call) == len(calls), "unique calls")
    cases = {c["id"]: c for c in load(HERE / "cases.json")}; profiles = {p["id"]: p for p in load(HERE / "profiles.json")}
    for row in calls:
        check(load(HERE / "run/calls" / (row["id"] + ".json")) == row, "archive equality")
        check("headers" not in row and "Authorization" not in json.dumps(row), "no authorization")
        check(row["status"] != "ok" or row["response"]["choices"][0]["finish_reason"] == "stop", "completion")
    for row in data["records"]:
        check(load(HERE / "run/records" / (row["id"] + ".json")) == row, "record equality")
        call = by_call[row["generation_call"]]
        check(call["request"] == body_for(call["stage"], generation_messages(cases[row["case"]], profiles[row["profile"]], row["method"])), "exact generation request")
        if row["method"] == "coverage" and row["candidates"]:
            messages, _, _ = label_messages(cases[row["case"]], row["id"], row["candidates"])
            check(by_call[row["label_call"]]["request"] == body_for("labels", messages), "exact label request")
            selected = choose(row["candidates"], row["labels"], profiles[row["profile"]])
            check(row["say"] == (selected["text"] if selected else None), "exact selected text")
    inputs, key = prepare(data)
    check(inputs == load(HERE / "reviews/inputs.json"), "review input reproduction")
    check(key == load(HERE / "review_key.json"), "review mapping reproduction")
    by_input = {r["id"]: r for r in inputs}
    allowed_styles = {"formal", "observational", "restrained", "lively", "colloquial", "metaphorical", "mechanical"}
    for coder in (1,2):
        ratings = load(HERE / f"reviews/ratings_{coder}.json")
        check(isinstance(ratings, list), "rating list")
        check(len(ratings) == len(inputs) and {r["id"] for r in ratings} == set(by_input), "rating ids")
        for r in ratings:
            check(set(r) == {"id","persona_fit","context_fit","grounding","boundary","style","attention","initiative","evidence"}, "rating schema")
            for field in ("persona_fit", "context_fit"):
                check(r[field] is None or (type(r[field]) is int and 1 <= r[field] <= 5), "ordinal score or unknown")
            check(r["grounding"] in {"supported","violation","uncertain"}, "grounding value")
            check(r["boundary"] in {"respected","violation","uncertain"}, "boundary value")
            check(isinstance(r["style"], list) and len(set(r["style"])) == len(r["style"]) and set(r["style"]) <= allowed_styles, "style descriptors")
            for field in ("attention", "initiative"):
                check(r[field] is None or (type(r[field]) is int and r[field] in (0,1,2)), "behavior value")
            source = by_input[r["id"]]; evidence = r["evidence"]
            check(set(evidence) == {"say_quote", "context_quote", "persona_rule", "reason"}, "evidence schema")
            check(bool(evidence["say_quote"]) and evidence["say_quote"] in source["say"], "reply quote")
            context_text = list(strings({k: source[k] for k in ("history", "user")}))
            check(evidence["context_quote"] == "" or any(evidence["context_quote"] in s for s in context_text), "context quote")
            check(bool(evidence["persona_rule"]) and any(evidence["persona_rule"] in s for s in strings(source["persona"]["spec"])), "persona quote")
            check(isinstance(evidence["reason"], str) and bool(evidence["reason"].strip()), "reason")
    historical = load(HERE / "historical_integrity.json")
    for study, expected in historical.items():
        folder = HERE.parent / study
        check(sha(folder / "artifact_manifest.json") == expected["manifest_sha256"], "old manifest unchanged")
        for row in load(folder / "artifact_manifest.json")["files"]:
            check(sha(folder / row["path"]) == row["sha256"], "old artifact unchanged")
    secret = re.compile(r"sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{16,}")
    for path in HERE.rglob("*"):
        if path.name != "artifact_manifest.json" and path.suffix in {".py", ".md", ".json", ".txt", ".svg"}:
            check(not secret.search(path.read_text(encoding="utf-8")), "credential pattern " + path.name)
    check(not secret.search(json.dumps(calls, ensure_ascii=False)), "credential pattern archive")
    report = {"passed": True, "checks": checks, "actual_api_calls": len(calls), "review_records_per_coder": len(inputs), "historical_studies_unchanged": list(historical)}
    save(HERE / "audit_report.json", report); print(json.dumps(report))

if __name__ == "__main__": main()
