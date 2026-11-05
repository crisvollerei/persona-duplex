"""Independently reconcile the completed study artifacts without API requests."""
from __future__ import annotations

import gzip
import hashlib
import importlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = HERE / "implementation_snapshot"


def study_stream_parser():
    """Replay historical streams with the implementation that produced them."""
    package_dir = SNAPSHOT / "persona"
    if not package_dir.is_dir():
        sys.path.insert(0, str(ROOT))
        return importlib.import_module("persona.stream").StreamParser
    package_name = "_persona_duplex_study_snapshot"
    spec = importlib.util.spec_from_file_location(
        package_name, package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(package_name + ".stream").StreamParser


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def final_conclusion(text):
    found = re.findall(r"结论\s*[:：]\s*([^\n]+)", text)
    return re.sub(r"[\s*`，。.!！]", "", found[-1]) if found else None


def factual_answer(text, case):
    answer = final_conclusion(text)
    if answer is None:
        return None
    if case["kind"] == "number":
        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:元|%)?", answer.replace(",", ""))
        return bool(match and Decimal(match[1]) == Decimal(case["gold"][0]))
    if case["kind"] == "time":
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", answer)
        return bool(match and tuple(map(int, match.groups())) in
                    {tuple(map(int, gold.split(":"))) for gold in case["gold"]})
    return answer in case["gold"]


def audit():
    data = json.loads((HERE / "experiment_results.json").read_text(encoding="utf-8"))
    if not data["manifest"].get("completed_at"):
        raise SystemExit("Study still running; audit requires manifest.completed_at.")
    plan = json.loads((HERE / "execution_plan.json").read_text(encoding="utf-8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    raw_path = HERE / "raw_calls.json.gz"
    with gzip.open(raw_path, "rt", encoding="utf-8") as source:
        calls = json.load(source)
    records, manifest = data["records"], data["manifest"]
    checks, failures = [], []

    def check(name, valid, detail=None):
        checks.append({"name": name, "passed": bool(valid)})
        if not valid:
            failures.append({"name": name, "detail": detail})

    plan_by_id = {job["trial_id"]: job for job in plan}
    ids = [record["trial_id"] for record in records]
    check("528 unique assigned and retained trials", len(plan) == len(plan_by_id) == len(ids) == len(set(ids)) == 528 and set(ids) == set(plan_by_id))
    check("all planned experimental cells retained", Counter(r["experiment"] for r in records) ==
          {"facts": 288, "traits": 96, "memory": 64, "repair": 16, "displacement": 32, "checker": 32})
    check("protocol hash", digest(protocol) == manifest["protocol_hash"] and protocol == data["protocol"])
    check("execution plan hash", digest(plan) == manifest["jobs_hash"])
    check("source manifest hash", digest(manifest["sources"]) == manifest["source_hash"])
    check("compressed raw calls hash", hashlib.sha256(raw_path.read_bytes()).hexdigest() == manifest["raw_calls_sha256"])
    current_differences = []
    for relative, expected in manifest["sources"].items():
        current = HERE / "run_experiments.py" if relative == "runner" else ROOT / relative
        source = SNAPSHOT / relative if relative.startswith("persona/") and SNAPSHOT.is_dir() else current
        check("frozen source " + relative, hashlib.sha256(source.read_bytes()).hexdigest() == expected)
        if relative.startswith("persona/") and hashlib.sha256(current.read_bytes()).hexdigest() != expected:
            current_differences.append(relative)
    StreamParser = study_stream_parser()

    call_ids = [call["global_call_id"] for call in calls]
    check("globally unique raw calls", len(call_ids) == len(set(call_ids)) == manifest["calls_count"])
    by_trial = defaultdict(list)
    for call in calls:
        by_trial[call["trial_id"]].append(call)
        if call["route"] == "stream" and call["status"] == "ok":
            check("raw chunks equal raw output " + call["global_call_id"], "".join(call["chunks"]) == call["output"])
        check("finished call status " + call["global_call_id"], call["status"] in ("ok", "error"))
        settings = call["request"]["settings"]
        check("requested model and temperature " + call["global_call_id"], settings["model"] == manifest["settings"]["model"] and settings["temperature"] == manifest["settings"]["temperature"])

    fixed_gold = {"budget": [str(2400-1200-300-400-200)], "percentage": [str(1200*120//100*80//100)],
                  "fraction": ["37.5"], "midnight": ["00:25", "0:25"], "syllogism": ["不能推出"],
                  "city_correction": ["苏州", "苏州市"], "meal_history": ["已吃"], "unknown": ["不可确定"]}
    trait_configs = []
    history_trials, replayed_streams, replayed_deliveries = 0, 0, 0
    delivery_equal, protocol_valid, protocol_inapplicable = [], [], 0
    counterexamples = []
    private_completed = ROOT / "data" / manifest.get("run_id", "no-private-run-id") / "completed"
    durable_checked = 0
    for record in records:
        trial_id = record["trial_id"]
        job, trace = plan_by_id[trial_id], by_trial[trial_id]
        check("trial metadata " + trial_id, all(record.get(key) == job.get(key) for key in ("experiment", "condition", "case_id", "replicate", "order")))
        check("record links exact raw calls " + trial_id, record["call_ids"] == [call["global_call_id"] for call in trace])
        if private_completed.is_dir():
            durable = json.loads((private_completed / (trial_id + ".json")).read_text(encoding="utf-8"))
            durable_calls = durable.pop("calls")
            published = {key: value for key, value in record.items() if key != "call_ids"}
            stripped_trace = [{key: value for key, value in call.items() if key not in ("trial_id", "global_call_id")} for call in trace]
            check("durable trial copy " + trial_id, durable == published and durable_calls == stripped_trace)
            durable_checked += 1
        if record["status"] == "error":
            check("error evidence retained " + trial_id, bool(record.get("error")) and bool(trace))
        if "config" in record:
            check("configuration hash " + trial_id, digest(record["config"]) == record["config_hash"])
        if record["experiment"] == "traits":
            cfg = json.loads(json.dumps(record["config"]))
            trait = job["config_changed"]
            spec = cfg["role"]["traits"][trait["name"]]
            check("intended factor value " + trial_id, spec["anchor"] == spec["value"] == trait["value"])
            spec["anchor"] = spec["value"] = .5
            trait_configs.append(cfg)
        turns = record["turns"]
        for turn in turns:
            delivered = "".join(turn["delivered_chunks"])
            equality = delivered == turn["say"]
            delivery_equal.append(equality)
            check("delivery equality flag " + trial_id, turn["stream_match"] == equality)
            if not equality:
                counterexamples.append({"trial_id": trial_id, "delivered_text": delivered,
                                        "stored_say": turn["say"], "protocol_valid": turn["valid"],
                                        "calls_count": turn["calls_count"], "call_ids": record["call_ids"]})
            check("turn call count " + trial_id, turn["calls_count"] == len(trace))
            if turn.get("validity") is not None:
                check("validity flag " + trial_id, turn["valid"] == turn["validity"]["valid"])
                protocol_valid.append(turn["valid"])
            else:
                protocol_inapplicable += 1
            streams = [call for call in trace if call["route"] == "stream"]
            if len(streams) == 1 and streams[0]["status"] == "ok":
                if job.get("mode") == "vanilla":
                    replayed = streams[0]["output"]
                    replayed_delivery = replayed
                else:
                    parser = StreamParser(expression_only=job.get("mode") == "persona")
                    replayed_parts = []
                    for piece in streams[0]["chunks"]:
                        replayed_parts.extend(parser.feed(piece))
                    replayed = parser.finish().say
                    replayed_delivery = "".join(replayed_parts)
                    if replayed.startswith(replayed_delivery):
                        replayed_delivery += replayed[len(replayed_delivery):]
                check("stream parser replay " + trial_id, replayed == turn["say"])
                check("callback delivery replay " + trial_id, replayed_delivery == delivered)
                replayed_streams += 1
                replayed_deliveries += 1
        history = job.get("case", {}).get("history")
        if history and trace:
            first_input = json.dumps(trace[0]["request"]["messages"], ensure_ascii=False)
            check("provided history reaches first model call " + trial_id, all(message["content"] in first_input for message in history))
            history_trials += 1
        if record["experiment"] == "facts":
            case = job["case"]
            check("independent gold " + trial_id, case["gold"] == fixed_gold[case["id"]])
            if turns:
                turn = turns[-1]
                say = factual_answer(turn["say"], case)
                direct = factual_answer((turn.get("head") or {}).get("direct", ""), case) if turn.get("head") else None
                check("independent fact score " + trial_id, record["evaluation"]["say_correct"] == (say is True) and record["evaluation"]["direct_correct"] == direct)
            else:
                check("failed facts retained as incorrect " + trial_id, record["evaluation"]["say_correct"] is False)
        elif record["experiment"] == "checker" and record["status"] == "ok":
            check("checker gold and score " + trial_id, record["checker_input"]["gold"] == ("inconsistent" if job["wrong"] else "consistent") and record["evaluation"]["checker_correct"] == (record["checker_output"].get("verdict") == record["checker_input"]["gold"]))
        elif record["experiment"] == "memory" and turns:
            case, text = job["case"], final_conclusion(turns[-1]["say"])
            if case["id"] == "drink":
                recalled = text in {"豆浆", "无糖豆浆", "热豆浆", "热的无糖豆浆", "热无糖豆浆"}
            elif case["id"] in ("focus", "walk"):
                recalled = text in {case["target"], case["target"]+"分钟"}
            else:
                recalled = text == case["target"]
            absent = job["condition"] in ("empty", "irrelevant")
            check("independent memory score " + trial_id, record["evaluation"]["recall"] == recalled and record["evaluation"]["memory_correct"] == (text == "未知" if absent else recalled))
    check("single-factor trait manipulation", len(trait_configs) == 96 and all(cfg == trait_configs[0] for cfg in trait_configs))
    report = {"audit_type": "independent offline reconciliation", "passed": not failures,
              "interpretation": "Passed checks establish artifact reconciliation, not universal protocol validity or delivery equality.",
              "trials": len(records), "raw_calls": len(calls), "checks": len(checks),
              "failures": failures, "history_trials_checked": history_trials, "streams_replayed": replayed_streams,
              "callback_deliveries_replayed": replayed_deliveries,
              "stream_delivery_equality": {"equal": sum(delivery_equal), "total": len(delivery_equal),
                                           "unequal": len(delivery_equal)-sum(delivery_equal)},
              "local_protocol_validity": {"valid": sum(protocol_valid), "applicable": len(protocol_valid),
                                          "invalid": len(protocol_valid)-sum(protocol_valid),
                                          "vanilla_not_applicable": protocol_inapplicable},
              "delivery_counterexamples": counterexamples,
              "implementation": {"replay_source": "implementation_snapshot/persona" if SNAPSHOT.is_dir() else "current persona",
                                     "study_source_hash": manifest["source_hash"],
                                     "current_framework_matches_study": not current_differences,
                                     "current_framework_differences": current_differences},
              "private_durable_trials_checked": durable_checked,
              "experiment_counts": dict(Counter(record["experiment"] for record in records)),
              "status_counts": dict(Counter(record["status"] for record in records)),
              "error_trials": [{"trial_id": r["trial_id"], "error_type": r["error"]["type"]} for r in records if r["status"] == "error"],
              "checks_detail": checks}
    (HERE / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checks_detail"}, ensure_ascii=False, indent=2))
    return not failures


if __name__ == "__main__":
    sys.exit(0 if audit() else 1)
