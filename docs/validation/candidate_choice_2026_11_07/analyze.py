"""Offline analysis of archived outputs. Never calls a model or rewrites ratings."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import benchmark as b
from selection import distance, parse_choice, parse_pool, random_select, select

HERE = Path(__file__).resolve().parent
METHODS = ("program", "prompt", "direct", "random")
CODERS = ("1", "2")
FEATURES = ("attention", "initiative")
BOOTSTRAPS = 5000
SEED = 2026110719
COLORS = ("#2673b8", "#e08136")


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def ratio(n, d):
    return n / d if d else None


def digest_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.pending = []

    def check(self, condition, name):
        self.checks += 1
        if not condition:
            self.failures.append(name)

    def need(self, condition, name):
        if not condition:
            self.pending.append(name)

    def result(self):
        return {"checks": self.checks, "failures": self.failures,
                "pending": sorted(set(self.pending)),
                "status": "failed" if self.failures else "incomplete" if self.pending else "passed"}


def optional_json(path, default=None):
    return b.load(path) if path.exists() else default


def raw_calls(phase):
    path = HERE / phase / "raw_calls.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else []


def rating_valid(row):
    return (all(k in row and (row[k] is None or type(row[k]) is int and row[k] in (0, 1, 2)) for k in FEATURES)
            and all(type(row.get(k)) is int and 1 <= row[k] <= 5 for k in ("naturalness", "appropriateness"))
            and all(type(row.get(k)) is bool for k in ("unsupported", "boundary_violation"))
            and isinstance(row.get("reason"), str))


def load_ratings(keys, audit):
    all_ratings = {}
    key_ids = set(keys)
    for coder in CODERS:
        merged = {}
        for stage in ("pool", "direct"):
            path = HERE / "reviews" / f"ratings_{stage}_{coder}.json"
            rows = optional_json(path, [])
            audit.check(len({r["id"] for r in rows}) == len(rows), f"unique {stage} ratings coder {coder}")
            for row in rows:
                audit.check(row["id"] in key_ids, f"known rating {coder}/{row['id']}")
                audit.check(rating_valid(row), f"rating schema {coder}/{row['id']}")
                audit.check(row["id"] not in merged or merged[row["id"]] == row,
                            f"consistent repeated rating {coder}/{row['id']}")
                merged[row["id"]] = row
        all_ratings[coder] = merged
    return all_ratings


def call_content(call):
    try:
        return call["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def verify_data(protocol, cases, profiles, pools_data, main_data, calls, keys, ratings, gate, audit):
    for name, expected in protocol["hashes"].items():
        audit.check((HERE / name).exists() and b.sha(HERE / name) == expected, "frozen hash " + name)
    protocol_hash = b.sha(HERE / "protocol.json")
    pools = {r["case"]: r for r in pools_data.get("records", [])}
    audit.check(len(pools) == len(pools_data.get("records", [])), "unique pool case IDs")
    audit.need(bool(pools_data), "candidate phase results absent")
    if pools_data:
        audit.check(set(pools) == set(cases), "all twelve pool outcomes retained")
    all_calls = {r["id"]: r for r in calls}
    audit.check(len(all_calls) == len(calls), "unique actual call IDs")
    audit.check(len(calls) <= 84, "actual call budget <= 84")
    for phase, data in (("pools", pools_data), ("main", main_data)):
        if not data:
            continue
        actual = [c for c in calls if (c["stage"] == "pools") == (phase == "pools")]
        audit.check(data["protocol_sha256"] == protocol_hash, phase + " protocol hash")
        audit.check(data["calls_count"] == len(actual), phase + " actual call count")
        audit.check(data["planned_units"] == len(data["records"]), phase + " retains planned outcomes")
        manifest = optional_json(HERE / phase / "manifest.json", {})
        audit.check(manifest.get("protocol_sha256") == protocol_hash, phase + " manifest protocol")
        audit.check(manifest.get("retries") == 0 and manifest.get("server_seed") is None,
                    phase + " no retry and no server seed")
        if phase == "main":
            audit.check(manifest.get("pools_sha256") == b.sha(HERE / "pools/results.json"), "main pool hash")
            audit.check(manifest.get("gate_sha256") == b.sha(HERE / "gate.json"), "main gate hash")
    for call in calls:
        cid = call["id"]
        stage = call["stage"]
        if stage == "pools":
            case = cases[cid]
            expected_messages = b.messages("pools", case)
        else:
            case_id, profile_id, method = cid.split("-")
            audit.check(method == stage, "call stage " + cid)
            expected_messages = b.messages(stage, cases[case_id], profiles[profile_id],
                                           pools[case_id]["candidates"] if stage == "prompt" else None)
        expected_request = {"model": b.MODEL, "messages": expected_messages, "temperature": .7,
                            "max_tokens": 2400 if stage == "pools" else 128 if stage == "prompt" else 600,
                            "stream": False}
        audit.check(call.get("request") == expected_request, "entire reconstructed request " + cid)
        audit.check(call.get("endpoint") == b.ENDPOINT and call.get("timeout_s") == 90, "transport settings " + cid)
        audit.check(isinstance(call.get("elapsed_s"), (float, int)) and call["elapsed_s"] >= 0, "elapsed time " + cid)
        if call["status"] == "ok":
            audit.check(call_content(call) == call.get("output"), "response content preserved " + cid)
            audit.check(call["response"]["choices"][0].get("finish_reason") == "stop", "finish reason " + cid)
        usage = call.get("response", {}).get("usage", {})
        if all(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            audit.check(usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"], "token arithmetic " + cid)
        local = HERE / ("pools" if stage == "pools" else "main") / "calls" / (cid + ".json")
        if local.exists():
            audit.check(b.load(local) == call, "raw archive equals call record " + cid)
    for case_id, row in pools.items():
        call = all_calls.get(row.get("call_id"))
        if row["status"] == "ok":
            audit.check(call is not None and call["status"] == "ok", "successful pool call " + case_id)
            try:
                parsed = parse_pool(call["output"])
                audit.check(parsed == row["candidates"], "pool parse preserved " + case_id)
            except (ValueError, KeyError, TypeError):
                audit.check(False, "pool parse preserved " + case_id)
    records = main_data.get("records", [])
    if main_data:
        expected_ids = {f"{c}-{p}-{m}" for c in cases for p in profiles for m in METHODS}
        audit.check(len(records) == 144 and {r["id"] for r in records} == expected_ids, "all 144 main outcomes retained")
        audit.check(gate.get("proceed") is True, "main started only after passing gate")
    if gate:
        audit.check(gate.get("pools_sha256") == b.sha(HERE / "pools/results.json"), "gate pool hash")
        groups = optional_json(HERE / "reviews/pool_groups.json", [])
        expected_groups = {g["scene_id"]: {c["id"] for c in g["candidates"]} for g in groups}
        coder_passes = []
        for coder in CODERS:
            path = HERE / "reviews" / f"gate_ratings_{coder}.json"
            gate_rows = optional_json(path, [])
            audit.check(len(gate_rows) == len(expected_groups) and {r["scene_id"] for r in gate_rows} == set(expected_groups),
                        "gate group coverage coder " + coder)
            for row in gate_rows:
                expected = expected_groups.get(row["scene_id"], set())
                audit.check(len(row["candidates"]) == len(expected) and {c["id"] for c in row["candidates"]} == expected,
                            "gate candidate coverage " + coder + "/" + row["scene_id"])
                audit.check(all(type(c["semantic_usable"]) is bool for c in row["candidates"])
                            and type(row["at_least_two_distinct_usable"]) is bool,
                            "gate schema " + coder + "/" + row["scene_id"])
                if row["at_least_two_distinct_usable"]:
                    pair = row.get("distinct_pair", [])
                    audit.check(len(set(pair)) == 2 and set(pair).issubset({c["id"] for c in row["candidates"] if c["semantic_usable"]}),
                                "gate distinct evidence " + coder + "/" + row["scene_id"])
            usable = sum(c["semantic_usable"] for r in gate_rows for c in r["candidates"])
            distinct = sum(r["at_least_two_distinct_usable"] for r in gate_rows)
            passed = usable >= 65 and distinct >= 10
            coder_passes.append(passed)
            saved = next((r for r in gate.get("coders", []) if str(r["coder"]) == coder), {})
            audit.check(saved.get("usable") == usable and saved.get("distinct_pools") == distinct
                        and saved.get("passed") == passed and saved.get("ratings_sha256") == (b.sha(path) if path.exists() else None),
                        "recomputed gate coder " + coder)
        expected_proceed = len(pools) == 12 and all(p["status"] == "ok" for p in pools.values()) and all(coder_passes)
        audit.check(gate.get("proceed") == expected_proceed, "whole-batch gate recomputed")
    for row in records:
        rid = row["id"]
        audit.check(rid == f"{row['case']}-{row['profile']}-{row['method']}", "main identity " + rid)
        pool = pools[row["case"]]["candidates"]
        profile = profiles[row["profile"]]
        if row["status"] != "ok":
            audit.check(row.get("say") is None, "no fabricated text on failure " + rid)
            continue
        if row["method"] == "direct":
            call = all_calls.get(row["call_id"], {})
            audit.check(row["say"] == call.get("output") and call.get("status") == "ok", "direct original text " + rid)
        else:
            candidates = {c["id"]: c for c in pool}
            candidate = candidates.get(row["selected_id"])
            audit.check(candidate is not None and row["say"] == candidate["text"], "selected verbatim text " + rid)
            if candidate:
                audit.check(row["self_label_distance"] == distance(candidate, profile), "self tag distance " + rid)
            if row["method"] == "program":
                expected = select(pool, profile)
            elif row["method"] == "random":
                expected = random_select(pool, row["case"], row["profile"])
            else:
                expected = parse_choice(all_calls[row["call_id"]]["output"], pool)
            audit.check(expected == row["selected_id"], "selection replay " + rid)
    text_keys = {}
    for key_id, key in keys.items():
        pair = (key["case"], key["text"])
        audit.check(pair not in text_keys, "unique scene-text review unit " + key_id)
        text_keys[pair] = key_id
        audit.check(key["case"] in cases, "known review case " + key_id)
        for candidate_id in key.get("candidate_ids", []):
            short_id = candidate_id.split("-")[-1]
            pool_map = {c["id"]: c for c in pools[key["case"]]["candidates"]}
            audit.check(short_id in pool_map and pool_map[short_id]["text"] == key["text"], "review candidate mapping " + key_id + "/" + candidate_id)
        for output_id in key.get("output_ids", []):
            output = next((r for r in records if r["id"] == output_id), None)
            audit.check(output is not None and output["case"] == key["case"] and output["say"] == key["text"], "review output mapping " + output_id)
    for pool in pools.values():
        for candidate in pool["candidates"]:
            pair = (pool["case"], candidate["text"])
            audit.need(pair in text_keys, "candidate review key missing")
            if pair in text_keys:
                for coder in CODERS:
                    audit.need(text_keys[pair] in ratings[coder], "candidate ratings incomplete coder " + coder)
    for row in records:
        if row["status"] == "ok":
            pair = (row["case"], row["say"])
            audit.need(pair in text_keys, "output review key missing")
            if pair in text_keys:
                for coder in CODERS:
                    audit.need(text_keys[pair] in ratings[coder], "output ratings incomplete coder " + coder)
                audit.check(row["id"] in keys[text_keys[pair]].get("output_ids", []), "complete output-to-review mapping " + row["id"])
    for stage in ("pool", "direct"):
        inputs = optional_json(HERE / "reviews" / (stage + "_inputs.json"), [])
        for item in inputs:
            key = keys.get(item["id"])
            audit.check(key is not None, "known anonymous input " + item["id"])
            if key is not None:
                audit.check(item == {"id": item["id"], **b.context(cases[key["case"]]), "say": key["text"]},
                            "anonymous input source and text " + item["id"])
    return text_keys


def metric_row(row, profile, rating):
    out = dict(row)
    out["rated"] = rating is not None
    if rating:
        out.update({k: rating[k] for k in ("attention", "initiative", "naturalness", "appropriateness", "unsupported", "boundary_violation")})
        for k in FEATURES:
            out[k + "_error"] = abs(rating[k] - profile["target"][k]) if rating[k] is not None else None
        out["distance"] = sum(out[k + "_error"] for k in FEATURES) if all(rating[k] is not None for k in FEATURES) else None
    return out


def summarize_rows(rows, cases):
    delivered = [r for r in rows if r["status"] == "ok"]
    rated = [r for r in delivered if r["rated"]]
    scored = [r for r in rated if r.get("distance") is not None]
    boundary_rows = [r for r in rows if cases[r["case"]]["boundaries"]]
    boundary_rated = [r for r in rated if cases[r["case"]]["boundaries"]]
    unsupported = sum(r["unsupported"] for r in rated)
    violated = sum(r["boundary_violation"] for r in rated)
    return {"planned": len(rows), "delivered": len(delivered), "delivery_rate": ratio(len(delivered), len(rows)),
            "rated": len(rated), "rating_coverage_delivered": ratio(len(rated), len(delivered)),
            "behavior_scored": len(scored), "behavior_coverage_delivered": ratio(len(scored), len(delivered)),
            "mean_distance": mean(r["distance"] for r in scored),
            "mean_attention_error": mean(r.get("attention_error") for r in rated),
            "mean_initiative_error": mean(r.get("initiative_error") for r in rated),
            "mean_attention": mean(r.get("attention") for r in rated),
            "mean_initiative": mean(r.get("initiative") for r in rated),
            "mean_naturalness": mean(r["naturalness"] for r in rated),
            "mean_appropriateness": mean(r["appropriateness"] for r in rated),
            "unsupported_count": unsupported, "unsupported_rate_rated": ratio(unsupported, len(rated)),
            "boundary_violation_count": violated, "boundary_violation_rate_all_rated": ratio(violated, len(rated)),
            "boundary_planned": len(boundary_rows), "boundary_rated": len(boundary_rated),
            "boundary_violation_rate_boundary_rated": ratio(sum(r["boundary_violation"] for r in boundary_rated), len(boundary_rated)),
            "statuses": dict(Counter(r["status"] for r in rows))}


def paired_bootstrap(rows, comparator, metric, case_ids, seed_key):
    index = {(r["case"], r["profile"], r["method"]): r for r in rows}
    profile_ids = sorted({r["profile"] for r in rows})
    differences = {}
    pairs = 0
    for cid in sorted(case_ids):
        ds = []
        for pid in profile_ids:
            a = index.get((cid, pid, "program"), {}).get(metric)
            z = index.get((cid, pid, comparator), {}).get(metric)
            if a is not None and z is not None:
                ds.append(a - z)
        if ds:
            differences[cid] = mean(ds)
            pairs += len(ds)
    values = list(differences.values())
    result = {"direction": "program_minus_" + comparator, "metric": metric, "n_scenes": len(values),
              "n_profile_pairs": pairs, "scene_differences": differences,
              "mean": mean(values), "ci95": None, "resamples": BOOTSTRAPS}
    if values:
        arr = np.array(values, dtype=float)
        rng = np.random.default_rng(digest_seed(seed_key))
        draws = rng.choice(arr, size=(BOOTSTRAPS, len(arr)), replace=True).mean(axis=1)
        result["ci95"] = [float(x) for x in np.quantile(draws, [.025, .975])]
    return result


def agreement(keys, ratings):
    output = {}
    for subset, ids in (("all_unique_texts", set(keys)),
                        ("pool_unique_texts", {i for i, k in keys.items() if k.get("candidate_ids")}),
                        ("direct_only_unique_texts", {i for i, k in keys.items() if not k.get("candidate_ids")})):
        common = sorted(ids & ratings["1"].keys() & ratings["2"].keys())
        output[subset] = {"unique_texts": len(ids), "paired_ratings": len(common)}
        for field in ("attention", "initiative", "naturalness", "appropriateness", "unsupported", "boundary_violation"):
            valid = [i for i in common if ratings["1"][i][field] is not None and ratings["2"][i][field] is not None]
            exact = sum(ratings["1"][i][field] == ratings["2"][i][field] for i in valid)
            output[subset][field] = {"nonmissing_pairs": len(valid), "exact_count": exact,
                                      "exact_rate": ratio(exact, len(valid)),
                                      "mean_absolute_difference": mean(abs(ratings["1"][i][field] - ratings["2"][i][field]) for i in valid)}
    return output


def pool_diagnostics(pools, cases, profiles, text_keys, ratings):
    diagnostics = {"by_coder": {}, "by_scene": []}
    all_errors = {coder: {k: [] for k in FEATURES} for coder in CODERS}
    group_keys = {r["id"]: r["case"] for r in optional_json(HERE / "pool_group_key.json", [])}
    gate_ratings = {coder: {group_keys[r["scene_id"]]: r for r in optional_json(HERE / "reviews" / f"gate_ratings_{coder}.json", [])}
                    for coder in CODERS}
    for cid, pool in sorted(pools.items()):
        scene = {"case": cid, "has_boundary": bool(cases[cid]["boundaries"]),
                 "parse_status": pool["status"], "candidates": len(pool["candidates"]), "coders": {}}
        for coder in CODERS:
            items = []
            for candidate in pool["candidates"]:
                key = text_keys.get((cid, candidate["text"]))
                rating = ratings[coder].get(key)
                if not rating:
                    continue
                items.append((candidate, rating))
                for feature in FEATURES:
                    if rating[feature] is not None:
                        all_errors[coder][feature].append(abs(candidate[feature] - rating[feature]))
            coder_row = {"rated_candidates": len(items), "mean_naturalness": mean(r["naturalness"] for _, r in items),
                         "mean_appropriateness": mean(r["appropriateness"] for _, r in items),
                         "unsupported_count": sum(r["unsupported"] for _, r in items),
                         "boundary_violation_count": sum(r["boundary_violation"] for _, r in items),
                         "profile_coverage": {}}
            gate_row = gate_ratings[coder].get(cid)
            coder_row.update(semantic_usable_count=sum(c["semantic_usable"] for c in gate_row["candidates"]) if gate_row else None,
                             at_least_two_distinct_usable=gate_row["at_least_two_distinct_usable"] if gate_row else None)
            for pid, profile in profiles.items():
                scored = [(c, r) for c, r in items if all(r[k] is not None for k in FEATURES)]
                basic = [(c, r) for c, r in scored if not r["unsupported"] and not r["boundary_violation"] and r["appropriateness"] >= 3]
                usable_ids = {c["id"] for c in gate_row["candidates"] if c["semantic_usable"]} if gate_row else set()
                usable = [(c, r) for c, r in scored if text_keys.get((cid, c["text"])) in usable_ids]
                ds = [distance(r, profile) for _, r in scored]
                bs = [distance(r, profile) for _, r in basic]
                us = [distance(r, profile) for _, r in usable]
                coder_row["profile_coverage"][pid] = {
                    "posthoc_min_distance_all": min(ds) if ds else None,
                    "posthoc_min_distance_basic_quality": min(bs) if bs else None,
                    "posthoc_min_distance_semantic_usable": min(us) if us else None,
                    "exact_match_count": sum(d == 0 for d in ds), "within_one_count": sum(d <= 1 for d in ds),
                    "behavior_scored_candidates": len(ds), "basic_quality_scored_candidates": len(bs),
                    "semantic_usable_scored_candidates": len(us)}
            scene["coders"][coder] = coder_row
        diagnostics["by_scene"].append(scene)
    for coder in CODERS:
        diagnostics["by_coder"][coder] = {
            k: {"nonmissing_candidates": len(v), "mean_absolute_error": mean(v),
                "exact_count": sum(x == 0 for x in v), "exact_rate": ratio(sum(x == 0 for x in v), len(v))}
            for k, v in all_errors[coder].items()}
        candidate_items = [(cid, ratings[coder].get(text_keys.get((cid, c["text"]))))
                           for cid, pool in pools.items() for c in pool["candidates"]]
        candidate_items = [(cid, r) for cid, r in candidate_items if r is not None]
        diagnostics["by_coder"][coder]["quality_and_distribution"] = {}
        for subset in ("all", "no_boundary", "boundary"):
            rs = [r for cid, r in candidate_items if subset == "all" or bool(cases[cid]["boundaries"]) == (subset == "boundary")]
            diagnostics["by_coder"][coder]["quality_and_distribution"][subset] = {
                "rated_candidates": len(rs), "mean_naturalness": mean(r["naturalness"] for r in rs),
                "mean_appropriateness": mean(r["appropriateness"] for r in rs),
                "unsupported_count": sum(r["unsupported"] for r in rs),
                "boundary_violation_count": sum(r["boundary_violation"] for r in rs),
                "feature_counts": {k: {str(v): sum(r[k] == v for r in rs) for v in (0, 1, 2, None)} for k in FEATURES}}
    diagnostics["interpretation"] = "Self-tags are model proposals. Posthoc best-candidate distances are coverage ceilings, not deployed outcomes."
    diagnostics["coverage_summary"] = {}
    for subset in ("all", "no_boundary", "boundary"):
        rows = [s for s in diagnostics["by_scene"] if subset == "all" or s["has_boundary"] == (subset == "boundary")]
        diagnostics["coverage_summary"][subset] = {}
        for coder in CODERS:
            diagnostics["coverage_summary"][subset][coder] = {}
            for pid in profiles:
                pcs = [s["coders"][coder]["profile_coverage"][pid] for s in rows]
                ds = [r["posthoc_min_distance_all"] for r in pcs if r["posthoc_min_distance_all"] is not None]
                bs = [r["posthoc_min_distance_basic_quality"] for r in pcs if r["posthoc_min_distance_basic_quality"] is not None]
                us = [r["posthoc_min_distance_semantic_usable"] for r in pcs if r["posthoc_min_distance_semantic_usable"] is not None]
                diagnostics["coverage_summary"][subset][coder][pid] = {
                    "scenes": len(rows), "scenes_with_behavior_coding": len(ds),
                    "mean_posthoc_min_distance_all": mean(ds), "mean_posthoc_min_distance_basic_quality": mean(bs),
                    "mean_posthoc_min_distance_semantic_usable": mean(us), "scenes_with_semantic_usable_coding": len(us),
                    "scenes_with_exact_candidate": sum(d == 0 for d in ds),
                    "scenes_with_candidate_within_one": sum(d <= 1 for d in ds),
                    "scenes_with_basic_quality_exact_candidate": sum(d == 0 for d in bs),
                    "scenes_with_semantic_usable_exact_candidate": sum(d == 0 for d in us)}
    return diagnostics


def sensitivity(pools, cases, profiles, text_keys, ratings, subsets):
    result = {"seed_count": 100, "first_seed": SEED + 1,
              "interpretation": "Offline reuse of the same pools and annotations; not new samples or API requests.", "summaries": {}, "replicates": []}
    for seed in range(SEED + 1, SEED + 101):
        selected = []
        for cid, pool_row in sorted(pools.items()):
            pool = pool_row["candidates"]
            for pid, profile in profiles.items():
                candidate = random.Random(digest_seed(f"sensitivity/{seed}/{cid}/{pid}")).choice(pool) if pool else None
                selected.append({"case": cid, "profile": pid, "method": "random", "status": "ok" if candidate else "pool_unavailable",
                                 "say": candidate["text"] if candidate else None})
        replicate = {"seed": seed, "subsets": {}}
        for name, case_ids in subsets.items():
            replicate["subsets"][name] = {}
            for coder in CODERS:
                annotated = [metric_row(r, profiles[r["profile"]], ratings[coder].get(text_keys.get((r["case"], r["say"]))))
                             for r in selected if r["case"] in case_ids]
                m = summarize_rows(annotated, cases)
                replicate["subsets"][name][coder] = {k: m[k] for k in ("mean_distance", "mean_naturalness", "mean_appropriateness", "unsupported_count", "boundary_violation_count")}
        result["replicates"].append(replicate)
    for name in subsets:
        result["summaries"][name] = {}
        for coder in CODERS:
            result["summaries"][name][coder] = {}
            for metric in ("mean_distance", "mean_naturalness", "mean_appropriateness", "unsupported_count", "boundary_violation_count"):
                values = [r["subsets"][name][coder][metric] for r in result["replicates"] if r["subsets"][name][coder][metric] is not None]
                result["summaries"][name][coder][metric] = {"mean": mean(values), "min": min(values) if values else None,
                    "max": max(values) if values else None, "p05_p95": [float(x) for x in np.quantile(values, [.05, .95])] if values else None}
    return result


def usage(call):
    return call.get("response", {}).get("usage", {})


def costs(calls, records, cases):
    by_id = {c["id"]: c for c in calls}
    stages = {}
    for stage in ("pools", "prompt", "direct"):
        group = [c for c in calls if c["stage"] == stage]
        stages[stage] = {"calls": len(group), "successful_calls": sum(c["status"] == "ok" for c in group),
                         "usage_available": sum("total_tokens" in usage(c) for c in group),
                         "prompt_tokens": sum(usage(c).get("prompt_tokens", 0) for c in group),
                         "completion_tokens": sum(usage(c).get("completion_tokens", 0) for c in group),
                         "total_tokens": sum(usage(c).get("total_tokens", 0) for c in group),
                         "mean_request_elapsed_s": mean(c.get("elapsed_s") for c in group)}
    deployment = {}
    allocations = {}
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        full, shared, seconds, n_calls = [], [], [], []
        for row in rows:
            pool = by_id.get(row["case"])
            own = by_id.get(row.get("call_id"))
            required = ([pool] if method != "direct" else []) + ([own] if method in ("prompt", "direct") else [])
            if all(c is not None and "total_tokens" in usage(c) for c in required):
                full.append(sum(usage(c)["total_tokens"] for c in required))
                shared.append((usage(pool)["total_tokens"] / 9 if method != "direct" else 0)
                              + (usage(own)["total_tokens"] if method in ("prompt", "direct") else 0))
                seconds.append(sum(c["elapsed_s"] for c in required))
                n_calls.append(len(required))
        deployment[method] = {"planned_outputs": len(rows), "complete_cost_records": len(full),
                              "mean_full_total_tokens_per_attempt": mean(full),
                              "mean_sum_request_elapsed_s": mean(seconds), "mean_calls_per_attempt": mean(n_calls)}
        allocations[method] = {"mean_allocated_research_tokens_per_attempt": mean(shared), "allocated_research_tokens": sum(shared)}
    return {"actual_research": {"calls": len(calls), "total_tokens": sum(s["total_tokens"] for s in stages.values()),
                                 "usage_available_calls": sum(s["usage_available"] for s in stages.values()),
                                 "by_stage": stages,
                                 "requested_models": dict(Counter(c["request"]["model"] for c in calls)),
                                 "observed_response_models": dict(Counter(c.get("response", {}).get("model", "missing") for c in calls)),
                                 "statuses": dict(Counter(c["status"] for c in calls))},
            "candidate_calls_by_scene": [{"case": c["id"], "status": c["status"], "elapsed_s": c.get("elapsed_s"),
                                           **{k: usage(c).get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}}
                                          for c in sorted(calls, key=lambda c: c["id"]) if c["stage"] == "pools"],
            "standalone_cost_estimates": deployment, "shared_research_allocation": allocations,
            "notes": ["Standalone selection assumes one whole pool-generation call per user reply; prompt adds its selection call.",
                      "Research pool cost is allocated across 3 profiles x 3 candidate methods = 9 outputs, only for accounting.",
                      "Standalone timing sums observed request durations; it is not a fresh deployment or latency benchmark.",
                      "Missing token usage is reported, not imputed. Codex annotation labour and costs are not priced."]}


def expansion_decision(metrics, agreements, main_data, gate):
    if not main_data:
        return {"recommend_holdout_expansion": False if gate.get("proceed") is False else None,
                "status": "stopped_at_candidate_gate" if gate.get("proceed") is False else "awaiting_main", "criteria": []}
    criteria = []
    for coder in CODERS:
        no = metrics["no_boundary"]
        allm = metrics["all"]
        p, r, d = (no[m][coder] for m in ("program", "random", "direct"))
        for name, value, threshold, direction in (
            ("random_minus_program_distance", None if r["mean_distance"] is None or p["mean_distance"] is None else r["mean_distance"] - p["mean_distance"], .5, "ge"),
            ("program_minus_direct_naturalness", None if p["mean_naturalness"] is None or d["mean_naturalness"] is None else p["mean_naturalness"] - d["mean_naturalness"], -.25, "ge"),
            ("program_minus_direct_appropriateness", None if p["mean_appropriateness"] is None or d["mean_appropriateness"] is None else p["mean_appropriateness"] - d["mean_appropriateness"], -.25, "ge")):
            criteria.append({"coder": coder, "criterion": name, "observed": value, "threshold": threshold,
                             "pass": value >= threshold if value is not None else None})
        p_all = allm["program"][coder]
        fully_rated = p_all["rated"] == p_all["planned"]
        for field in ("unsupported_count", "boundary_violation_count"):
            criteria.append({"coder": coder, "criterion": "program_" + field, "observed": p_all[field], "threshold": 0,
                             "pass": p_all[field] == 0 if fully_rated else None})
    for field in FEATURES:
        value = agreements["pool_unique_texts"][field]["exact_rate"]
        criteria.append({"criterion": "pool_coder_exact_agreement_" + field, "observed": value, "threshold": .6,
                         "pass": value >= .6 if value is not None else None})
    complete = all(c["pass"] is not None for c in criteria)
    return {"recommend_holdout_expansion": all(c["pass"] for c in criteria) if complete else None,
            "status": "all_exploratory_gates_passed" if complete and all(c["pass"] for c in criteria) else "stop_expansion" if complete else "incomplete_ratings",
            "criteria": criteria, "interpretation": "Exploratory engineering gates, not significance or noninferiority tests; product adoption requires separate cost/quality judgment."}


def save_figure(fig, out, name):
    fig.savefig(out / "figures" / (name + ".png"), dpi=180, bbox_inches="tight", metadata={"Software": "candidate_choice offline analysis"})
    fig.savefig(out / "figures" / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def bars(ax, values, ylabel, ylim=None):
    x = np.arange(len(METHODS))
    for j, coder in enumerate(CODERS):
        nums = [values(m, coder) for m in METHODS]
        plotted = [np.nan if v is None else v for v in nums]
        positions = x + (j - .5) * .34
        ax.bar(positions, plotted, .32, label="Coder " + coder, color=COLORS[j])
        for pos, val in zip(positions, nums):
            if val is not None:
                ax.annotate(f"{val:.2f}", (pos, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, [m.title() for m in METHODS])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def candidate_figures(summary, out):
    """Four substantive candidate-stage plots; no invented main-comparison bars."""
    gate = summary["candidate_gate"]
    gd = {str(r["coder"]): r for r in gate.get("coders", [])}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), layout="constrained")
    for ax, metric, cutoff, maximum, title in (
        (axes[0], "usable", 65, 72, "Semantically usable candidates"),
        (axes[1], "distinct_pools", 10, 12, "Pools with distinct usable responses")):
        vals = [gd.get(c, {}).get(metric) for c in CODERS]
        ax.bar(["Coder 1", "Coder 2"], [np.nan if v is None else v for v in vals], color=COLORS, width=.55)
        ax.axhline(cutoff, color="#9a3636", linestyle="--", label=f"Frozen threshold: {cutoff}/{maximum}")
        ax.set_ylim(0, maximum * 1.08); ax.set_ylabel(f"Count / {maximum}"); ax.set_title(title)
        for i, value in enumerate(vals):
            if value is not None:
                ax.text(i, max(value / 2, .5), f"{value}/{maximum}", ha="center", color="white", fontweight="bold", fontsize=13)
        ax.legend(loc="upper left", fontsize=9); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    limited = [s["case"] + (" *" if s["has_boundary"] else "") for s in summary["pool_diagnostics"]["by_scene"]
               if all(s["coders"][c]["at_least_two_distinct_usable"] is False for c in CODERS)]
    detail = "Limited variety in both coders: " + ", ".join(limited) + " (* explicit user boundary)" if limited else "Candidate-stage diagnostic only"
    fig.suptitle("Whole-batch screening gate — no main comparison was executed\n" + detail)
    save_figure(fig, out, "fig1_pool_gate")

    diag = summary["pool_diagnostics"]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5), layout="constrained")
    x = np.arange(2)
    for j, coder in enumerate(CODERS):
        for ax, metric in ((axes[0], "exact_rate"), (axes[1], "mean_absolute_error")):
            vals = [diag["by_coder"][coder][f][metric] for f in FEATURES]
            ax.bar(x + (j - .5) * .34, [np.nan if v is None else v for v in vals], .32, color=COLORS[j], label="Coder " + coder)
            for pos, val in zip(x + (j - .5) * .34, vals):
                if val is not None:
                    ax.annotate(f"{val:.2f}", (pos, val), ha="center", textcoords="offset points", xytext=(0, 4), fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks(x, ["Attention", "Initiative"]); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    axes[0].set_title("Self-tags vs. independent coding"); axes[0].set_ylabel("Exact agreement"); axes[0].set_ylim(0, 1.12); axes[0].legend(fontsize=9)
    axes[1].set_title("Self-tag numerical error"); axes[1].set_ylabel("Mean absolute error (0-2)"); axes[1].set_ylim(0, 2.1)
    values = [summary["coder_agreement"]["pool_unique_texts"][f]["exact_rate"] for f in FEATURES]
    axes[2].bar(["Attention", "Initiative"], [np.nan if v is None else v for v in values], color="#4a9763", width=.55)
    for i, val in enumerate(values):
        if val is not None:
            axes[2].annotate(f"{val:.2f}", (i, val), ha="center", textcoords="offset points", xytext=(0, 4), fontsize=9)
    axes[2].axhline(.6, color="#9a3636", linestyle="--", label="Exploratory reliability gate: 0.60")
    axes[2].set_ylim(0, 1.12); axes[2].set_ylabel("Exact agreement"); axes[2].set_title("Coder 1 vs. coder 2"); axes[2].legend(fontsize=8)
    fig.suptitle("Seventy-two unique candidate texts — labels are not ground truth")
    save_figure(fig, out, "fig2_semantic_labels")

    scenes = diag["by_scene"]
    profile_ids = list(scenes[0]["coders"]["1"]["profile_coverage"]) if scenes else []
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), layout="constrained")
    cmap = matplotlib.colormaps["YlOrRd"].copy(); cmap.set_bad("#d7d7d7")
    for ax, coder in zip(axes, CODERS):
        values = [[s["coders"][coder]["profile_coverage"][pid]["posthoc_min_distance_semantic_usable"] for pid in profile_ids] for s in scenes]
        arr = np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)
        im = ax.imshow(arr, vmin=0, vmax=4, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(profile_ids)), profile_ids)
        ax.set_yticks(range(len(scenes)), [s["case"] + (" *" if s["has_boundary"] else "") for s in scenes])
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                ax.text(j, i, "NA" if val is None else str(val), ha="center", va="center", color="white" if val is not None and val >= 3 else "#222222")
        ax.set_title("Coder " + coder + ": minimum usable-candidate distance")
        ax.set_xlabel("Anonymous preference profile")
    fig.colorbar(im, ax=axes, ticks=range(5), label="Posthoc minimum distance (0-4); lower is closer", shrink=.8)
    fig.suptitle("Candidate coverage ceiling — not a deployed selection result\n* Explicit user boundary; gray = no scored usable candidate")
    save_figure(fig, out, "fig3_candidate_coverage")

    calls = summary["costs"]["candidate_calls_by_scene"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    x = np.arange(len(calls)); labels = [r["case"] for r in calls]
    inputs = [r["prompt_tokens"] if r["prompt_tokens"] is not None else np.nan for r in calls]
    outputs = [r["completion_tokens"] if r["completion_tokens"] is not None else np.nan for r in calls]
    axes[0].bar(x, inputs, label="Input tokens", color="#2673b8")
    axes[0].bar(x, outputs, bottom=inputs, label="Output tokens", color="#8ab7d9")
    axes[0].set_ylabel("Observed tokens per pool-generation request"); axes[0].set_title("Actual candidate-generation token usage"); axes[0].legend(fontsize=9)
    axes[1].bar(x, [r["elapsed_s"] if r["elapsed_s"] is not None else np.nan for r in calls], color="#e08136")
    axes[1].set_ylabel("Observed request elapsed time (seconds)"); axes[1].set_title("Per-request duration; three concurrent workers")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=45); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    actual = summary["costs"]["actual_research"]
    fig.suptitle(f"Actual study: {actual['calls']} requests, {actual['total_tokens']:,} total tokens — annotation labour unpriced")
    save_figure(fig, out, "fig4_pool_cost")


def figures(summary, out):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "candidate-choice-20261107"})
    metrics = summary["metrics"]
    has_main = summary["counts"]["main_planned_outputs"] > 0
    if not has_main:
        candidate_figures(summary, out)
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    if has_main:
        for ax, subset, title in zip(axes, ("no_boundary", "all"), ("Seven unconstrained scenes", "All twelve scenes")):
            bars(ax, lambda m, c: metrics[subset][m][c]["mean_distance"], "Observed behavior distance (0-4)", (0, 4.4))
            ax.set_title(title)
        axes[0].legend()
        fig.suptitle("Actual text vs. preference targets — lower is closer")
    else:
        for ax, feature in zip(axes, FEATURES):
            nums = [summary["pool_diagnostics"]["by_coder"][c][feature]["mean_absolute_error"] for c in CODERS]
            ax.bar(["Coder 1", "Coder 2"], [np.nan if v is None else v for v in nums], color=COLORS)
            ax.set_title(feature.title() + ": candidate self-tag error")
            ax.set_ylabel("Mean absolute error (0-2)"); ax.set_ylim(0, 2.1)
        fig.suptitle("Candidate stage only — main comparison not run")
    save_figure(fig, out, "fig1_behavior")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    if has_main:
        specs = (("mean_naturalness", "Naturalness (1-5)", (0, 5.7)),
                 ("mean_appropriateness", "Appropriateness (1-5)", (0, 5.7)),
                 ("unsupported_rate_rated", "Unsupported-claim rate / rated outputs", (0, 1.12)),
                 ("boundary_violation_rate_boundary_rated", "Boundary violation rate / rated boundary outputs", (0, 1.12)))
        for ax, (metric, label, limit) in zip(axes.flat, specs):
            bars(ax, lambda m, c, k=metric: metrics["all"][m][c][k], label, limit)
        axes[0, 0].legend()
        fig.suptitle("Independent text coding — all planned conditions retained")
    else:
        scene_rows = summary["pool_diagnostics"]["by_scene"]
        for ax, metric, label in zip(axes.flat,
            ("mean_naturalness", "mean_appropriateness", "unsupported_count", "boundary_violation_count"),
            ("Candidate naturalness", "Candidate appropriateness", "Unsupported candidates", "Boundary-violating candidates")):
            for j, coder in enumerate(CODERS):
                values = [s["coders"][coder][metric] for s in scene_rows]
                ax.plot(range(len(values)), [np.nan if v is None else v for v in values], marker="o", color=COLORS[j], label="Coder " + coder)
            ax.set_xticks(range(len(scene_rows)), [s["case"] for s in scene_rows], rotation=45)
            ax.set_title(label); ax.grid(alpha=.2)
        axes[0, 0].legend()
        fig.suptitle("Candidate pool quality — main comparison not run")
    save_figure(fig, out, "fig2_quality")

    cost = summary["costs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    if has_main:
        x = np.arange(4)
        for j, (block, field, label, color) in enumerate((
            ("standalone_cost_estimates", "mean_full_total_tokens_per_attempt", "Whole pool per reply", "#2673b8"),
            ("shared_research_allocation", "mean_allocated_research_tokens_per_attempt", "Shared-pool study allocation", "#a8bfd3"))):
            vals = [cost[block][m][field] for m in METHODS]
            axes[0].bar(x + (j - .5) * .36, [np.nan if v is None else v for v in vals], .34, label=label, color=color)
        axes[0].set_xticks(x, [m.title() for m in METHODS]); axes[0].set_ylabel("Mean total tokens per attempted reply")
        axes[0].legend(fontsize=8); axes[0].set_title("Deployment estimate vs. experimental sharing")
    else:
        axes[0].text(.5, .5, "Main comparison not run\nNo per-method deployment estimate", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()
    stages = cost["actual_research"]["by_stage"]
    names = ("pools", "prompt", "direct")
    vals = [stages[s]["total_tokens"] for s in names]
    axes[1].bar(["Pool generation", "Prompt selection", "Direct reply"], vals, color=("#2673b8", "#e08136", "#4a9763"))
    for i, s in enumerate(names):
        axes[1].annotate(f"{stages[s]['calls']} calls\n{vals[i]:,} tokens", (i, vals[i]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(vals + [1]) * 1.25); axes[1].set_ylabel("Actual total tokens")
    axes[1].set_title("Observed study expenditure")
    fig.suptitle("Token usage includes inputs and outputs; annotation labour unpriced")
    save_figure(fig, out, "fig3_cost")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    x = np.arange(2)
    for j, coder in enumerate(CODERS):
        vals = [summary["pool_diagnostics"]["by_coder"][coder][k]["exact_rate"] for k in FEATURES]
        axes[0].bar(x + (j - .5) * .34, [np.nan if v is None else v for v in vals], .32, label="Self-tags vs coder " + coder, color=COLORS[j])
    axes[0].set_xticks(x, ["Attention", "Initiative"]); axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Exact agreement / nonmissing pairs"); axes[0].legend(fontsize=9)
    axes[0].set_title("Model self-tags are not behavioral truth")
    vals = [summary["coder_agreement"]["pool_unique_texts"][k]["exact_rate"] for k in FEATURES]
    axes[1].bar(["Attention", "Initiative"], [np.nan if v is None else v for v in vals], color="#4a9763")
    axes[1].axhline(.6, linestyle="--", color="#8a3e35", label="Exploratory gate: 0.60")
    axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("Exact agreement / nonmissing pairs")
    axes[1].set_title("Independent coder agreement on unique pool texts"); axes[1].legend(fontsize=9)
    save_figure(fig, out, "fig4_label_reliability")


def analyze(out):
    audit = Audit()
    protocol = b.load(HERE / "protocol.json")
    cases = {r["id"]: r for r in b.load(HERE / "cases.json")}
    profiles = {r["id"]: r for r in b.load(HERE / "profiles.json")}
    pools_data = optional_json(HERE / "pools/results.json", {})
    main_data = optional_json(HERE / "main/results.json", {})
    gate = optional_json(HERE / "gate.json", {})
    calls = raw_calls("pools") + raw_calls("main")
    key_rows = optional_json(HERE / "review_key.json", [])
    audit.check(len({r["id"] for r in key_rows}) == len(key_rows), "unique review IDs")
    keys = {r["id"]: r for r in key_rows}
    ratings = load_ratings(keys, audit)
    text_keys = verify_data(protocol, cases, profiles, pools_data, main_data, calls, keys, ratings, gate, audit)
    pools = {r["case"]: r for r in pools_data.get("records", [])}
    records = main_data.get("records", [])
    subsets = {"all": set(cases), "no_boundary": {c for c in cases if not cases[c]["boundaries"]},
               "boundary": {c for c in cases if cases[c]["boundaries"]}}
    annotated = {coder: [metric_row(row, profiles[row["profile"]], ratings[coder].get(text_keys.get((row["case"], row.get("say")))))
                          for row in records] for coder in CODERS}
    metrics, pairs = {}, {}
    for subset, ids in subsets.items():
        metrics[subset] = {method: {coder: summarize_rows([r for r in annotated[coder] if r["case"] in ids and r["method"] == method], cases)
                                   for coder in CODERS} for method in METHODS}
        pairs[subset] = {coder: {comparator: {metric: paired_bootstrap(annotated[coder], comparator, metric, ids, f"{SEED}/{subset}/{coder}/{comparator}/{metric}")
                      for metric in ("distance", "naturalness", "appropriateness")} for comparator in ("random", "direct", "prompt")} for coder in CODERS}
    agreements = agreement(keys, ratings)
    summary = {"study_id": protocol["study_id"], "protocol_sha256": b.sha(HERE / "protocol.json"),
               "counts": {"scenes": len(cases), "unconstrained_scenes": len(subsets["no_boundary"]),
                          "boundary_scenes": len(subsets["boundary"]), "profiles": len(profiles),
                          "pool_units": len(pools), "pool_parse_successes": sum(p["status"] == "ok" for p in pools.values()),
                          "candidate_texts": sum(len(p["candidates"]) for p in pools.values()),
                          "protocol_planned_main_outputs": 144, "main_stage_executed": bool(main_data),
                          "main_planned_outputs": len(records), "main_delivered_outputs": sum(r["status"] == "ok" for r in records),
                          "unique_review_texts": len(keys), "actual_calls": len(calls)},
               "candidate_gate": gate, "metrics": metrics, "paired_scene_bootstrap": pairs,
               "coder_agreement": agreements, "pool_diagnostics": pool_diagnostics(pools, cases, profiles, text_keys, ratings),
               "random_sensitivity": sensitivity(pools, cases, profiles, text_keys, ratings, subsets),
               "costs": costs(calls, records, cases),
               "expansion_decision": expansion_decision(metrics, agreements, main_data, gate),
               "notes": ["Preference distance is an engineering operationalization, not a validated psychological scale.",
                         "Coders evaluate actual unique texts independently; duplicated method outputs do not add independent evidence.",
                         "Scene-paired bootstrap intervals describe this 12-scene development pilot, not held-out generalization.",
                         "No hypothesis tests or human noninferiority claims are made.",
                         "When the candidate gate stops the study, main-method effects are unobserved; posthoc candidate coverage and random sensitivity are diagnostic only.",
                         "User boundaries can legitimately reduce semantic variety; failure of a small engineering gate is not general persona failure."]}
    out.mkdir(parents=True, exist_ok=True)
    b.save(out / "summary.json", summary)
    b.save(out / "audit_report.json", audit.result())
    b.save(out / "analysis_environment.json", {"python": platform.python_version(), "platform": platform.platform(),
        "matplotlib": matplotlib.__version__, "numpy": np.__version__, "analysis_sha256": b.sha(Path(__file__)),
        "bootstrap_resamples": BOOTSTRAPS, "seed": SEED, "font": "DejaVu Sans", "offline": True})
    figures(summary, out)
    return summary, audit.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    summary, audit = analyze(args.out)
    print(json.dumps({"counts": summary["counts"], "audit": audit,
                      "expansion": summary["expansion_decision"]["status"]}, ensure_ascii=False))
    return 1 if audit["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
