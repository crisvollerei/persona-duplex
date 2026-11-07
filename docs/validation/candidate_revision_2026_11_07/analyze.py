"""Offline component analysis: generation quality, label validity and selection."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import benchmark as b
from mechanism import blind_candidates, validate_annotations, unavailable_annotations, choose
from selection import parse_pool

HERE = Path(__file__).resolve().parent
GENERATIONS = ("baseline", "coverage")
SOURCES = ("self", "evidence")
FEATURES = ("attention", "initiative")
CODERS = ("1", "2")
SEED = 2026110727
BOOTSTRAPS = 5000
COLORS = ("#2673b8", "#e08136")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def rate(n, d):
    return n / d if d else None


def known(value):
    return type(value) is int and value in (0, 1, 2)


def distance(row, profile):
    return sum(abs(row[k] - profile["target"][k]) for k in FEATURES) if all(known(row.get(k)) for k in FEATURES) else None


def usable(rating):
    return (rating is not None and rating["naturalness"] >= 3 and rating["appropriateness"] >= 4
            and not rating["unsupported"] and not rating["boundary_violation"])


def valid_rating(row):
    return (all(k in row and (row[k] is None or known(row[k])) for k in FEATURES)
            and all(type(row.get(k)) is int and 1 <= row[k] <= 5 for k in ("naturalness", "appropriateness"))
            and all(type(row.get(k)) is bool for k in ("unsupported", "boundary_violation"))
            and isinstance(row.get("reason"), str))


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def check(self, condition, name):
        self.checks += 1
        if not condition:
            self.failures.append(name)

    def result(self):
        return {"checks": self.checks, "status": "passed" if not self.failures else "failed", "failures": self.failures}


def read_calls():
    import json
    return json.loads(gzip.decompress((HERE / "run/raw_calls.json.gz").read_bytes()))


def response_content(call):
    try:
        return call["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def audit_sources(data, calls, cases, profiles, keys, ratings, audit):
    protocol = b.load(HERE / "protocol.json")
    for name, expected in protocol["hashes"].items():
        audit.check((HERE / name).exists() and b.sha(HERE / name) == expected, "frozen hash " + name)
    audit.check(b.sha(HERE / "common_prompt.txt") == protocol["baseline_origin"]["common_sha256"], "baseline common prompt provenance")
    audit.check(b.sha(HERE / "baseline_prompt.txt") == protocol["baseline_origin"]["pool_prompt_sha256"], "baseline candidate prompt provenance")
    audit.check(data["protocol_sha256"] == b.sha(HERE / "protocol.json"), "results protocol hash")
    manifest = b.load(HERE / "run/manifest.json")
    audit.check(manifest["protocol_sha256"] == data["protocol_sha256"], "manifest protocol hash")
    audit.check(manifest["planned_calls"] == 32 and manifest["server_seed"] is None and manifest["retries"] == 0,
                "call budget, no server seed, no retry")
    audit.check(len(calls) == data["actual_calls"] <= 32, "actual call count")
    by_call = {r["id"]: r for r in calls}
    audit.check(len(by_call) == len(calls), "unique actual call IDs")
    records = data["records"]
    pools = {r["id"]: r for r in records}
    expected_pools = {f"{c}-{g}" for c in cases for g in GENERATIONS}
    audit.check(len(records) == 16 and set(pools) == expected_pools, "all 16 planned pool outcomes")
    for call in calls:
        cid = call["id"]
        pool_id = cid.rsplit("-", 1)[0]
        pool = pools[pool_id]
        case = cases[pool["case"]]
        if call["stage"] == "pool":
            messages = b.pool_messages(case, pool["generation"])
        else:
            messages, hidden, mapping = b.label_messages(case, pool_id, pool["candidates"])
            import json
            payload = json.loads(messages[1]["content"])
            audit.check(set(payload) == {"context", "candidates"}, "no plan or profile in label payload " + cid)
            audit.check(all(set(c) == {"id", "text"} for c in payload["candidates"]), "no self tags in label input " + cid)
            audit.check(set(mapping) == {f"X{i:02}" for i in range(1, 7)} and set(mapping.values()) == {c["id"] for c in pool["candidates"]},
                        "bijective shuffled aliases " + cid)
            audit.check(mapping == pool["label_id_mapping"], "stored alias mapping " + cid)
        audit.check(call["request"] == b.body_for(call["stage"], messages), "complete request reconstruction " + cid)
        audit.check(call["endpoint"] == b.ENDPOINT and call["timeout_s"] == 90, "transport settings " + cid)
        audit.check(isinstance(call.get("elapsed_s"), (int, float)) and call["elapsed_s"] >= 0, "request duration " + cid)
        if call["status"] == "ok":
            audit.check(call.get("output") == response_content(call), "response output preserved " + cid)
            audit.check(call["response"]["choices"][0]["finish_reason"] == "stop", "complete generation " + cid)
        usage = call.get("response", {}).get("usage", {})
        if all(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            audit.check(usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"], "token arithmetic " + cid)
        path = HERE / "run/calls" / (cid + ".json")
        if path.exists():
            audit.check(b.load(path) == call, "archive equals raw call file " + cid)
    for pool in records:
        pid = pool["id"]
        audit.check(pid == pool["case"] + "-" + pool["generation"], "pool identity " + pid)
        if pool["status"] != "ok":
            audit.check(not pool["candidates"] and not pool["selections"], "failed pool has no fabricated selections " + pid)
            continue
        call = by_call[pool["pool_call"]]
        audit.check(call["status"] == "ok" and parse_pool(call["output"]) == pool["candidates"], "original pool parse " + pid)
        hidden, mapping = blind_candidates(pid, pool["candidates"])
        label_call = by_call.get(pool["label_call"])
        if label_call and label_call["status"] == "ok":
            try:
                expected_labels = validate_annotations(label_call["output"], hidden, mapping)
            except (ValueError, TypeError, KeyError):
                expected_labels = unavailable_annotations(pool["candidates"], "invalid_labels")
                audit.check(pool["label_status"] == "invalid_labels", "invalid label status " + pid)
        else:
            expected_labels = unavailable_annotations(pool["candidates"], pool["label_status"])
        audit.check(pool["labels"] == expected_labels, "validated labels and unknowns reconstructed " + pid)
        expected_selection_keys = {(s, p) for s in SOURCES for p in profiles}
        audit.check(len(pool["selections"]) == 6 and {(r["label_source"], r["profile"]) for r in pool["selections"]} == expected_selection_keys,
                    "six selection outcomes " + pid)
        for row in pool["selections"]:
            labels = pool["candidates"] if row["label_source"] == "self" else pool["labels"]
            expected = choose(pool["candidates"], labels, profiles[row["profile"]])
            audit.check(row["selected_id"] == (expected["candidate_id"] if expected else None)
                        and row["say"] == (expected["text"] if expected else None)
                        and row["status"] == ("ok" if expected else "abstained"),
                        "deterministic verbatim choice " + pid + "/" + row["label_source"] + "/" + row["profile"])
    pairs = {}
    references = {}
    for key_id, key in keys.items():
        pair = (key["case"], key["text"])
        audit.check(pair not in pairs, "unique scene-text coding unit " + key_id)
        pairs[pair] = key_id
        for reference in key["references"]:
            ref = reference["pool_id"], reference["candidate_id"]
            audit.check(ref not in references, "unique candidate review reference " + "/".join(ref))
            references[ref] = key_id
            pool = pools[ref[0]]
            candidate = next(c for c in pool["candidates"] if c["id"] == ref[1])
            audit.check(pool["case"] == key["case"] and candidate["text"] == key["text"], "review source text " + key_id)
    audit.check(set(references) == {(p["id"], c["id"]) for p in records for c in p["candidates"]}, "all generated texts anonymously mapped")
    inputs = b.load(HERE / "reviews/inputs.json")
    audit.check(len(inputs) == len(keys) and {r["id"] for r in inputs} == set(keys), "anonymous input coverage")
    for row in inputs:
        key = keys[row["id"]]
        audit.check(row == {"id": row["id"], **b.context(cases[key["case"]]), "say": key["text"]}, "only context and actual text in review " + row["id"])
    for coder in CODERS:
        audit.check(set(ratings[coder]) == set(keys), "complete independent coding " + coder)
        for key_id, row in ratings[coder].items():
            audit.check(valid_rating(row), "rating schema " + coder + "/" + key_id)
    return references


def ci(values, key):
    values = {k: v for k, v in values.items() if v is not None}
    nums = list(values.values())
    result = {"scene_values": values, "n_scenes": len(nums), "mean": mean(nums), "ci95": None,
              "resamples": BOOTSTRAPS, "unit": "scene"}
    if nums:
        seed = int.from_bytes(hashlib.sha256(f"{SEED}/{key}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        draws = rng.choice(np.array(nums, dtype=float), size=(BOOTSTRAPS, len(nums)), replace=True).mean(axis=1)
        result["ci95"] = [float(v) for v in np.quantile(draws, [.025, .975])]
    return result


def summarize_quality(rows, planned):
    rs = [r["rating"] for r in rows if r["rating"] is not None]
    count_usable = sum(usable(r) for r in rs)
    return {"planned_candidates": planned, "generated_candidates": sum(r["candidate"] is not None for r in rows),
            "rated_candidates": len(rs), "rating_coverage_planned": rate(len(rs), planned),
            "usable_count": count_usable, "usable_rate_planned": rate(count_usable, planned),
            "usable_rate_rated": rate(count_usable, len(rs)),
            "mean_naturalness": mean(r["naturalness"] for r in rs),
            "mean_appropriateness": mean(r["appropriateness"] for r in rs),
            "unsupported_count": sum(r["unsupported"] for r in rs),
            "boundary_violation_count": sum(r["boundary_violation"] for r in rs),
            "feature_distribution": {k: {str(v): sum(r[k] == v for r in rs) for v in (0, 1, 2, None)} for k in FEATURES}}


def label_metric(rows, source, feature, planned):
    model_known = sum(known(r[source].get(feature)) for r in rows)
    blind_known = sum(r["rating"] is not None and known(r["rating"][feature]) for r in rows)
    both = [r for r in rows if known(r[source].get(feature)) and r["rating"] is not None and known(r["rating"][feature])]
    exact = sum(r[source][feature] == r["rating"][feature] for r in both)
    matrix = [[0] * 4 for _ in range(4)]
    for row in rows:
        predicted = row[source].get(feature)
        actual = row["rating"].get(feature) if row["rating"] else None
        matrix[predicted if known(predicted) else 3][actual if known(actual) else 3] += 1
    return {"planned_candidates": planned, "nonunknown_labels": model_known,
            "nonunknown_coverage_planned": rate(model_known, planned),
            "blind_coding_nonunknown": blind_known, "joint_nonunknown": len(both),
            "exact_matches": exact, "exact_rate_planned": rate(exact, planned),
            "exact_rate_given_model_nonunknown": rate(exact, model_known),
            "exact_rate_given_joint_nonunknown": rate(exact, len(both)),
            "mean_absolute_error_joint_nonunknown": mean(abs(r[source][feature] - r["rating"][feature]) for r in both),
            "confusion": {"rows": "model validated label", "columns": "independent blind coding",
                          "labels": [0, 1, 2, "unknown_or_missing"], "counts": matrix}}


def summarize_choices(rows):
    delivered = [r for r in rows if r["status"] == "ok"]
    scored = [r for r in delivered if r["distance"] is not None]
    rated = [r for r in delivered if r["rating"] is not None]
    return {"planned_choices": len(rows), "delivered_choices": len(delivered), "delivery_rate": rate(len(delivered), len(rows)),
            "abstained_choices": sum(r["status"] == "abstained" for r in rows),
            "unavailable_choices": sum(r["status"] not in ("ok", "abstained") for r in rows),
            "behavior_scored_choices": len(scored), "behavior_coverage_planned": rate(len(scored), len(rows)),
            "mean_distance": mean(r["distance"] for r in scored), "mean_regret": mean(r["regret"] for r in scored),
            "rated_choices": len(rated), "usable_choices": sum(usable(r["rating"]) for r in rated),
            "mean_naturalness": mean(r["rating"]["naturalness"] for r in rated),
            "mean_appropriateness": mean(r["rating"]["appropriateness"] for r in rated),
            "unsupported_count": sum(r["rating"]["unsupported"] for r in rated),
            "boundary_violation_count": sum(r["rating"]["boundary_violation"] for r in rated)}


def actual_costs(calls, records):
    by_call = {c["id"]: c for c in calls}
    usage = lambda c: c.get("response", {}).get("usage", {})
    actual = {"calls": len(calls), "usage_available_calls": sum("total_tokens" in usage(c) for c in calls),
              "prompt_tokens": sum(usage(c).get("prompt_tokens", 0) for c in calls),
              "completion_tokens": sum(usage(c).get("completion_tokens", 0) for c in calls),
              "total_tokens": sum(usage(c).get("total_tokens", 0) for c in calls),
              "requested_models": dict(Counter(c["request"]["model"] for c in calls)),
              "observed_models": dict(Counter(c.get("response", {}).get("model", "missing") for c in calls)),
              "by_stage": {}}
    for stage in ("pool", "labels"):
        cs = [c for c in calls if c["stage"] == stage]
        actual["by_stage"][stage] = {"calls": len(cs), "total_tokens": sum(usage(c).get("total_tokens", 0) for c in cs),
                                    "mean_observed_request_elapsed_s": mean(c["elapsed_s"] for c in cs)}
    standalone = {}
    for generation in GENERATIONS:
        standalone[generation] = {}
        ps = [p for p in records if p["generation"] == generation]
        for source in SOURCES:
            costs = []
            for pool in ps:
                required = [by_call.get(pool["pool_call"])]
                if source == "evidence":
                    required.append(by_call.get(pool["label_call"]))
                if all(c is not None and "total_tokens" in usage(c) for c in required):
                    costs.append({"case": pool["case"], "total_tokens": sum(usage(c)["total_tokens"] for c in required),
                                  "pool_tokens": usage(required[0])["total_tokens"],
                                  "label_tokens": usage(required[1])["total_tokens"] if source == "evidence" else 0,
                                  "sum_observed_request_elapsed_s": sum(c["elapsed_s"] for c in required), "calls": len(required)})
            standalone[generation][source] = {"planned_pools": len(ps), "complete_cost_records": len(costs),
                "mean_full_tokens_per_reply": mean(c["total_tokens"] for c in costs),
                "mean_pool_tokens": mean(c["pool_tokens"] for c in costs), "mean_label_tokens": mean(c["label_tokens"] for c in costs),
                "mean_calls_per_reply": mean(c["calls"] for c in costs),
                "mean_reconstructed_request_time_s": mean(c["sum_observed_request_elapsed_s"] for c in costs), "by_scene": costs}
    return {"actual_research": actual, "standalone_reconstructed_cost": standalone,
            "notes": ["One self reply bears a complete pool call; one evidence reply bears a complete pool and label call.",
                      "The full-pool costs are accounting reconstructions from 16 observed pools, not 96 new deployments.",
                      "Summed two-stage durations are reconstructed, not measured user end-to-end latency.",
                      "Independent Codex coding labour is not priced; its cost is not assumed to be zero."]}


def coder_agreement(keys, ratings):
    result = {"unique_texts": len(keys)}
    for feature in ("attention", "initiative", "naturalness", "appropriateness", "unsupported", "boundary_violation"):
        ids = [i for i in keys if ratings["1"][i][feature] is not None and ratings["2"][i][feature] is not None]
        exact = sum(ratings["1"][i][feature] == ratings["2"][i][feature] for i in ids)
        result[feature] = {"joint_nonunknown": len(ids), "exact_count": exact, "exact_rate": rate(exact, len(ids)),
                           "mean_absolute_difference": mean(abs(ratings["1"][i][feature] - ratings["2"][i][feature]) for i in ids)}
    return result


def build_metrics(records, cases, profiles, references, ratings):
    pools = {p["id"]: p for p in records}
    subsets = {"all": list(cases), "open": [c for c in cases if not cases[c]["boundaries"]],
               "boundary": [c for c in cases if cases[c]["boundaries"]]}
    candidate_rows, choice_rows, pool_stats = {}, {}, {}
    for coder in CODERS:
        candidate_rows[coder], choice_rows[coder], pool_stats[coder] = [], [], {}
        for cid in cases:
            for generation in GENERATIONS:
                pool = pools[cid + "-" + generation]
                cs = {c["id"]: c for c in pool["candidates"]}
                ls = {r["id"]: r for r in pool["labels"]}
                rows = []
                for i in range(1, 7):
                    candidate_id = f"C{i:02}"
                    candidate = cs.get(candidate_id)
                    rating = ratings[coder].get(references.get((pool["id"], candidate_id)))
                    row = {"case": cid, "pool_id": pool["id"], "generation": generation, "candidate_id": candidate_id,
                           "candidate": candidate, "rating": rating, "self": candidate or {}, "evidence": ls.get(candidate_id, {})}
                    rows.append(row); candidate_rows[coder].append(row)
                coverage = {}
                for profile_id, profile in profiles.items():
                    ds = [distance(r["rating"], profile) for r in rows if r["rating"] is not None]
                    ds = [d for d in ds if d is not None]
                    us = [distance(r["rating"], profile) for r in rows if usable(r["rating"])]
                    us = [d for d in us if d is not None]
                    coverage[profile_id] = {"usable_exact_target": 0 in us, "posthoc_min_distance_all_codable": min(ds) if ds else None,
                                            "posthoc_min_distance_usable": min(us) if us else None,
                                            "behavior_codable_candidates": len(ds), "usable_codable_candidates": len(us)}
                pool_stats[coder][pool["id"]] = {"case": cid, "generation": generation, "quality": summarize_quality(rows, 6),
                    "profiles": coverage, "usable_target_coverage_count": sum(r["usable_exact_target"] for r in coverage.values()),
                    "labels": {source: {feature: label_metric(rows, source, feature, 6) for feature in FEATURES} for source in SOURCES}}
                selections = {(r["label_source"], r["profile"]): r for r in pool["selections"]}
                for source in SOURCES:
                    for profile_id, profile in profiles.items():
                        selected = selections.get((source, profile_id), {"status": "pool_unavailable", "selected_id": None, "say": None})
                        rating = ratings[coder].get(references.get((pool["id"], selected["selected_id"]))) if selected["status"] == "ok" else None
                        d = distance(rating, profile) if rating else None
                        oracle = coverage[profile_id]["posthoc_min_distance_all_codable"]
                        choice_rows[coder].append({"pool_id": pool["id"], "case": cid, "generation": generation,
                            "label_source": source, "profile": profile_id, **selected, "rating": rating, "distance": d,
                            "posthoc_min_distance": oracle, "regret": d - oracle if d is not None and oracle is not None else None})
    quality, coverage, label_results, selections, bootstrap = {}, {}, {}, {}, {}
    for subset, case_ids in subsets.items():
        quality[subset], coverage[subset], selections[subset], label_results[subset], bootstrap[subset] = {}, {}, {}, {}, {}
        for generation in GENERATIONS:
            quality[subset][generation], coverage[subset][generation], selections[subset][generation], label_results[subset][generation] = {}, {}, {}, {}
            for coder in CODERS:
                rows = [r for r in candidate_rows[coder] if r["case"] in case_ids and r["generation"] == generation]
                quality[subset][generation][coder] = summarize_quality(rows, len(case_ids) * 6)
                scene_coverage = {cid: pool_stats[coder][cid + "-" + generation]["usable_target_coverage_count"] for cid in case_ids}
                coverage[subset][generation][coder] = {"planned_target_cells": len(case_ids) * len(profiles),
                    "covered_target_cells": sum(scene_coverage.values()), "coverage_rate": rate(sum(scene_coverage.values()), len(case_ids) * len(profiles)),
                    "by_scene": scene_coverage,
                    "by_profile": {pid: {"covered_scenes": sum(pool_stats[coder][cid + "-" + generation]["profiles"][pid]["usable_exact_target"] for cid in case_ids),
                        "mean_posthoc_min_distance_usable": mean(pool_stats[coder][cid + "-" + generation]["profiles"][pid]["posthoc_min_distance_usable"] for cid in case_ids)} for pid in profiles}}
                label_results[subset][generation][coder] = {source: {feature: label_metric(rows, source, feature, len(case_ids) * 6)
                                                                   for feature in FEATURES} for source in SOURCES}
                selections[subset][generation][coder] = {source: summarize_choices([r for r in choice_rows[coder] if r["case"] in case_ids and r["generation"] == generation and r["label_source"] == source]) for source in SOURCES}
        for coder in CODERS:
            bs = {"generation_differences": {}, "label_differences": {}, "selection_differences": {}}
            for metric in ("usable_target_coverage_count", "usable_rate_planned", "mean_naturalness", "mean_appropriateness"):
                deltas = {}
                for cid in case_ids:
                    a, z = (pool_stats[coder][cid + "-" + g] for g in GENERATIONS)
                    av = a[metric] if metric == "usable_target_coverage_count" else a["quality"][metric]
                    zv = z[metric] if metric == "usable_target_coverage_count" else z["quality"][metric]
                    deltas[cid] = zv - av if av is not None and zv is not None else None
                bs["generation_differences"][metric] = ci(deltas, f"{subset}/{coder}/generation/{metric}")
            for generation in GENERATIONS:
                bs["label_differences"][generation] = {}
                for feature in FEATURES:
                    deltas = {cid: pool_stats[coder][cid + "-" + generation]["labels"]["evidence"][feature]["exact_rate_planned"]
                                   - pool_stats[coder][cid + "-" + generation]["labels"]["self"][feature]["exact_rate_planned"] for cid in case_ids}
                    bs["label_differences"][generation][feature] = ci(deltas, f"{subset}/{coder}/{generation}/labels/{feature}")
                bs["selection_differences"][generation] = {}
                for metric in ("distance", "regret", "naturalness", "appropriateness"):
                    deltas = {}
                    for cid in case_ids:
                        diffs = []
                        for pid in profiles:
                            pair = [next(r for r in choice_rows[coder] if r["case"] == cid and r["generation"] == generation and r["profile"] == pid and r["label_source"] == source) for source in SOURCES]
                            vals = [r[metric] if metric in ("distance", "regret") else r["rating"][metric] if r["rating"] else None for r in pair]
                            if all(v is not None for v in vals):
                                diffs.append(vals[1] - vals[0])
                        deltas[cid] = mean(diffs)
                    bs["selection_differences"][generation][metric] = ci(deltas, f"{subset}/{coder}/{generation}/choices/{metric}")
            bootstrap[subset][coder] = bs
    return {"candidate_quality": quality, "usable_target_coverage": coverage, "label_reliability": label_results,
            "offline_selection": selections, "paired_scene_bootstrap": bootstrap, "pool_details": pool_stats}


def hypotheses(metrics):
    h1, h2 = {}, {}
    for coder in CODERS:
        open_cov = metrics["usable_target_coverage"]["open"]
        old, new = open_cov["baseline"][coder], open_cov["coverage"][coder]
        improvements = sum(new["by_scene"][c] > old["by_scene"][c] for c in old["by_scene"])
        delta = new["covered_target_cells"] - old["covered_target_cells"]
        oldq, newq = (metrics["candidate_quality"]["all"][g][coder] for g in GENERATIONS)
        boundary = metrics["candidate_quality"]["boundary"]["coverage"][coder]
        criteria = [{"name": "open_scenes_improved", "observed": improvements, "threshold": 4, "pass": improvements >= 4},
                    {"name": "open_target_cells_gained", "observed": delta, "threshold": 4, "pass": delta >= 4},
                    {"name": "coverage_usable_rate_all_planned", "observed": newq["usable_rate_planned"], "threshold": .85,
                     "pass": newq["usable_rate_planned"] >= .85},
                    {"name": "coverage_boundary_violations", "observed": boundary["boundary_violation_count"], "threshold": 0,
                     "pass": boundary["rated_candidates"] == 12 and boundary["boundary_violation_count"] == 0}]
        for key in ("mean_naturalness", "mean_appropriateness"):
            change = newq[key] - oldq[key] if newq[key] is not None and oldq[key] is not None else None
            criteria.append({"name": "coverage_minus_baseline_" + key, "observed": change, "threshold": -.25,
                             "pass": change >= -.25 if change is not None else False})
        h1[coder] = {"pass": all(r["pass"] for r in criteria), "criteria": criteria}
        label = metrics["label_reliability"]["all"]["coverage"][coder]
        attention_delta = label["evidence"]["attention"]["exact_rate_planned"] - label["self"]["attention"]["exact_rate_planned"]
        initiative_delta = label["evidence"]["initiative"]["exact_rate_planned"] - label["self"]["initiative"]["exact_rate_planned"]
        criteria = [{"name": "attention_exact_rate_gain_all_48", "observed": attention_delta, "threshold": .1, "pass": attention_delta >= .1},
                    {"name": "attention_exact_rate_evidence_all_48", "observed": label["evidence"]["attention"]["exact_rate_planned"], "threshold": .7,
                     "pass": label["evidence"]["attention"]["exact_rate_planned"] >= .7},
                    {"name": "initiative_exact_rate_change_all_48", "observed": initiative_delta, "threshold": -.05, "pass": initiative_delta >= -.05}]
        for feature in FEATURES:
            value = label["evidence"][feature]["nonunknown_coverage_planned"]
            criteria.append({"name": feature + "_validated_nonunknown_coverage_all_48", "observed": value, "threshold": .9, "pass": value >= .9})
        h2[coder] = {"pass": all(r["pass"] for r in criteria), "criteria": criteria,
                     "selection_effect_open": metrics["paired_scene_bootstrap"]["open"][coder]["selection_differences"]["coverage"]}
    return {"H1_generation": {"supports_holdout_expansion": all(v["pass"] for v in h1.values()), "coders": h1},
            "H2_labels": {"passes_label_screen": all(v["pass"] for v in h2.values()), "coders": h2},
            "interpretation": "Engineering screens, not significance or noninferiority tests. Label-screen success alone does not justify deployment: selection, quality and full cost must also be considered."}


def savefig(fig, out, name):
    fig.savefig(out / "figures" / (name + ".png"), dpi=180, bbox_inches="tight", metadata={"Software": "candidate revision offline analysis"})
    fig.savefig(out / "figures" / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def bars(ax, labels, values, ylabel, ylim, points=False):
    x = np.arange(len(labels))
    for j, coder in enumerate(CODERS):
        ys = [values(i, coder) for i in range(len(labels))]
        ys = [np.nan if v is None else v for v in ys]
        positions = x + (j - .5) * .30
        if points:
            ax.scatter(positions, ys, color=COLORS[j], label="Coder " + coder, s=50)
        else:
            ax.bar(positions, ys, width=.28, color=COLORS[j], label="Coder " + coder)
        for pos, y in zip(positions, ys):
            if not np.isnan(y):
                ax.annotate(f"{y:.2f}", (pos, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)


def make_figures(summary, out):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "candidate-revision-20261107"})
    q, cov, lab, sel = (summary[k] for k in ("candidate_quality", "usable_target_coverage", "label_reliability", "offline_selection"))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), layout="constrained")
    names = ["Free six candidates", "Coverage briefs"]
    bars(axes[0, 0], names, lambda i, c: cov["open"][GENERATIONS[i]][c]["covered_target_cells"], "Usable exact-target cells / 18", (0, 20))
    axes[0, 0].set_title("Six open scenes x three preference profiles")
    bars(axes[0, 1], names, lambda i, c: q["all"][GENERATIONS[i]][c]["usable_rate_planned"], "Usable candidates / 48 planned", (0, 1.12))
    axes[0, 1].axhline(.85, linestyle="--", color="#8b3c35", label="Usability gate: 0.85")
    axes[0, 1].set_title("Candidate usability includes facts and boundaries")
    bars(axes[1, 0], names, lambda i, c: q["all"][GENERATIONS[i]][c]["mean_naturalness"], "Naturalness (1-5)", (1, 5.3), points=True)
    bars(axes[1, 1], names, lambda i, c: q["all"][GENERATIONS[i]][c]["mean_appropriateness"], "Appropriateness (1-5)", (1, 5.3), points=True)
    axes[0, 0].legend(); axes[0, 1].legend(fontsize=8)
    fig.suptitle("Generation intervention: actual text coverage and quality\nLabel sources share the same texts; quality is not counted twice")
    savefig(fig, out, "fig1_coverage_quality")

    conditions = [(g, s) for g in GENERATIONS for s in SOURCES]
    names = ["Baseline\nself", "Baseline\nevidence", "Coverage\nself", "Coverage\nevidence"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, feature in zip(axes, FEATURES):
        bars(ax, names, lambda i, c: lab["all"][conditions[i][0]][c][conditions[i][1]][feature]["exact_rate_planned"], "Exact matches / all 48 planned candidates", (0, 1.18))
        xs = np.arange(4)
        vals = [lab["all"][g]["1"][s][feature]["nonunknown_coverage_planned"] for g, s in conditions]
        ax.scatter(xs, vals, facecolors="none", edgecolors="#333333", marker="D", s=55, label="Validated nonunknown coverage")
        ax.set_title(feature.title())
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Label acquisition and agreement — unknown and missing labels are not successes")
    savefig(fig, out, "fig2_labels")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), layout="constrained")
    for column, generation in enumerate(GENERATIONS):
        for row, metric, title in ((0, "mean_distance", "Actual target distance"), (1, "mean_regret", "Regret vs. same-pool coded minimum")):
            ax = axes[row, column]
            bars(ax, ["Self labels", "Evidence labels"], lambda i, c, g=generation, k=metric: sel["open"][g][c][SOURCES[i]][k], title + " (0-4)", (0, 4.4))
            ax.set_title(generation.title() + ": 18 planned offline choices per source")
    axes[0, 0].legend()
    fig.suptitle("Same-pool selection diagnostics on six open scenes\nLower is closer; the posthoc minimum is not a deployed oracle")
    savefig(fig, out, "fig3_offline_selection")

    cost = summary["costs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    standalone = cost["standalone_reconstructed_cost"]
    xs = np.arange(4)
    pool_tokens = [standalone[g][s]["mean_pool_tokens"] for g, s in conditions]
    label_tokens = [standalone[g][s]["mean_label_tokens"] for g, s in conditions]
    pool_tokens = [np.nan if v is None else v for v in pool_tokens]
    label_tokens = [np.nan if v is None else v for v in label_tokens]
    axes[0].bar(xs, pool_tokens, label="Whole pool generation", color="#2673b8")
    axes[0].bar(xs, label_tokens, bottom=pool_tokens, label="Whole evidence-label call", color="#e08136")
    axes[0].set_xticks(xs, names); axes[0].set_ylabel("Reconstructed total tokens per one reply")
    axes[0].set_title("No division by candidates or profiles"); axes[0].legend(fontsize=8)
    actual = cost["actual_research"]
    stages = actual["by_stage"]
    vals = [stages[s]["total_tokens"] for s in ("pool", "labels")]
    axes[1].bar(["Pool generation", "Evidence labeling"], vals, color=("#2673b8", "#e08136"), width=.55)
    for i, s in enumerate(("pool", "labels")):
        axes[1].annotate(f"{stages[s]['calls']} calls\n{vals[i]:,} tokens", (i, vals[i]), xytext=(0, 5), textcoords="offset points", ha="center")
    axes[1].set_ylim(0, max(vals + [1]) * 1.22); axes[1].set_ylabel("Observed study total tokens")
    axes[1].set_title(f"Actual requests: {actual['calls']}; offline choices are not API calls")
    fig.suptitle("Full costs from recorded requests — Codex coding labour is unpriced")
    savefig(fig, out, "fig4_cost")


def analyze(out):
    for coder in CODERS:
        if not (HERE / "reviews" / f"ratings_{coder}.json").exists():
            raise SystemExit("Both independent coding files are required; no provisional results will be published.")
    audit = Audit()
    protocol = b.load(HERE / "protocol.json")
    cases = {c["id"]: c for c in b.load(HERE / "cases.json")}
    profiles = {p["id"]: p for p in b.load(HERE / "profiles.json")}
    data, calls = b.load(HERE / "run/results.json"), read_calls()
    key_rows = b.load(HERE / "review_key.json")
    keys = {r["id"]: r for r in key_rows}
    audit.check(len(keys) == len(key_rows), "unique review identifiers")
    ratings = {}
    for coder in CODERS:
        rows = b.load(HERE / "reviews" / f"ratings_{coder}.json")
        ratings[coder] = {r["id"]: r for r in rows}
        audit.check(len(rows) == len(ratings[coder]), "unique coder IDs " + coder)
    references = audit_sources(data, calls, cases, profiles, keys, ratings, audit)
    if audit.failures:
        out.mkdir(parents=True, exist_ok=True); b.save(out / "audit_report.json", audit.result())
        raise ValueError("Input audit failed: " + "; ".join(audit.failures[:8]))
    metrics = build_metrics(data["records"], cases, profiles, references, ratings)
    summary = {"study_id": protocol["study_id"], "protocol_sha256": b.sha(HERE / "protocol.json"),
        "counts": {"scenes": len(cases), "open_scenes": sum(not c["boundaries"] for c in cases.values()),
                   "boundary_scenes": sum(bool(c["boundaries"]) for c in cases.values()), "planned_pools": 16,
                   "successful_pools": sum(p["status"] == "ok" for p in data["records"]),
                   "planned_candidates": 96, "generated_candidate_records": sum(len(p["candidates"]) for p in data["records"]),
                   "unique_scene_texts": len(keys), "actual_api_calls": len(calls),
                   "planned_offline_choices": 96, "recorded_offline_choices": sum(len(p["selections"]) for p in data["records"]),
                   "delivered_offline_choices": sum(s["status"] == "ok" for p in data["records"] for s in p["selections"])},
        **metrics, "coder_agreement": coder_agreement(keys, ratings), "hypotheses": hypotheses(metrics),
        "validation_errors": {g: dict(Counter(e for p in data["records"] if p["generation"] == g for r in p["labels"] for e in r["validation_errors"])) for g in GENERATIONS},
        "costs": actual_costs(calls, data["records"]),
        "notes": ["Two component factors share candidate text; self/evidence are not independent generation or quality samples.",
                  "Label exact-rate denominators are planned candidates; missing/unknown is never exact-match success.",
                  "Selection means exclude unscored/abstained text and always include coverage counts.",
                  "The regret minimum uses all same-pool codable candidates, not just usable ones, and is posthoc diagnostic only.",
                  "Bootstrap resamples scenes, not candidates or profile copies. Intervals are descriptive.",
                  "New scenes were authored after the previous study; they were not a previously sealed holdout.",
                  "This study does not compare strong-prompt selection or direct single replies and cannot establish superiority over traditional prompting."]}
    out.mkdir(parents=True, exist_ok=True)
    b.save(out / "summary.json", summary); b.save(out / "audit_report.json", audit.result())
    b.save(out / "analysis_environment.json", {"python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "font": "DejaVu Sans", "offline": True,
        "bootstrap_resamples": BOOTSTRAPS, "seed": SEED, "analysis_sha256": b.sha(Path(__file__))})
    make_figures(summary, out)
    return summary, audit.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", type=Path, default=HERE); args = parser.parse_args()
    summary, audit = analyze(args.out)
    import json
    print(json.dumps({"counts": summary["counts"], "audit": audit,
        "H1_expand": summary["hypotheses"]["H1_generation"]["supports_holdout_expansion"],
        "H2_label_screen": summary["hypotheses"]["H2_labels"]["passes_label_screen"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
