"""Offline persona-conditioned evaluation; dimensions never form a total score."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import benchmark as b
from mechanism import choose, validate_annotations, unavailable_annotations
from prepare_reviews import prepare
from selection import parse_pool, random_select

HERE = Path(__file__).resolve().parent
CODERS = ("1", "2")
PROFILES = ("S", "L")
METHODS = ("direct", "coverage")
FEATURES = ("attention", "initiative")
STYLE = {"formal", "observational", "restrained", "lively", "colloquial", "metaphorical", "mechanical"}
COLORS = ("#2673b8", "#e08136")


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def rate(n, d):
    return n / d if d else None


def known(v):
    return type(v) is int and v in (0, 1, 2)


def distance(rating, profile):
    return sum(abs(rating[k] - profile["target"][k]) for k in FEATURES) if rating and all(known(rating.get(k)) for k in FEATURES) else None


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from strings(value)


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)

    def result(self):
        return {"checks": self.checks, "status": "passed" if not self.failures else "failed", "failures": self.failures,
                "interpretation": "Material and computation checks; not an evaluation pass rate."}


def rating_schema(row):
    if set(row) != {"id", "persona_fit", "context_fit", "grounding", "boundary", "style", "attention", "initiative", "evidence"}:
        return False
    return (all(row[k] is None or type(row[k]) is int and 1 <= row[k] <= 5 for k in ("persona_fit", "context_fit"))
            and row["grounding"] in ("supported", "violation", "uncertain")
            and row["boundary"] in ("respected", "violation", "uncertain")
            and all(row[k] is None or known(row[k]) for k in FEATURES)
            and isinstance(row["style"], list) and len(row["style"]) == len(set(row["style"])) and set(row["style"]) <= STYLE
            and isinstance(row["evidence"], dict) and set(row["evidence"]) == {"say_quote", "context_quote", "persona_rule", "reason"}
            and all(isinstance(v, str) for v in row["evidence"].values()))


def validate(data, calls, cases, profiles, key_rows, ratings, audit):
    protocol = b.load(HERE / "protocol.json")
    for name, expected in protocol["hashes"].items():
        audit.check((HERE / name).exists() and b.sha(HERE / name) == expected, "frozen hash " + name)
    audit.check(data["protocol_sha256"] == b.sha(HERE / "protocol.json"), "protocol reference")
    audit.check(len(calls) == data["actual_calls"] <= 36, "actual call budget")
    by_call = {r["id"]: r for r in calls}
    audit.check(len(by_call) == len(calls), "unique call IDs")
    records = {r["id"]: r for r in data["records"]}
    expected = {f"{c}-{p}-{m}" for c in cases for p in profiles for m in METHODS}
    audit.check(len(data["records"]) == 24 and set(records) == expected, "all 24 planned final records")
    for call in calls:
        cid = call["id"]
        audit.check(b.load(HERE / "run/calls" / (cid + ".json")) == call, "raw archive equality " + cid)
        if call["status"] == "ok":
            choice = call["response"]["choices"][0]
            audit.check(choice["finish_reason"] == "stop" and choice["message"]["content"] == call["output"], "complete preserved output " + cid)
        usage = call.get("response", {}).get("usage", {})
        if all(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            audit.check(usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"], "token arithmetic " + cid)
    for rid, row in records.items():
        call = by_call.get(row["generation_call"])
        if call:
            audit.check(call["request"] == b.body_for(call["stage"], b.generation_messages(cases[row["case"]], profiles[row["profile"]], row["method"])), "whole generation request " + rid)
        if row["method"] == "direct":
            if row["say"] is not None:
                audit.check(call is not None and call["status"] == "ok" and row["say"] == call["output"].strip(), "direct text preserved " + rid)
            continue
        if not row["candidates"]:
            audit.check(row["say"] is None, "no fabricated text for missing pool " + rid)
            continue
        audit.check(call is not None and parse_pool(call["output"]) == row["candidates"], "original pool parse " + rid)
        messages, hidden, mapping = b.label_messages(cases[row["case"]], rid, row["candidates"])
        payload = json.loads(messages[1]["content"])
        audit.check(set(payload) == {"context", "candidates"} and all(set(c) == {"id", "text"} for c in payload["candidates"]), "label input hides persona, plan and self tags " + rid)
        audit.check(mapping == row["label_id_mapping"], "blind mapping preserved " + rid)
        label_call = by_call.get(row["label_call"])
        if label_call:
            audit.check(label_call["request"] == b.body_for("labels", messages), "whole label request " + rid)
        if label_call and label_call["status"] == "ok":
            try:
                expected_labels = validate_annotations(label_call["output"], hidden, mapping)
            except (ValueError, TypeError, KeyError):
                expected_labels = unavailable_annotations(row["candidates"], "invalid_labels")
        else:
            expected_labels = unavailable_annotations(row["candidates"], row["label_status"])
        audit.check(expected_labels == row["labels"], "validated evidence labels " + rid)
        choice = choose(row["candidates"], row["labels"], profiles[row["profile"]])
        audit.check(row["say"] == (choice["text"] if choice else None) and row["selected_id"] == (choice["candidate_id"] if choice else None), "verbatim program selection " + rid)
        audit.check(row["random_selected_id"] == random_select(row["candidates"], row["case"], row["profile"]), "frozen random diagnostic " + rid)
    inputs, expected_key = prepare(data)
    audit.check(inputs == b.load(HERE / "reviews/inputs.json"), "anonymous persona-visible input reproduction")
    audit.check(key_rows == expected_key, "review mapping reproduction")
    by_input = {r["id"]: r for r in inputs}
    for coder in CODERS:
        audit.check(set(ratings[coder]) == set(by_input), "all unique texts coded " + coder)
        for key, rating in ratings[coder].items():
            audit.check(rating_schema(rating), "rating schema " + coder + "/" + key)
            if not rating_schema(rating) or key not in by_input:
                continue
            item, ev = by_input[key], rating["evidence"]
            audit.check(bool(ev["say_quote"]) and ev["say_quote"] in item["say"], "reply evidence " + coder + "/" + key)
            texts = list(strings({k: item[k] for k in ("history", "user", "facts", "boundaries")}))
            audit.check(ev["context_quote"] == "" or any(ev["context_quote"] in s for s in texts), "context evidence " + coder + "/" + key)
            audit.check(bool(ev["persona_rule"]) and any(ev["persona_rule"] in s for s in strings(item["persona"]["spec"])), "persona rule evidence " + coder + "/" + key)
            audit.check(bool(ev["reason"].strip()), "nonempty rating reason " + coder + "/" + key)
    refs, fixture_refs = {}, {}
    for row in key_rows:
        if row["kind"] == "fixture":
            audit.check(row["fixture_id"] not in fixture_refs, "unique fixture mapping " + row["fixture_id"])
            fixture_refs[row["fixture_id"]] = row["id"]
        else:
            for ref in row["references"]:
                pair = ref["record_id"], ref["candidate_id"]
                audit.check(pair not in refs, "unique online reference " + "/".join(pair))
                refs[pair] = row["id"]
    return refs, fixture_refs


def score_summary(rows, field):
    rs = [r["rating"] for r in rows if r["rating"] is not None]
    values = [r[field] for r in rs]
    known_values = [v for v in values if v is not None]
    return {"known": len(known_values), "unknown": sum(v is None for v in values), "missing_rating": len(rows) - len(rs),
            "mean": mean(known_values), "distribution": {str(i): values.count(i) for i in range(1, 6)},
            "minimum": min(known_values) if known_values else None, "maximum": max(known_values) if known_values else None}


def categorical(rows, field, categories):
    rs = [r["rating"] for r in rows if r["rating"] is not None]
    return {"planned": len(rows), "rated": len(rs), "missing_rating": len(rows) - len(rs),
            "counts": {k: sum(r[field] == k for r in rs) for k in categories}}


def summarize(rows, cases):
    boundary_rows = [r for r in rows if cases[r["case"]]["boundaries"]]
    return {"planned": len(rows), "delivered": sum(r["status"] == "ok" for r in rows),
            "persona_fit": score_summary(rows, "persona_fit"), "context_fit": score_summary(rows, "context_fit"),
            "grounding": categorical(rows, "grounding", ("supported", "violation", "uncertain")),
            "boundary_applicable": categorical(boundary_rows, "boundary", ("respected", "violation", "uncertain")),
            "boundary_not_applicable_count": len(rows) - len(boundary_rows),
            "style_descriptors": {s: sum(r["rating"] is not None and s in r["rating"]["style"] for r in rows) for s in sorted(STYLE)}}


def differences(left, right, field):
    index = {r["case"]: r for r in left}
    values = {}
    for row in right:
        a = index.get(row["case"], {}).get("rating")
        z = row["rating"]
        values[row["case"]] = z[field] - a[field] if a and z and a[field] is not None and z[field] is not None else None
    return {"case_differences": values, "known_pairs": sum(v is not None for v in values.values()),
            "unknown_or_missing_pairs": sum(v is None for v in values.values()), "mean_difference": mean(values.values())}


def label_stats(rows, source, feature):
    both = [r for r in rows if known(r[source].get(feature)) and r["rating"] and known(r["rating"].get(feature))]
    model_known = sum(known(r[source].get(feature)) for r in rows)
    exact = sum(r[source][feature] == r["rating"][feature] for r in both)
    matrix = [[0] * 4 for _ in range(4)]
    for row in rows:
        predicted = row[source].get(feature)
        actual = row["rating"].get(feature) if row["rating"] else None
        matrix[predicted if known(predicted) else 3][actual if known(actual) else 3] += 1
    return {"planned_candidates": len(rows), "model_nonunknown": model_known, "model_unknown_or_missing": len(rows) - model_known,
            "model_coverage": rate(model_known, len(rows)), "joint_nonunknown": len(both), "exact_matches": exact,
            "exact_rate_all_planned": rate(exact, len(rows)), "exact_rate_joint_nonunknown": rate(exact, len(both)),
            "mean_absolute_error_joint_nonunknown": mean(abs(r[source][feature] - r["rating"][feature]) for r in both),
            "confusion": {"rows": "model labels", "columns": "independent reviewer coding", "labels": [0, 1, 2, "unknown_or_missing"], "counts": matrix}}


def online_metrics(records, cases, profiles, references, ratings):
    main_rows, random_rows, candidate_rows, pools = {}, {}, {}, {}
    for coder in CODERS:
        main_rows[coder], random_rows[coder], candidate_rows[coder], pools[coder] = [], [], [], []
        for row in records:
            candidate_id = "direct" if row["method"] == "direct" else row["selected_id"]
            rid = references.get((row["id"], candidate_id))
            main_rows[coder].append({"case": row["case"], "profile": row["profile"], "method": row["method"],
                "record_id": row["id"], "status": "ok" if row["say"] is not None else row["status"], "say": row["say"],
                "review_id": rid, "rating": ratings[coder].get(rid)})
            if row["method"] != "coverage":
                continue
            cs = {c["id"]: c for c in row["candidates"]}
            ls = {r["id"]: r for r in row["labels"]}
            current = []
            for i in range(1, 7):
                cid = f"C{i:02}"; ref = references.get((row["id"], cid))
                item = {"case": row["case"], "profile": row["profile"], "record_id": row["id"], "candidate_id": cid,
                        "status": "ok" if cid in cs else "unavailable", "rating": ratings[coder].get(ref),
                        "self": cs.get(cid, {}), "evidence": ls.get(cid, {}), "review_id": ref}
                current.append(item); candidate_rows[coder].append(item)
            random_id = row["random_selected_id"]
            random_ref = references.get((row["id"], random_id))
            random_rows[coder].append({"case": row["case"], "profile": row["profile"], "record_id": row["id"],
                "status": "ok" if random_id in cs else "unavailable", "selected_id": random_id,
                "say": cs[random_id]["text"] if random_id in cs else None, "review_id": random_ref, "rating": ratings[coder].get(random_ref)})
            ds = [distance(item["rating"], profiles[row["profile"]]) for item in current]
            ds = [d for d in ds if d is not None]
            chosen_rating = ratings[coder].get(rid)
            random_rating = ratings[coder].get(random_ref)
            pools[coder].append({"record_id": row["id"], "case": row["case"], "profile": row["profile"],
                "codable_candidates": len(ds), "posthoc_min_distance": min(ds) if ds else None,
                "exact_target_candidate_count": sum(d == 0 for d in ds),
                "selected_distance": distance(chosen_rating, profiles[row["profile"]]),
                "random_distance": distance(random_rating, profiles[row["profile"]])})
    main, paired, random_summary, diagnostic, pool_summary = {}, {}, {}, {}, {}
    for pid in PROFILES:
        main[pid], paired[pid], random_summary[pid], diagnostic[pid], pool_summary[pid] = {}, {}, {}, {}, {}
        for method in METHODS:
            main[pid][method] = {c: summarize([r for r in main_rows[c] if r["profile"] == pid and r["method"] == method], cases) for c in CODERS}
        for coder in CODERS:
            direct = [r for r in main_rows[coder] if r["profile"] == pid and r["method"] == "direct"]
            selected = [r for r in main_rows[coder] if r["profile"] == pid and r["method"] == "coverage"]
            random_selected = [r for r in random_rows[coder] if r["profile"] == pid]
            paired[pid][coder] = {field: differences(direct, selected, field) for field in ("persona_fit", "context_fit")}
            random_summary[pid][coder] = {"scores": summarize(random_selected, cases),
                "program_minus_random": {field: differences(random_selected, selected, field) for field in ("persona_fit", "context_fit")}}
            cand = [r for r in candidate_rows[coder] if r["profile"] == pid]
            diagnostic[pid][coder] = {"quality_dimensions": summarize(cand, cases),
                "labels": {source: {feature: label_stats(cand, source, feature) for feature in FEATURES} for source in ("self", "evidence")}}
            ps = [r for r in pools[coder] if r["profile"] == pid]
            pool_summary[pid][coder] = {"pools": ps, "known_minima": sum(r["posthoc_min_distance"] is not None for r in ps),
                "mean_posthoc_min_distance": mean(r["posthoc_min_distance"] for r in ps),
                "known_selected_distances": sum(r["selected_distance"] is not None for r in ps),
                "mean_selected_distance": mean(r["selected_distance"] for r in ps),
                "known_random_distances": sum(r["random_distance"] is not None for r in ps),
                "mean_random_distance": mean(r["random_distance"] for r in ps)}
    diagnostic["all_profiles"] = {coder: {"labels": {source: {feature: label_stats(candidate_rows[coder], source, feature) for feature in FEATURES}
                                                                           for source in ("self", "evidence")}} for coder in CODERS}
    return {"main": main, "paired_coverage_minus_direct": paired,
            "final_outputs": [{"record_id": r["id"], "case": r["case"], "profile": r["profile"], "method": r["method"], "status": r["status"], "say": r["say"]} for r in records],
            "random_diagnostic": {"planned_offline_choices": 12, "by_profile": random_summary,
                                  "outputs": [{k: v for k, v in r.items() if k != "rating"} for r in random_rows["1"]],
                                  "notes": "One frozen draw per pool; not a random-policy expectation, new generation or main output."},
            "candidates": {"planned": 72, "by_profile": diagnostic, "distance_diagnostics": pool_summary,
                           "notes": "Candidate counts are diagnostic records, not independent scenarios or additional main outputs. Target distance does not define persona fit or text quality."}}


def calibration(fixture_refs, ratings):
    key = b.load(HERE / "fixture_key.json")
    items = {r["id"]: r for r in key["items"]}
    result = {"handwritten_items": len(items), "pairs": [], "defect_diagnostics": {}}
    for pair in key["pairs"]:
        entry = {"pair_id": pair["pair_id"], "fixture_ids": pair["items"], "coders": {}}
        for coder in CODERS:
            rs = {pid: ratings[coder][fixture_refs[fid]] for pid, fid in pair["items"].items()}
            a, z = rs["S"], rs["L"]
            entry["coders"][coder] = {"persona_scores": {p: r["persona_fit"] for p, r in rs.items()},
                "persona_S_minus_L": a["persona_fit"] - z["persona_fit"] if a["persona_fit"] is not None and z["persona_fit"] is not None else None,
                "context_scores": {p: r["context_fit"] for p, r in rs.items()},
                "grounding": {p: r["grounding"] for p, r in rs.items()}, "grounding_stable": a["grounding"] == z["grounding"],
                "both_grounding_supported": all(r["grounding"] == "supported" for r in rs.values()),
                "style_descriptors": {p: r["style"] for p, r in rs.items()}}
        result["pairs"].append(entry)
    for coder in CODERS:
        details = [{"fixture_id": fid, "review_id": fixture_refs[fid], "expected_grounding": item["expected_grounding"],
                    "expected_boundary": item["expected_boundary"], "observed_grounding": ratings[coder][fixture_refs[fid]]["grounding"],
                    "observed_boundary": ratings[coder][fixture_refs[fid]]["boundary"],
                    "persona_fit": ratings[coder][fixture_refs[fid]]["persona_fit"],
                    "context_fit": ratings[coder][fixture_refs[fid]]["context_fit"]} for fid, item in items.items()]
        factual = [r for r in details if r["expected_grounding"] == "violation"]
        boundaries = [r for r in details if r["expected_boundary"] == "violation"]
        result["defect_diagnostics"][coder] = {
            "factual_defects": {"authored": len(factual), "flagged": sum(r["observed_grounding"] == "violation" for r in factual),
                                "uncertain": sum(r["observed_grounding"] == "uncertain" for r in factual), "items": factual},
            "boundary_defects": {"authored": len(boundaries), "flagged": sum(r["observed_boundary"] == "violation" for r in boundaries),
                                 "uncertain": sum(r["observed_boundary"] == "uncertain" for r in boundaries), "items": boundaries},
            "correct_text_items": {"authored": 8, "grounding_supported": sum(r["expected_grounding"] == "supported" and r["expected_boundary"] == "respected" and r["observed_grounding"] == "supported" for r in details)},
            "stable_grounding_pairs": sum(p["coders"][coder]["grounding_stable"] for p in result["pairs"]),
            "all_items": details}
    result["interpretation"] = "Author-designed construct checks, not population accuracy, human truth or a comprehensive pass test. No required persona-score ordering or numeric threshold."
    return result


def agreement(key_rows, ratings, cases):
    result = {}
    fixtures = {r["id"]: r for r in b.load(HERE / "fixtures.json")}
    for kind in ("online", "fixture"):
        ids = [r["id"] for r in key_rows if r["kind"] == kind]
        result[kind] = {"unique_review_units": len(ids)}
        for field in ("persona_fit", "context_fit", "grounding", "boundary", "attention", "initiative"):
            pairs = [i for i in ids if ratings["1"][i][field] is not None and ratings["2"][i][field] is not None]
            exact = sum(ratings["1"][i][field] == ratings["2"][i][field] for i in pairs)
            result[kind][field] = {"joint_known": len(pairs), "unknown_pairs": len(ids) - len(pairs), "exact_count": exact, "exact_rate": rate(exact, len(pairs))}
        applicable = [r["id"] for r in key_rows if r["kind"] == kind
                      and bool(cases[r["case"]]["boundaries"] if kind == "online" else fixtures[r["fixture_id"]]["boundaries"])]
        exact = sum(ratings["1"][i]["boundary"] == ratings["2"][i]["boundary"] for i in applicable)
        result[kind]["boundary"].update(scope="all records, including those without an explicit boundary",
                                         nonapplicable_records=len(ids) - len(applicable))
        result[kind]["boundary_explicit_only"] = {"scope": "only records with an explicit user boundary",
            "applicable_records": len(applicable), "joint_known": len(applicable), "exact_count": exact,
            "exact_rate": rate(exact, len(applicable))}
    return result


def cost_analysis(calls, records):
    usage = lambda c: c.get("response", {}).get("usage", {})
    by_call = {c["id"]: c for c in calls}
    actual = {"calls": len(calls), "usage_available": sum("total_tokens" in usage(c) for c in calls),
              "prompt_tokens": sum(usage(c).get("prompt_tokens", 0) for c in calls),
              "completion_tokens": sum(usage(c).get("completion_tokens", 0) for c in calls),
              "total_tokens": sum(usage(c).get("total_tokens", 0) for c in calls),
              "observed_models": dict(Counter(c.get("response", {}).get("model", "missing") for c in calls)), "by_stage": {}}
    for stage in ("direct", "pool", "labels"):
        rows = [c for c in calls if c["stage"] == stage]
        actual["by_stage"][stage] = {"calls": len(rows), "total_tokens": sum(usage(c).get("total_tokens", 0) for c in rows),
                                    "mean_observed_request_seconds": mean(c["elapsed_s"] for c in rows)}
    final = {}
    for pid in PROFILES:
        final[pid] = {}
        for method in METHODS:
            costs = []
            for row in records:
                if row["profile"] != pid or row["method"] != method:
                    continue
                required = [by_call.get(row["generation_call"])]
                if method == "coverage":
                    required.append(by_call.get(row["label_call"]))
                if all(c is not None and "total_tokens" in usage(c) for c in required):
                    costs.append({"case": row["case"], "tokens": sum(usage(c)["total_tokens"] for c in required),
                                  "generation_tokens": usage(required[0])["total_tokens"],
                                  "label_tokens": usage(required[1])["total_tokens"] if method == "coverage" else 0,
                                  "sum_observed_request_seconds": sum(c["elapsed_s"] for c in required), "calls": len(required)})
            final[pid][method] = {"planned_final_outputs": 6, "complete_cost_records": len(costs),
                "mean_full_tokens": mean(r["tokens"] for r in costs), "mean_generation_tokens": mean(r["generation_tokens"] for r in costs),
                "mean_label_tokens": mean(r["label_tokens"] for r in costs), "mean_calls": mean(r["calls"] for r in costs),
                "mean_reconstructed_seconds": mean(r["sum_observed_request_seconds"] for r in costs), "by_scene": costs}
    return {"actual_research": actual, "per_final_full_route": final,
            "notes": "Coverage bears one entire six-candidate generation plus one entire label call, without division by candidates. Summed request times are reconstructed route times, not observed client end-to-end latency. Codex coding labour is unpriced, not zero-cost."}


def savefig(fig, out, name):
    fig.savefig(out / "figures" / (name + ".png"), dpi=180, bbox_inches="tight", metadata={"Software": "persona-conditioned offline evaluation"})
    fig.savefig(out / "figures" / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def make_figures(summary, out):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "persona-evaluation-20261107"})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), layout="constrained")
    for column, pid in enumerate(PROFILES):
        for row, field in enumerate(("persona_fit", "context_fit")):
            ax = axes[row, column]
            for j, coder in enumerate(CODERS):
                xs = np.arange(2) + (j - .5) * .22
                stats = [summary["main"][pid][m][coder][field] for m in METHODS]
                vals = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
                ax.scatter(xs, vals, s=55, color=COLORS[j], label="Coder " + coder)
                for x, y, st in zip(xs, vals, stats):
                    if not np.isnan(y):
                        ax.annotate(f"{y:.2f}\nn={st['known']}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
            ax.set_xticks([0, 1], ["Direct", "Candidate selection"]); ax.set_xlim(-.4, 1.4); ax.set_ylim(1, 5.6)
            ax.set_ylabel(("Persona fit" if field == "persona_fit" else "Context fit") + " (1-5)")
            ax.set_title("Shorekeeper (S)" if pid == "S" else "Lively companion (L)"); ax.grid(axis="y", alpha=.18)
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Twenty-four final replies — persona and context scored separately\nStyle descriptors are not quality thresholds; unknown scores are excluded and counted")
    savefig(fig, out, "fig1_persona_context")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    conditions = [(p, m) for p in PROFILES for m in METHODS]
    names = ["S / direct", "S / selection", "L / direct", "L / selection"]
    cat_colors = ("#6f9f84", "#bc5750", "#d3ac61")
    for i, coder in enumerate(CODERS):
        for j, field, categories, maximum in ((0, "grounding", ("supported", "violation", "uncertain"), 6),
                                               (1, "boundary_applicable", ("respected", "violation", "uncertain"), 2)):
            ax = axes[i, j]; bottoms = np.zeros(4)
            for category, color in zip(categories, cat_colors):
                vals = [summary["main"][p][m][coder][field]["counts"][category] for p, m in conditions]
                ax.bar(np.arange(4), vals, bottom=bottoms, color=color, label=category.title())
                for x, val, base in zip(range(4), vals, bottoms):
                    if val:
                        ax.text(x, base + val / 2, str(val), ha="center", va="center", color="white", fontweight="bold")
                bottoms += np.array(vals)
            ax.set_ylim(0, maximum + .7); ax.set_yticks(range(maximum + 1)); ax.set_xticks(range(4), names, rotation=15)
            ax.set_ylabel(f"Count / {maximum} planned")
            ax.set_title(f"Coder {coder}: " + ("grounding, six scenes" if field == "grounding" else "explicit boundaries, E01 and E02 only"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Reviewer grounding and boundary judgments — independent dimensions\nBoundary 'respected' uses green; no overall pass score")
    savefig(fig, out, "fig2_grounding_boundaries")

    cal = summary["calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    names = ["Formal\ncalculation", "Formal\ncompanionship", "Lively\ncalculation", "Lively\nobservation"]
    for j, coder in enumerate(CODERS):
        xs = np.arange(4) + (j - .5) * .22
        vals = [p["coders"][coder]["persona_S_minus_L"] for p in cal["pairs"]]
        axes[0].scatter(xs, [np.nan if v is None else v for v in vals], s=55, color=COLORS[j], label="Coder " + coder)
        for x, value in zip(xs, vals):
            if value is not None:
                axes[0].annotate(f"{value:+.0f}", (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
        stats = cal["defect_diagnostics"][coder]
        detected = [stats[k]["flagged"] for k in ("factual_defects", "boundary_defects")]
        axes[1].bar(np.arange(2) + (j - .5) * .32, detected, .3, color=COLORS[j], label="Coder " + coder)
        for x, val in zip(np.arange(2) + (j - .5) * .32, detected):
            axes[1].annotate(f"{val}/2", (x, val), xytext=(0, 5), textcoords="offset points", ha="center")
    axes[0].axhline(0, color="#777777", linewidth=.8); axes[0].set_xticks(range(4), names)
    axes[0].set_ylim(-4.6, 4.6); axes[0].set_ylabel("Same-text persona-fit difference: S minus L")
    stable = "; ".join(f"coder {c}: {cal['defect_diagnostics'][c]['stable_grounding_pairs']}/4" for c in CODERS)
    axes[0].set_title("Stable grounding pairs — " + stable)
    axes[1].set_xticks([0, 1], ["Authored factual defects", "Authored boundary defects"])
    axes[1].set_ylim(0, 2.5); axes[1].set_yticks([0, 1, 2]); axes[1].set_ylabel("Flagged as violation / two authored items")
    axes[1].set_title("Construct checks, not population accuracy"); axes[0].legend(loc="upper right")
    fig.suptitle("Twelve handwritten calibration items, reported separately from online results")
    savefig(fig, out, "fig3_calibration")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    pairs = [(f, s) for f in FEATURES for s in ("self", "evidence")]
    for j, coder in enumerate(CODERS):
        vals = [summary["candidates"]["by_profile"]["all_profiles"][coder]["labels"][s][f]["exact_rate_all_planned"] for f, s in pairs]
        axes[0].bar(np.arange(4) + (j - .5) * .32, vals, .3, color=COLORS[j], label="Coder " + coder)
        for x, val in zip(np.arange(4) + (j - .5) * .32, vals):
            axes[0].annotate(f"{val:.2f}", (x, val), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    coverage = [summary["candidates"]["by_profile"]["all_profiles"]["1"]["labels"][s][f]["model_coverage"] for f, s in pairs]
    axes[0].scatter(range(4), coverage, marker="D", facecolors="none", edgecolors="#333333", s=50, label="Nonunknown label coverage")
    axes[0].set_xticks(range(4), ["Attention\nself", "Attention\nevidence", "Initiative\nself", "Initiative\nevidence"])
    axes[0].set_ylim(0, 1.15); axes[0].set_ylabel("Matches / all 72 planned candidate records")
    axes[0].set_title("Behavior-label agreement, not persona accuracy")
    route_costs = summary["costs"]["per_final_full_route"]
    generation = [route_costs[p][m]["mean_generation_tokens"] for p, m in conditions]
    labels_cost = [route_costs[p][m]["mean_label_tokens"] for p, m in conditions]
    axes[1].bar(range(4), generation, label="Generation call", color="#719ec3")
    axes[1].bar(range(4), labels_cost, bottom=generation, label="Evidence-label call", color="#dcac75")
    axes[1].set_xticks(range(4), names := ["S / direct", "S / selection", "L / direct", "L / selection"], rotation=15)
    axes[1].set_ylabel("Mean full-route tokens per final reply"); axes[1].set_title("No division by six candidates")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Semantic label diagnostics and complete route cost\nUnknown labels are not successful matches; reviewer labour remains unpriced")
    savefig(fig, out, "fig4_labels_cost")


def analyze(out):
    for coder in CODERS:
        if not (HERE / "reviews" / f"ratings_{coder}.json").exists():
            raise SystemExit("Both independent coding files are required; no provisional ratings will be published.")
    cases = {c["id"]: c for c in b.load(HERE / "cases.json")}
    profiles = {p["id"]: p for p in b.load(HERE / "profiles.json")}
    data = b.load(HERE / "run/results.json")
    calls = json.loads(gzip.decompress((HERE / "run/raw_calls.json.gz").read_bytes()))
    key_rows = b.load(HERE / "review_key.json")
    ratings, audit = {}, Audit()
    for coder in CODERS:
        rows = b.load(HERE / "reviews" / f"ratings_{coder}.json")
        ratings[coder] = {r["id"]: r for r in rows}
        audit.check(len(rows) == len(ratings[coder]), "unique rating IDs " + coder)
    refs, fixture_refs = validate(data, calls, cases, profiles, key_rows, ratings, audit)
    if audit.failures:
        out.mkdir(parents=True, exist_ok=True); b.save(out / "analysis_audit_report.json", audit.result())
        raise ValueError("Analysis input audit failed: " + "; ".join(audit.failures[:8]))
    summary = {"study_id": "persona_evaluation_2026_11_07", "protocol_sha256": b.sha(HERE / "protocol.json"),
        "counts": {"planned_final_outputs": 24, "delivered_final_outputs": sum(r["say"] is not None for r in data["records"]),
            "planned_candidate_records": 72, "generated_candidate_records": sum(len(r["candidates"]) for r in data["records"]),
            "unique_online_review_units": sum(r["kind"] == "online" for r in key_rows), "fixture_review_units": len(fixture_refs),
            "actual_api_calls": len(calls), "random_offline_diagnostics": sum(r["random_selected_id"] is not None for r in data["records"])},
        **online_metrics(data["records"], cases, profiles, refs, ratings),
        "calibration": calibration(fixture_refs, ratings), "reviewer_agreement": agreement(key_rows, ratings, cases),
        "costs": cost_analysis(calls, data["records"]),
        "notes": ["No naturalness score, composite score, usability rule or numeric persona acceptance threshold is defined.",
            "Formal, observational, mechanical, restrained and lively are descriptive styles, not a quality ordering.",
            "Persona and context means are ordinal summaries; distributions and unknown counts are retained.",
            "Grounding and boundary labels are reviewer judgments with three states; a high persona score cannot offset a violation.",
            "Main route comparisons use six paired scenarios per persona, each sampled once; no superiority or significance claim is made.",
            "Independent Codex tasks are model-assisted reviewers, not independent human truth."]}
    audit.check(summary["counts"]["planned_final_outputs"] == sum(summary["main"][p][m]["1"]["planned"] for p in PROFILES for m in METHODS), "main denominator stays 24")
    audit.check(summary["counts"]["planned_candidate_records"] == sum(summary["candidates"]["by_profile"][p]["1"]["quality_dimensions"]["planned"] for p in PROFILES), "candidate denominator stays 72")
    audit.check(summary["random_diagnostic"]["planned_offline_choices"] == 12, "random diagnostic remains separate")
    out.mkdir(parents=True, exist_ok=True)
    b.save(out / "summary.json", summary); b.save(out / "analysis_audit_report.json", audit.result())
    b.save(out / "analysis_environment.json", {"python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "font": "DejaVu Sans", "offline": True,
        "analysis_sha256": b.sha(Path(__file__)), "uncertainty": "paired descriptive differences only; no unplanned significance testing"})
    make_figures(summary, out)
    return summary, audit.result()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out", type=Path, default=HERE); args = p.parse_args()
    summary, audit = analyze(args.out)
    print(json.dumps({"counts": summary["counts"], "analysis_audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
