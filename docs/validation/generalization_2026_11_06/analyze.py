"""Offline analysis of the fixed generalization study; never sends model requests.

Every denominator is reconstructed from the frozen assignment plan. Incomplete
records therefore remain failures rather than disappearing from the analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROLES = ("neutral", "blue", "dragon")
ROLE_LABELS = {"neutral": "中性", "blue": "蓝鱼", "dragon": "白龙"}
FACT_CONDITIONS = tuple(f"{mode}_{role}" for role in ROLES for mode in ("plain", "text"))
MEMORY_CONDITIONS = ("supported", "irrelevant", "unresolved", "stale_corrected", "empty_corrected", "retracted")
MEMORY_LABELS = {"supported": "有效记忆", "irrelevant": "无关记忆", "unresolved": "未解决冲突",
                 "stale_corrected": "旧记忆＋当前更正", "empty_corrected": "空记忆＋当前更正", "retracted": "旧设置撤回"}
PROBE_CONDITIONS = ("updated_memory", "old_memory_current_correction", "empty")
PROBE_LABELS = {"updated_memory": "真实更新记忆", "old_memory_current_correction": "旧记忆＋当前更正", "empty": "空白对照"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_scorer(data_dir, protocol):
    candidates = (data_dir / "runner_snapshot.py", data_dir / "study.py", HERE / "study.py")
    scorer_path = next((p for p in candidates if p.exists() and sha(p) == protocol["runner_sha256"]), None)
    if scorer_path is None:
        raise ValueError("No study runner matching the frozen protocol hash; refusing to change the scorer")
    spec = importlib.util.spec_from_file_location("_generalization_frozen_scorer", scorer_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The frozen runner lives one level below its original location. Load its
    # framework imports from that run's archive rather than today's checkout;
    # the scorer itself is pure and does not instantiate a client or engine.
    archived_root = data_dir / "implementation_snapshot"
    original_path = list(sys.path)
    original_persona = {name: value for name, value in sys.modules.items()
                        if name == "persona" or name.startswith("persona.")}
    for name in original_persona:
        del sys.modules[name]
    sys.path.insert(0, str(archived_root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == "persona" or name.startswith("persona."):
                del sys.modules[name]
        sys.modules.update(original_persona)
    return module


def wilson(correct, total):
    """Descriptive 95% Wilson interval; not an independence guarantee."""
    if not total:
        return None
    z = 1.959963984540054
    p, z2 = correct / total, z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    width = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return [max(0.0, center - width), min(1.0, center + width)]


def percentile(values, quantile):
    index = (len(values) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    return values[low] * (high - index) + values[high] * (index - low) if low != high else values[low]


def bootstrap(differences, seed=20261106, resamples=4000):
    values = list(differences.values())
    if not values:
        return dict(difference=None, interval=None, clusters=0, resamples=resamples)
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return dict(difference=statistics.mean(values), interval=[percentile(samples, .025), percentile(samples, .975)],
                clusters=len(values), cluster_differences=differences, resamples=resamples,
                method="paired cluster percentile bootstrap; descriptive, no equivalence/significance claim")


def summarize_rows(rows):
    result = dict(assigned=len(rows), observed=sum(row["observed"] for row in rows),
                  missing=sum(not row["observed"] for row in rows))
    for field in ("strict_correct", "labelled_content_correct", "label_present", "label_is_last_line"):
        count = sum(bool(row.get("evaluation", {}).get(field)) for row in rows)
        result[field] = dict(correct=count, total=len(rows), rate=count / len(rows) if rows else None,
                             wilson_95=wilson(count, len(rows)))
    observed = [row["turn"] for row in rows if row["observed"]]
    for name, values in (("say_characters", [len(turn["say"]) for turn in observed]),
                         ("elapsed_s", [turn["elapsed_s"] for turn in observed if isinstance(turn.get("elapsed_s"), (int, float))])):
        result[name] = dict(observed=len(values), mean=statistics.mean(values) if values else None,
                            median=statistics.median(values) if values else None,
                            minimum=min(values) if values else None, maximum=max(values) if values else None)
    return result


def protocol_stats(rows):
    observed = [row for row in rows if row["observed"]]
    equal = sum("".join(row["turn"].get("delivered_chunks", [])) == row["turn"]["say"] for row in observed)
    return dict(assigned=len(rows), observed=len(observed), missing=len(rows) - len(observed),
                protocol_valid=sum(row["turn"].get("valid") is True for row in observed),
                delivery_equals_saved=equal,
                foreground_single_call=sum(row["turn"].get("calls_count") == 1 for row in observed),
                denominator_note="assigned outputs; missing outputs count as unsuccessful; ordinary-prompt outputs have no HEAD/SAY/TAIL validity")


def collect(data_dir):
    data_dir = Path(data_dir)
    protocol, plan, results = [load(data_dir / name) for name in ("protocol.json", "execution_plan.json", "results.json")]
    scorer = load_scorer(data_dir, protocol)
    raw = json.loads(gzip.decompress((data_dir / "raw_calls.json.gz").read_bytes()))
    assignments = {job["trial_id"]: job for job in plan}
    records = {record["trial_id"]: record for record in results["records"]}
    if len(assignments) != len(plan) or len(records) != len(results["records"]):
        raise ValueError("Duplicate assigned or recorded trial IDs")
    if not records.keys() <= assignments.keys():
        raise ValueError("Recorded trial outside the fixed assignment plan")
    audits = []
    def check(name, passed):
        audits.append(dict(check=name, passed=bool(passed)))
    check("protocol_hash", results["protocol_hash"] == sha(data_dir / "protocol.json"))
    check("plan_hash", results["plan_hash"] == sha(data_dir / "execution_plan.json"))
    check("raw_calls_hash", results["raw_calls_sha256"] == sha(data_dir / "raw_calls.json.gz"))
    check("raw_call_count", results["calls_count"] == len(raw))
    check("bounded_calls", len(raw) <= protocol["hard_api_request_limit"])
    raw_ids = [call["trial_id"] + "/" + call["call_id"] for call in raw]
    check("unique_raw_call_ids", len(raw_ids) == len(set(raw_ids)))
    saved_call_ids = [call_id for record in records.values() for call_id in record["call_ids"]]
    check("call_id_correspondence", sorted(saved_call_ids) == sorted(raw_ids))
    for relative, expected in protocol["source_hashes"].items():
        archived = data_dir / "implementation_snapshot" / relative
        check("source:" + relative, archived.exists() and sha(archived) == expected)
    rows, training_rows, workflows = [], [], []
    discrepancies = []

    def observe(job, condition, turn, target, stored=None, probe=None):
        observed = isinstance(turn, dict) and isinstance(turn.get("say"), str)
        evaluation = scorer.score(turn["say"], target) if observed else scorer.score("", target)
        if stored is not None and stored != evaluation:
            discrepancies.append(dict(trial_id=job["trial_id"], condition=condition, saved=stored, recomputed=evaluation))
        if observed and turn.get("stream_match") is not None:
            check("delivery_flag:" + job["trial_id"] + "/" + condition,
                  turn["stream_match"] == ("".join(turn.get("delivered_chunks", [])) == turn["say"]))
        return dict(trial_id=job["trial_id"], experiment=job["experiment"], condition=condition,
                    template=job.get("case", {}).get("template"), case_id=job.get("case", {}).get("case_id"),
                    variant=job.get("variant", job.get("case", {}).get("variant")), replicate=job.get("replicate"),
                    observed=observed, target=target, evaluation=evaluation, turn=turn if observed else None,
                    memory_injected=probe.get("memory_injected", []) if probe else None)

    for job in plan:
        record = records.get(job["trial_id"], {})
        if job["experiment"] in ("facts", "controlled_memory"):
            target = job["case"] if job["experiment"] == "facts" else scorer.memory_input(job["case"], job["condition"])[2]
            rows.append(observe(job, job["condition"], record.get("turn"), target, record.get("evaluation")))
        else:
            probes = {probe["condition"]: probe for probe in record.get("probes", [])}
            if len(probes) != len(record.get("probes", [])):
                raise ValueError("Duplicate probe condition in " + job["trial_id"])
            new = ((23, 6), (31, 7))[job["variant"]]
            for condition in PROBE_CONDITIONS:
                probe = probes.get(condition, {})
                target = dict(kind="label", gold="未知" if condition == "empty" else f"专注{new[0]}分钟,休息{new[1]}分钟", unit="")
                rows.append(observe(job, condition, probe.get("turn"), target, probe.get("evaluation"), probe))
            assigned_training = record.get("training", [])
            for index in range(4):
                turn = assigned_training[index].get("turn") if index < len(assigned_training) else None
                observed = isinstance(turn, dict) and isinstance(turn.get("say"), str)
                training_rows.append(dict(trial_id=job["trial_id"], condition="training", observed=observed, turn=turn))
            workflows.append(dict(trial_id=job["trial_id"], variant=job["variant"], replicate=job["replicate"],
                                  status=record.get("status", "missing"), rebuild_equal=record.get("rebuild_equal") is True,
                                  reopened_ledger_equal=record.get("reopened_ledger_equal") is True,
                                  traits_unchanged=record.get("traits_unchanged") is True,
                                  stages=record.get("stages", [])))
    check("saved_scores_match", not discrepancies)
    if discrepancies:
        raise ValueError("Saved scoring differs from frozen scorer: " + json.dumps(discrepancies[:2], ensure_ascii=False))
    failed = [audit["check"] for audit in audits if not audit["passed"]]
    if failed:
        raise ValueError("Artifact consistency check failed: " + ", ".join(failed[:10]))
    return protocol, plan, results, raw, rows, training_rows, workflows, audits


def build_summary(data_dir):
    protocol, plan, results, raw, rows, training_rows, workflows, audits = collect(data_dir)
    facts = [row for row in rows if row["experiment"] == "facts"]
    controlled = [row for row in rows if row["experiment"] == "controlled_memory"]
    learned = [row for row in rows if row["experiment"] == "learned_update"]
    framework = [row for row in rows if not (row["experiment"] == "facts" and row["condition"].startswith("plain_"))] + training_rows
    facts_summary = {condition: summarize_rows([row for row in facts if row["condition"] == condition]) for condition in FACT_CONDITIONS}
    memory_summary = {condition: summarize_rows([row for row in controlled if row["condition"] == condition]) for condition in MEMORY_CONDITIONS}
    learned_summary = {condition: summarize_rows([row for row in learned if row["condition"] == condition]) for condition in PROBE_CONDITIONS}
    for condition in MEMORY_CONDITIONS:
        selected = [row for row in controlled if row["condition"] == condition]
        memory_summary[condition]["explicit_unknown_conclusions"] = sum(row["evaluation"]["labelled_answer"] == "未知" for row in selected)
        memory_summary[condition]["non_unknown_labelled_conclusions"] = sum(row["evaluation"]["label_present"] and row["evaluation"]["labelled_answer"] != "未知" for row in selected)
    contrasts = {}
    for role in ROLES:
        cluster = defaultdict(lambda: {"text": [], "plain": []})
        for row in facts:
            mode, selected_role = row["condition"].split("_")
            if selected_role == role:
                cluster[row["template"]][mode].append(int(row["evaluation"]["strict_correct"]))
        differences = {name: statistics.mean(value["text"]) - statistics.mean(value["plain"]) for name, value in sorted(cluster.items())}
        contrasts["facts_text_minus_plain_" + role] = bootstrap(differences)
    cluster = defaultdict(lambda: {"stale_corrected": [], "empty_corrected": []})
    for row in controlled:
        if row["condition"] in ("stale_corrected", "empty_corrected"):
            cluster[row["template"]][row["condition"]].append(int(row["evaluation"]["strict_correct"]))
    contrasts["memory_stale_minus_empty_correction"] = bootstrap({name: statistics.mean(value["stale_corrected"]) - statistics.mean(value["empty_corrected"]) for name, value in sorted(cluster.items())})
    models, fingerprints, finishes, usage_fields = Counter(), Counter(), Counter(), Counter()
    usage_totals = Counter()
    usage_present = 0
    for call in raw:
        response = call.get("response") or {}
        models[response.get("model") or "unavailable"] += 1
        fingerprints[response.get("system_fingerprint") or "unavailable"] += 1
        finishes[response.get("finish_reason") or "unavailable"] += 1
        usage = response.get("usage")
        if isinstance(usage, dict):
            usage_present += 1
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(field), int):
                    usage_totals[field] += usage[field]
                    usage_fields[field] += 1
    quote_support = []
    for workflow in workflows:
        new = ((23, 6), (31, 7))[workflow["variant"]]
        updated = next((stage for stage in workflow["stages"] if stage.get("phase") == "updated"), {})
        judgments = updated.get("judgments", [])
        quotes = [quote.get("quote", "") for judgment in judgments for quote in judgment.get("evidence_quotes", [])]
        has_pair = lambda text: f"专注{new[0]}分钟" in text and f"休息{new[1]}分钟" in text
        quote_support.append(dict(trial_id=workflow["trial_id"], updated_judgments=len(judgments),
                                  new_pair_in_one_user_quote=any(has_pair(text) for text in quotes),
                                  new_pair_in_one_judgment=any(has_pair(item.get("judgment", "")) for item in judgments)))
    summary = dict(study_id=protocol["study_id"], inputs=dict(protocol_sha256=sha(Path(data_dir) / "protocol.json"),
        results_sha256=sha(Path(data_dir) / "results.json"), raw_calls_sha256=sha(Path(data_dir) / "raw_calls.json.gz")),
        design=dict(assigned_units=len(plan), recorded_units=len(results["records"]),
                    assigned_dialogue_outputs=len(rows) + len(training_rows),
                    observed_dialogue_outputs=sum(row["observed"] for row in rows + training_rows),
                    main_scored_outputs=len(rows), training_outputs=len(training_rows),
                    unit_errors=sum(record.get("status") != "ok" for record in results["records"]),
                    missing_units=len(plan) - len(results["records"])),
        calls=dict(actual=len(raw), hard_limit=protocol["hard_api_request_limit"],
                   errors=sum(call.get("status") != "ok" for call in raw),
                   requested_model=protocol["parameters"]["model"], actual_models=dict(models),
                   system_fingerprints=dict(fingerprints), finish_reasons=dict(finishes),
                   usage_present=usage_present, usage_missing=len(raw) - usage_present,
                   token_totals_known=dict(usage_totals), token_field_coverage=dict(usage_fields),
                   token_note="Only returned usage is summed. Missing usage is not zero; no cost estimate is inferred."),
        facts=facts_summary, controlled_memory=memory_summary, learned_update=learned_summary,
        learned_workflows=dict(assigned=len(workflows), rebuild_equal=sum(workflow["rebuild_equal"] for workflow in workflows),
                               reopened_ledger_equal=sum(workflow["reopened_ledger_equal"] for workflow in workflows),
                               traits_unchanged=sum(workflow["traits_unchanged"] for workflow in workflows),
                               updated_pair_in_user_quote=sum(item["new_pair_in_one_user_quote"] for item in quote_support),
                               updated_pair_in_judgment=sum(item["new_pair_in_one_judgment"] for item in quote_support),
                               quote_support=quote_support,
                               quote_note="Lexical presence in a retained whole quote/judgment only; not a semantic claim that stale memory was resolved."),
        contrasts=contrasts, framework_delivery=protocol_stats(framework),
        framework_groups={"facts": protocol_stats([row for row in framework if row.get("experiment") == "facts"]),
                          "controlled_memory": protocol_stats([row for row in framework if row.get("experiment") == "controlled_memory"]),
                          "learned_probes": protocol_stats(learned), "training": protocol_stats(training_rows)},
        audit=dict(checks=len(audits), failed=0, details=audits),
        limitations=["Strict scores jointly measure requested answer and final-line format; labelled content does not evaluate every assertion.",
                     "Wilson intervals are descriptive per-output intervals; paired differences resample templates/topics, not correlated instances.",
                     "Only 8 fact templates and 4 controlled-memory topics; learned workflows use 2 number-pair variants.",
                     "Plain and text are whole prompt packages, not a causal isolation experiment.",
                     "The synthetic conflict case explicitly states missing chronology; memory correction is a user assertion, not externally verified truth.",
                     "No human/personality gold labels, no confirmatory p-values, no proof of equivalence or general safety."])
    return summary, rows, training_rows, workflows


def ratio(cell):
    return f"{cell['correct']}/{cell['total']}"


def interval_text(interval):
    return "未定义" if interval is None else f"[{interval[0]*100:.1f}%, {interval[1]*100:.1f}%]"


def tables(summary):
    lines = ["# 固定方案的离线重算结果", "", "所有分母来自执行计划；缺失与异常不删除。严格通过同时要求结论内容、单位与末行格式，不能替代全文事实评价。", "",
             "## 新题对照", "", "| 提示与表达条件 | 严格结论 | 95% Wilson 描述区间 | 标签内结论正确 | 有结论标签 | 实际输出数 | 平均字数 | 平均秒数 |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for condition, group in summary["facts"].items():
        mean_len, mean_time = group["say_characters"]["mean"], group["elapsed_s"]["mean"]
        lines.append(f"| {condition} | {ratio(group['strict_correct'])} | {interval_text(group['strict_correct']['wilson_95'])} | {ratio(group['labelled_content_correct'])} | {ratio(group['label_present'])} | {group['observed']} | {mean_len:.1f} | {mean_time:.3f} |" if mean_len is not None and mean_time is not None else f"| {condition} | {ratio(group['strict_correct'])} | {interval_text(group['strict_correct']['wilson_95'])} | {ratio(group['labelled_content_correct'])} | {ratio(group['label_present'])} | {group['observed']} | — | — |")
    lines += ["", "## 受控记忆", "", "| 条件 | 严格结论 | 95% Wilson 描述区间 | 标签内结论正确 | 显式未知 | 实际输出数 |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for condition, group in summary["controlled_memory"].items():
        lines.append(f"| {MEMORY_LABELS[condition]} | {ratio(group['strict_correct'])} | {interval_text(group['strict_correct']['wilson_95'])} | {ratio(group['labelled_content_correct'])} | {group['explicit_unknown_conclusions']}/{group['assigned']} | {group['observed']} |")
    lines += ["", "## 真实固化与更新", "", "| 新会话探针 | 严格结论 | 95% Wilson 描述区间 | 标签内结论正确 | 实际输出数 |", "| --- | ---: | --- | ---: | ---: |"]
    for condition, group in summary["learned_update"].items():
        lines.append(f"| {PROBE_LABELS[condition]} | {ratio(group['strict_correct'])} | {interval_text(group['strict_correct']['wilson_95'])} | {ratio(group['labelled_content_correct'])} | {group['observed']} |")
    lines += ["", "## 配对描述差异", "", "下列区间按事实模板或记忆主题聚类重采样；百分点差异不用于等效性或显著性结论。", "", "| 对比 | 差异（百分点） | 95% 聚类 bootstrap 区间（百分点） | 聚类数 |", "| --- | ---: | --- | ---: |"]
    for name, contrast in summary["contrasts"].items():
        interval = contrast["interval"]
        lines.append(f"| {name} | {100*contrast['difference']:.1f} | [{100*interval[0]:.1f}, {100*interval[1]:.1f}] | {contrast['clusters']} |")
    lines += ["", "## 运行与交付", "", "| 指标 | 结果 |", "| --- | ---: |"]
    design, calls, framework = summary["design"], summary["calls"], summary["framework_delivery"]
    for name, value in (("已记录单元 / 已分配单元", f"{design['recorded_units']}/{design['assigned_units']}"),
                        ("实际输出 / 计划输出", f"{design['observed_dialogue_outputs']}/{design['assigned_dialogue_outputs']}"),
                        ("实际请求 / 硬上限", f"{calls['actual']}/{calls['hard_limit']}"),
                        ("请求错误", calls["errors"]), ("单元异常", design["unit_errors"]),
                        ("框架协议有效 / 框架计划输出", f"{framework['protocol_valid']}/{framework['assigned']}"),
                        ("框架交付一致 / 框架计划输出", f"{framework['delivery_equals_saved']}/{framework['assigned']}"),
                        ("框架前台一次调用 / 框架计划输出", f"{framework['foreground_single_call']}/{framework['assigned']}"),
                        ("真实训练账本重建保持 / 计划工作流", f"{summary['learned_workflows']['rebuild_equal']}/{summary['learned_workflows']['assigned']}"),
                        ("重新打开账本保持 / 计划工作流", f"{summary['learned_workflows']['reopened_ledger_equal']}/{summary['learned_workflows']['assigned']}"),
                        ("人格参数保持 / 计划工作流", f"{summary['learned_workflows']['traits_unchanged']}/{summary['learned_workflows']['assigned']}"),
                        ("新数字双值保留在同一完整用户引用 / 计划工作流", f"{summary['learned_workflows']['updated_pair_in_user_quote']}/{summary['learned_workflows']['assigned']}"),
                        ("有用量记录 / 实际请求", f"{calls['usage_present']}/{calls['actual']}"),
                        ("已返回用量的 token 总计", calls["token_totals_known"].get("total_tokens", "不可用")),
                        ("材料对应检查", f"{summary['audit']['checks']}/{summary['audit']['checks']}")):
        lines.append(f"| {name} | {value} |")
    lines += ["", "请求模型：`" + calls["requested_model"] + "`。实际返回模型：`" + json.dumps(calls["actual_models"], ensure_ascii=False) + "`。", "", "普通提示的文本由同一原始增量直接拼接，交付一致性不作为独立框架验证；HEAD/SAY/TAIL 协议分母仅包括框架输出。原始用量缺失不是零，现有统计不推算费用。", ""]
    return "\n".join(lines)


def figures(out, summary, rows, workflows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC") if name in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": preferred, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.titleweight": "bold", "savefig.dpi": 180})
    destination = Path(out) / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    teal, amber = "#237c82", "#bd7036"

    def finish(fig, name, note):
        fig.text(.02, .015, note, fontsize=8, color="#555555", va="bottom")
        fig.tight_layout(rect=(0, .075, 1, .98))
        fig.savefig(destination / name, metadata={"Software": "persona-duplex offline study analysis"})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), sharey=True)
    for axis, metric, title in zip(axes, ("strict_correct", "labelled_content_correct"), ("末行格式与答案共同通过", "标签内答案正确（不要求末行）")):
        for offset, mode, label, color in ((-.19, "plain", "普通提示", amber), (.19, "text", "默认 text", teal)):
            groups = [summary["facts"][f"{mode}_{role}"][metric] for role in ROLES]
            positions = [index + offset for index in range(3)]
            values = [group["rate"] for group in groups]
            errors = [[group["rate"] - group["wilson_95"][0] for group in groups],
                      [group["wilson_95"][1] - group["rate"] for group in groups]]
            bars = axis.bar(positions, values, .34, color=color, label=label, yerr=errors, capsize=3)
            for bar, group in zip(bars, groups):
                axis.text(bar.get_x()+bar.get_width()/2, group["rate"]+.035, ratio(group), ha="center", fontsize=8,
                          bbox=dict(facecolor="white", edgecolor="none", pad=.3))
        axis.set_xticks(range(3), [ROLE_LABELS[role] for role in ROLES])
        axis.set_ylim(0, 1.18); axis.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        axis.set_title(title); axis.grid(axis="y", alpha=.17); axis.set_axisbelow(True)
    axes[0].set_ylabel("通过比例"); axes[1].legend(loc="lower right")
    finish(fig, "fig1_new_fact_tasks.png", "每条件16个预定实例；误差线为逐输出 Wilson 描述区间。两种提示组织的比较不证明推理隔离或全文事实正确。")

    fig, axis = plt.subplots(figsize=(10.8, 5.0))
    for index, condition in enumerate(MEMORY_CONDITIONS):
        cell = summary["controlled_memory"][condition]["strict_correct"]
        color = teal if condition in ("supported", "stale_corrected", "empty_corrected") else amber
        axis.bar(index, cell["rate"], color=color, width=.62,
                 yerr=[[cell["rate"] - cell["wilson_95"][0]], [cell["wilson_95"][1] - cell["rate"]]], capsize=4)
        axis.text(index, cell["rate"]+.035, ratio(cell), ha="center",
                  bbox=dict(facecolor="white", edgecolor="none", pad=.3))
    axis.set_xticks(range(6), [MEMORY_LABELS[c].replace("＋", "＋\n") for c in MEMORY_CONDITIONS])
    axis.set_ylim(0, 1.18); axis.set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
    axis.set_ylabel("严格通过比例"); axis.set_title("受控记忆：按当前证据作答，证据不足时保持未知")
    axis.grid(axis="y", alpha=.17); axis.set_axisbelow(True)
    finish(fig, "fig2_controlled_memory.png", "每条件8个预定实例；青色条件应给出具体新值，棕色条件应回答未知。误差线为 Wilson 描述区间。")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    axis = axes[0]
    for index, condition in enumerate(PROBE_CONDITIONS):
        cell = summary["learned_update"][condition]["strict_correct"]
        axis.bar(index, cell["rate"], color=teal if condition != "empty" else amber, width=.6,
                 yerr=[[cell["rate"] - cell["wilson_95"][0]], [cell["wilson_95"][1] - cell["rate"]]], capsize=4)
        axis.text(index, cell["rate"]+.035, ratio(cell), ha="center",
                  bbox=dict(facecolor="white", edgecolor="none", pad=.3))
    axis.set_xticks(range(3), ["真实更新\n记忆", "旧记忆＋\n当前更正", "空白\n对照"])
    axis.set_ylim(0,1.18); axis.set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
    axis.set_ylabel("严格通过比例"); axis.set_title("独立新会话的更新探针")
    axis.grid(axis="y",alpha=.17); axis.set_axisbelow(True)
    ordered = sorted(workflows, key=lambda item:(item["variant"],item["replicate"]))
    lookup = {(row["trial_id"],row["condition"]):row["evaluation"]["strict_correct"] for row in rows if row["experiment"] == "learned_update"}
    matrix = [[int(lookup.get((workflow["trial_id"], condition),False)) for condition in PROBE_CONDITIONS] for workflow in ordered]
    axes[1].imshow(matrix, vmin=0, vmax=1, cmap=matplotlib.colors.ListedColormap(["#eee6db",teal]), aspect="auto")
    for y, values in enumerate(matrix):
        for x, value in enumerate(values): axes[1].text(x,y,"通过" if value else "未过",ha="center",va="center",color="white" if value else "#694d36",fontsize=9)
    axes[1].set_xticks(range(3), ["更新记忆", "当前更正", "空白"])
    axes[1].set_yticks(range(len(ordered)), [f"变体{item['variant']+1} / 重复{item['replicate']+1}" for item in ordered])
    axes[1].set_title("六条真实训练工作流")
    finish(fig, "fig3_learned_memory_update.png", "每工作流4轮真实训练、两次固化、3个独立探针；两组新数字各3次重复。空白对照的目标答案为未知。")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    selected = [[row["turn"] for row in rows if row["experiment"]=="facts" and row["condition"]==condition and row["observed"]] for condition in FACT_CONDITIONS]
    labels = [ROLE_LABELS[role]+("\n普通" if mode=="plain" else "\ntext") for role in ROLES for mode in ("plain","text")]
    for axis, key, title in ((axes[0], "length", "SAY 字符数"),(axes[1], "elapsed_s", "端到端响应秒数")):
        values = [[len(turn["say"]) if key=="length" else turn["elapsed_s"] for turn in group] for group in selected]
        # Matplotlib receives a NaN only to draw a visibly empty, unobserved group.
        boxes = axis.boxplot([value or [float("nan")] for value in values], patch_artist=True, tick_labels=labels, widths=.55)
        for index, patch in enumerate(boxes["boxes"]): patch.set_facecolor(amber if index%2==0 else teal); patch.set_alpha(.72)
        axis.set_title(title); axis.grid(axis="y", alpha=.17); axis.set_axisbelow(True)
    finish(fig,"fig4_length_and_latency.png","仅统计实际返回的事实任务输出；字符按 Python len(str) 计。字数与时延不作为人格效果或正确性的替代指标。")
    return [str(path.relative_to(out)).replace("\\","/") for path in sorted(destination.glob("fig*.png"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE, help="Frozen protocol, assignment, results, and raw calls directory")
    parser.add_argument("--out", type=Path, default=HERE, help="Derived summary, tables, and figure directory")
    parser.add_argument("--no-figures", action="store_true", help="Recompute data only, without optional plotting dependencies")
    args = parser.parse_args()
    summary, rows, training_rows, workflows = build_summary(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        summary["figures"] = figures(args.out,summary,rows,workflows)
    dump(args.out/"summary.json",summary)
    (args.out/"results_tables.md").write_text(tables(summary),encoding="utf-8",newline="\n")
    print(json.dumps({"assigned_units":summary["design"]["assigned_units"],"observed_outputs":summary["design"]["observed_dialogue_outputs"],
                      "actual_calls":summary["calls"]["actual"],"audit_checks":summary["audit"]["checks"],"network":False},ensure_ascii=False))


if __name__ == "__main__": main()
