"""Analyze the follow-up study without model calls or access to credentials.

Run from the repository root with Python >= 3.11 and matplotlib installed.
Intervals describe this task set; trials and multiple turns are not assumed
to be independent samples of the population of all conversations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics


BASE = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_REPS = 4000
LABELS = {
    "vanilla": "普通提示", "structured_base": "结构化基线",
    "vanilla_base": "普通提示·基线", "vanilla_blue": "普通提示·蓝鱼",
    "vanilla_dragon": "普通提示·白龙", "structured_blue": "结构化·蓝鱼",
    "structured_dragon": "结构化·白龙",
    "blue_whale": "蓝色大肥鱼", "white_dragon": "白色龙娘",
    "persona_base": "分步基线", "persona_blue": "分步蓝鱼",
    "persona_dragon": "分步白龙", "empty": "空白记忆",
    "correct": "正确记忆", "irrelevant": "无关记忆",
    "contradictory": "冲突记忆", "healed": "明确原谅",
    "stop": "明确停提", "ambiguous": "模糊反馈",
    "formal_neutral": "正式但无受伤证据",
}
FACT_ORDER = ["vanilla_base", "vanilla_blue", "vanilla_dragon",
              "structured_base", "structured_blue", "structured_dragon",
              "persona_base", "persona_blue", "persona_dragon"]


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    proportion = successes / total
    scale = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / scale
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / scale
    return [max(0, center - half), min(1, center + half)]


def case_id(record):
    return str(record.get("case_id", record.get("trial_id", "unknown")))


def evaluation(record, name):
    value = (record.get("evaluation") or {}).get(name)
    if isinstance(value, bool):
        return value
    # State names are retained as names, rather than coerced to truthy strings.
    if name == "expected_state" and isinstance(value, str):
        actual = (record.get("evaluation") or {}).get("actual_state")
        if actual is not None:
            return value == actual
    return None


def corrected_conclusion(text):
    matches=re.findall(r"结论\s*[:：]\s*([^\n]+)",text)
    if not matches:return None
    # Preserve decimal points. Only terminal ASCII sentence stops are removed.
    return re.sub(r"[\s*`，。!！]","",matches[-1]).rstrip(".")


def corrected_answer(text, case):
    result=corrected_conclusion(text)
    if result is None:return None
    if case["kind"]=="number":
        number=re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:元|%)?",result.replace(",",""))
        return bool(number and float(number.group(1))==float(case["gold"][0]))
    if case["kind"]=="time":
        actual=re.fullmatch(r"(\d{1,2}):(\d{2})",result)
        if not actual:return False
        actual_time=tuple(map(int,actual.groups()))
        return any(actual_time==tuple(map(int,gold.split(":"))) for gold in case["gold"])
    return result in case["gold"]


def add_corrected_scores(data):
    amendment={"status":"post-hoc deterministic scorer implementation correction",
               "rule":"Preserve internal decimal points; remove only trailing ASCII sentence-stop dots. All other strict-format and gold rules unchanged.",
               "original_result_file_unchanged":True,"runner_unchanged":True,
               "scorer_source_sha256":hashlib.sha256((inspect.getsource(corrected_conclusion)+inspect.getsource(corrected_answer)).encode("utf-8")).hexdigest(),
               "affected_trials":[]}
    for record in data.get("records",[]):
        if record.get("experiment")!="facts":continue
        ev=record.setdefault("evaluation",{})
        turns=record.get("turns",[])
        if turns and record.get("gold"):
            ev["say_correct_decimal_fixed"]=corrected_answer(turns[-1].get("say",""),record["gold"]) is True
            head=turns[-1].get("head") or {}
            ev["direct_correct_decimal_fixed"]=corrected_answer(head.get("direct",""),record["gold"]) if head else None
        else:
            ev["say_correct_decimal_fixed"]=False
            ev["direct_correct_decimal_fixed"]=None
        changes={}
        for original,fixed in [("say_correct","say_correct_decimal_fixed"),("direct_correct","direct_correct_decimal_fixed")]:
            if ev.get(original)!=ev[fixed]:changes[original]={"original":ev.get(original),"corrected":ev[fixed]}
        if changes:amendment["affected_trials"].append({"trial_id":record.get("trial_id"),"condition":record.get("condition"),"case_id":record.get("case_id"),"changes":changes})
    amendment["affected_trial_count"]=len(amendment["affected_trials"])
    return amendment


def rate(records, name, include_failures=False):
    values = []
    missing = 0
    for record in records:
        value = evaluation(record, name)
        if record.get("status") == "error" and include_failures:
            value = False
        if value is None:
            missing += 1
        else:
            values.append(value)
    successes = sum(values)
    return {"successes": successes, "n": len(values), "missing": missing,
            "rate": successes / len(values) if values else None,
            "wilson95": wilson(successes, len(values)),
            "scope": "record-level descriptive; repeated cases are correlated"}


def clustered_summary(records, value_function):
    by_case = defaultdict(list)
    for record in records:
        value = value_function(record)
        if value is not None and math.isfinite(value):
            by_case[case_id(record)].append(value)
    if not by_case:
        return {"mean": None, "bootstrap95": [None, None], "cases": 0, "n": 0}
    # Repetitions stay within each case. Each case receives equal weight.
    case_means = [statistics.mean(v) for _, v in sorted(by_case.items())]
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [statistics.mean(rng.choices(case_means, k=len(case_means)))
             for _ in range(BOOTSTRAP_REPS)]
    return {"mean": statistics.mean(case_means),
            "bootstrap95": [percentile(draws, .025), percentile(draws, .975)],
            "cases": len(case_means), "n": sum(map(len, by_case.values())),
            "scope": "equal-case means; percentile case-cluster bootstrap"}


def paired_case_difference(reference, candidate, metric):
    def group(records):
        output = defaultdict(list)
        for record in records:
            value = evaluation(record, metric)
            if record.get("status") == "error":
                value = False
            if value is not None:
                output[case_id(record)].append(float(value))
        return output
    left, right = group(reference), group(candidate)
    shared = sorted(left.keys() & right.keys())
    differences = [statistics.mean(right[k]) - statistics.mean(left[k]) for k in shared]
    if not shared:
        return {"difference": None, "bootstrap95": [None, None], "paired_cases": 0}
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [statistics.mean(rng.choices(differences, k=len(differences)))
             for _ in range(BOOTSTRAP_REPS)]
    return {"difference": statistics.mean(differences),
            "bootstrap95": [percentile(draws, .025), percentile(draws, .975)],
            "paired_cases": len(shared), "case_ids": shared,
            "unpaired_reference_cases": sorted(left.keys() - right.keys()),
            "unpaired_candidate_cases": sorted(right.keys() - left.keys()),
            "scope": "candidate minus reference; matched case means; no population claim"}


def trial_length(record):
    lengths = [len(t["say"]) for t in record.get("turns", [])
               if isinstance(t.get("say"), str)]
    return mean(lengths)


def elapsed(record):
    values = [t.get("elapsed_s") for t in record.get("turns", [])
              if isinstance(t.get("elapsed_s"), (int, float))]
    return sum(values) if values else None


def trait_setting(record):
    changed = record.get("config_changed") or {}
    if changed.get("name") in ("intervention", "brevity"):
        return changed["name"], float(changed["value"])
    for name in ("intervention", "brevity"):
        if name in changed:
            return name, float(changed[name])
    match = re.search(r"(intervention|brevity)[_=: -]+(0(?:\.\d+)?|1(?:\.0+)?)", str(record.get("condition", "")))
    return (match.group(1), float(match.group(2))) if match else (None, None)


def summarize(data):
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    trial_ids = [r["trial_id"] for r in records if r.get("trial_id") is not None]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("duplicate trial_id values would double-count resumed trials")
    groups = defaultdict(list)
    for record in records:
        groups[(str(record.get("experiment", "unknown")), str(record.get("condition", "unknown")))].append(record)
    output = {"analysis": {"bootstrap_seed": BOOTSTRAP_SEED,
                            "bootstrap_replicates": BOOTSTRAP_REPS,
                            "python": platform.python_version(),
                            "unit": "trial record; correlated turns are aggregated within trial",
                            "intervals": "Wilson: descriptive trial ratios; bootstrap: case-cluster resampling"},
              "record_count": len(records), "status": dict(Counter(r.get("status", "missing") for r in records)),
              "records_without_trial_id": len(records)-len(trial_ids),
              "planned_trials": (data.get("manifest") or {}).get("planned_trials"),
              "groups": {}, "facts_paired_vs_structured_base": {},"facts_paired_decimal_fixed_vs_structured_base": {}}
    metrics = ["direct_correct", "say_correct", "direct_correct_decimal_fixed", "say_correct_decimal_fixed", "answer_format_present", "memory_correct", "recall",
               "contradiction_obeyed", "unsupported_claim", "expected_state",
               "no_sore_point", "checker_correct"]
    for (experiment, condition), selected in sorted(groups.items()):
        turns = [t for r in selected for t in r.get("turns", [])]
        validity = [t["valid"] for t in turns if isinstance(t.get("valid"), bool)]
        stream = [t["stream_match"] for t in turns if isinstance(t.get("stream_match"), bool)]
        exported_calls = [len(r.get("call_ids", r.get("calls", []))) for r in selected]
        entry = {"experiment": experiment, "condition": condition, "n": len(selected),
                 "cases": len({case_id(r) for r in selected}),
                 "status": dict(Counter(r.get("status", "missing") for r in selected)),
                 "turn_count": len(turns),
                 "protocol_valid": {"successes": sum(validity), "n": len(validity), "missing": len(turns)-len(validity)},
                 "stream_match": {"successes": sum(stream), "n": len(stream), "missing": len(turns)-len(stream)},
                 "exported_calls": sum(exported_calls),
                 "say_chars": clustered_summary(selected, trial_length),
                 "elapsed_seconds": clustered_summary(selected, elapsed),
                 "displace": dict(Counter(str((t.get("head") or {}).get("displace", "missing")) for t in turns)),
                 "evaluation": {m: rate(selected, m) for m in metrics}}
        if experiment == "facts":
            entry["say_answer_marker_coverage"] = rate(selected, "answer_format_present", include_failures=True)
            if not condition.startswith("vanilla"):
                parsed = entry["evaluation"]["direct_correct"]["n"]
                entry["direct_parse_coverage"] = {
                    "successes": parsed, "n": len(selected), "missing": 0,
                    "rate": parsed / len(selected) if selected else None,
                    "wilson95": wilson(parsed, len(selected)),
                    "scope": "conclusion-marker rule scorable; not whole direct semantic correctness"}
                entry["evaluation"]["direct_correct"]["scope"] = "conditional on conclusion-marker rule scorable direct only"
                entry["evaluation"]["direct_correct_decimal_fixed"]["scope"] = "conditional on conclusion-marker rule scorable direct only; decimal cleaning corrected"
        for m in ("say_correct", "say_correct_decimal_fixed", "memory_correct", "expected_state", "checker_correct"):
            applicable = ((experiment == "facts" and m == "say_correct")
                          or (experiment == "facts" and m == "say_correct_decimal_fixed")
                          or (experiment == "memory" and m == "memory_correct")
                          or (experiment == "repair" and m == "expected_state")
                          or (experiment == "checker" and m == "checker_correct"))
            if entry["evaluation"][m]["n"] or applicable:
                entry["evaluation"][m + "_including_errors"] = rate(selected, m, include_failures=True)
        output["groups"][experiment + "/" + condition] = entry
    reference = groups.get(("facts", "structured_base"), [])
    for (experiment, condition), selected in sorted(groups.items()):
        if experiment == "facts" and condition != "structured_base":
            output["facts_paired_vs_structured_base"][condition] = paired_case_difference(reference, selected, "say_correct")
            output["facts_paired_decimal_fixed_vs_structured_base"][condition] = paired_case_difference(reference, selected, "say_correct_decimal_fixed")
    return output


def summarize_supplement(data):
    records=data.get("records",[])
    normalized={"learned":[],"empty":[]}
    diagnostics=[]
    all_turns=[]
    for record in records:
        all_turns.extend(record.get("training",[]))
        judgments=record.get("consolidated_judgments",[])
        evidence=set(record.get("evidence_turn_ids",[]))
        has_numbers=lambda text: bool(re.search(r"(?<!\d)17\s*分钟",text) and re.search(r"(?<!\d)3\s*分钟",text))
        replay=record.get("replay",{})
        diagnostics.append({"replicate":record.get("replicate"),"status":record.get("status"),
                            "judgment_count":len(judgments),
                            "judgment_target_pair_occurrence":any(has_numbers(j.get("judgment","")) for j in judgments),
                            "judgment_evidence_ids":[j.get("evidence_ids",[]) for j in judgments],
                            "evidence_ids_valid_recomputed":bool(judgments) and all(bool(j.get("evidence_ids")) and set(j["evidence_ids"])<=evidence for j in judgments),
                            "traits_unchanged_recomputed":bool(record.get("traits_before")) and record.get("traits_before")==record.get("traits_after"),
                            "replay_judgments_equal_recomputed":"user_judgments" in replay.get("before",{}) and replay.get("before",{}).get("user_judgments")==replay.get("after",{}).get("user_judgments"),
                            "replay_traits_equal_recomputed":bool(replay.get("before",{}).get("traits")) and replay.get("before",{}).get("traits")==replay.get("after",{}).get("traits")})
        for condition in normalized:
            found=[p for p in record.get("probes",[]) if p.get("condition")==condition]
            if len(found)>1:raise ValueError("duplicate supplement probe within replicate")
            probe=found[0] if found else {}
            turn=probe.get("turn",{});all_turns.extend([turn] if turn else [])
            ev=probe.get("evaluation",{})
            normalized[condition].append({"case_id":str(record.get("replicate")),"status":"ok" if found else "error",
                "evaluation":{"recall_both":ev.get("recall_both",False),"unknown":ev.get("unknown",False),
                              "format_present":ev.get("format_present",False),
                              "body_target_pair_occurrence":has_numbers(turn.get("say","")),
                              "empty_history":probe.get("history_before")==[]}})
    output={"replicates":len(records),"planned_replicates":data.get("protocol",{}).get("replicates"),
            "status":dict(Counter(r.get("status","missing") for r in records)),
            "dialogue_outputs":len(all_turns),"calls":sum(len(r.get("call_ids",[])) for r in records),
            "valid_turns":sum(t.get("valid") is True for t in all_turns),
            "stream_match_turns":sum(t.get("stream_match") is True for t in all_turns),
            "diagnostics":diagnostics,"conditions":{}}
    for condition,selected in normalized.items():
        output["conditions"][condition]={m:rate(selected,m,include_failures=True) for m in
                                         ["recall_both","unknown","format_present","body_target_pair_occurrence","empty_history"]}
    output["posthoc_body_pair_difference"]=paired_case_difference(normalized["empty"],normalized["learned"],"body_target_pair_occurrence")
    output["posthoc_scope"]="body_target_pair_occurrence is a post-hoc lexical occurrence check, not a semantic correctness score"
    output["time_scope"]="same-run cross-session, not cross-day retention"
    return output


def summarize_annotations(data, records):
    by_id={r["trial_id"]:r for r in records}
    output={"scope":data.get("scope"),"facts":{},"advice":{}}
    for category,metric in [("facts","label"),("advice","action_suggestion")]:
        ratings=(data.get(category) or {}).get("ratings",[])
        ids=[r["trial_id"] for r in ratings]
        if len(set(ids))!=len(ids):raise ValueError("duplicate annotated trial_id")
        grouped=defaultdict(list)
        for rating in ratings:
            original=by_id[rating["trial_id"]]
            if original["condition"]!=rating["condition"]:raise ValueError("annotation condition mismatch")
            grouped[original["condition"]].append(rating)
        for condition,selected in sorted(grouped.items()):
            n=len(selected)
            k=sum(r[metric]=="correct" if category=="facts" else r[metric] is True for r in selected)
            info={"successes":k,"n":n,"rate":k/n if n else None,"wilson95":wilson(k,n)}
            if category=="facts":
                info["labels"]=dict(Counter(r["label"] for r in selected))
                info["matched_subset_strict_successes"]=sum(by_id[r["trial_id"]]["evaluation"]["say_correct_decimal_fixed"] is True for r in selected)
                info["replicates"]=sorted({r["replicate"] for r in selected})
            output[category][condition]=info
    return output


def setup_plots():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"] if f in available]
    plt.rcParams.update({"font.family": preferred or ["sans-serif"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.facecolor": "white"})
    return plt, matplotlib.__version__


def label(condition):
    return LABELS.get(condition, condition)


def rate_panel(ax, conditions, values, title, color="#286b91"):
    x, ys, lo, hi, names = [], [], [], [], []
    for condition, value in zip(conditions, values):
        if value["rate"] is None:
            continue
        x.append(len(x)); ys.append(value["rate"])
        lo.append(max(0, value["rate"]-value["wilson95"][0]));hi.append(max(0, value["wilson95"][1]-value["rate"]))
        names.append(label(condition))
        ax.text(x[-1], value["rate"]+.035, f'{value["successes"]}/{value["n"]}', ha="center", fontsize=9)
    if x:
        ax.errorbar(x,ys,yerr=[lo,hi],fmt="o",markersize=7,capsize=4,color=color)
    else:
        ax.text(.5,.5,"无可评分记录",ha="center",transform=ax.transAxes)
    ax.set_xticks(x,names,rotation=24,ha="right")
    ax.set_ylim(-.04,1.18);ax.set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
    ax.set_title(title);ax.grid(axis="y",alpha=.2)


def make_figures(data, summary, out):
    plt, version = setup_plots()
    summary["analysis"]["matplotlib"] = version
    records = data.get("records", [])
    figures = out/"figures";figures.mkdir(parents=True,exist_ok=True)
    names = []
    def save(fig, name, caption):
        fig.text(.02,.015,caption,fontsize=8,color="#555555")
        fig.tight_layout(rect=(0,.075,1,.96))
        fig.savefig(figures/name,dpi=200,bbox_inches="tight")
        plt.close(fig); names.append("figures/"+name)
    facts = [v for v in summary["groups"].values() if v["experiment"]=="facts"]
    if facts:
        ordered = sorted(facts,key=lambda v: FACT_ORDER.index(v["condition"]) if v["condition"] in FACT_ORDER else 999)
        conditions=[v["condition"] for v in ordered]
        fig,axes=plt.subplots(1,2,figsize=(15,5.2))
        vals=[v["evaluation"]["say_correct_decimal_fixed_including_errors"] for v in ordered]
        rate_panel(axes[0],conditions,vals,"(a) 指定结论严格匹配率（数值规范化）")
        raw=[v["evaluation"]["say_correct"]["rate"] for v in ordered]
        axes[0].scatter([i+.1 for i in range(len(raw))],raw,marker="x",color="#777777",label="原始自动评分")
        axes[0].legend(loc="lower left",fontsize=8)
        pairs=summary["facts_paired_decimal_fixed_vs_structured_base"]
        valid=[(k,v) for k,v in pairs.items() if v["difference"] is not None]
        for i,(condition,v) in enumerate(valid):
            a,b=v["bootstrap95"];point=v["difference"]
            axes[1].errorbar(i,point,yerr=[[point-a],[b-point]],fmt="o",capsize=4,color="#b36b26")
            axes[1].annotate(f'k={v["paired_cases"]}',(i,point),xytext=(0,10),textcoords="offset points",ha="center",fontsize=8)
        axes[1].axhline(0,color="#666666",linewidth=1)
        axes[1].set_xticks(range(len(valid)),[label(k) for k,v in valid],rotation=24,ha="right")
        axes[1].set_title("(b) 相对结构化基线的修正严格匹配率差")
        axes[1].set_ylabel("匹配率差（候选 − 基线）");axes[1].grid(axis="y",alpha=.2)
        save(fig,"fig1_factual_accuracy.png","数值规范化保留内部小数点；其余格式规则与标准答案不变。左为描述性 Wilson 区间，右按 case 配对 bootstrap；不是全文事实正确率。")
    traits=[r for r in records if r.get("experiment")=="traits"]
    if traits:
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
        for ax,dimension,color in zip(axes,["intervention","brevity"],["#286b91","#b36b26"]):
            by_value=defaultdict(list)
            for record in traits:
                name,value=trait_setting(record)
                if name==dimension:by_value[value].append(record)
            for value,selected in sorted(by_value.items()):
                info=clustered_summary(selected,trial_length);point=info["mean"]
                if point is None:continue
                a,b=info["bootstrap95"]
                ax.errorbar(value,point,yerr=[[point-a],[b-point]],fmt="o",capsize=5,color=color)
                ax.annotate(f'n={info["n"]}, k={info["cases"]}',(value,point),xytext=(0,10),textcoords="offset points",ha="center",fontsize=8)
            ax.set_title(dimension+" 单因素取值与回复长度")
            ax.set_xlabel("参数值");ax.set_ylabel("每轮 SAY 平均字符数");ax.set_xlim(0,1);ax.grid(alpha=.2)
        save(fig,"fig2_single_factor_traits.png","每个 trial 先对其轮次求均值，再对 case 等权；误差线为 case-cluster bootstrap 95% 区间。n=记录数，k=任务数。")
    memory=[v for v in summary["groups"].values() if v["experiment"]=="memory"]
    if memory:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        panels=[("memory_correct_including_errors","(a) 记忆/未知结论任务正确率"),
                ("recall","(b) 结论槽目标词召回"),
                ("contradiction_obeyed","(c) 冲突记忆下遵循当前纠正"),
                ("unsupported_claim","(d) 缺证据时填写非未知结论（越低越好）")]
        for ax,(metric,title) in zip(axes.flat,panels):
            rate_panel(ax,[v["condition"] for v in memory],[v["evaluation"][metric] for v in memory],title)
        save(fig,"fig3_memory_controls.png","(a) 执行错误计失败；其余仅计保存 true/false 的记录。标注为成功数/可评分数，区间为描述性 Wilson 95%。(d) 仅检验结论槽，不是全文幻觉率。")
    repair=[v for v in summary["groups"].values() if v["experiment"]=="repair"]
    if repair:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        rate_panel(axes[0],[v["condition"] for v in repair],[v["evaluation"].get("expected_state_including_errors",v["evaluation"]["expected_state"]) for v in repair],"(a) 预期修复状态匹配")
        selected=[r for r in records if r.get("experiment")=="repair"]
        states=sorted({str((r.get("evaluation") or {}).get("actual_state","missing")) for r in selected})
        conditions=[v["condition"] for v in repair];bottom=[0]*len(conditions)
        for state in states:
            counts=[sum(str((r.get("evaluation") or {}).get("actual_state","missing"))==state and r.get("condition")==c for r in selected) for c in conditions]
            axes[1].bar(range(len(conditions)),counts,bottom=bottom,label=state)
            bottom=[a+b for a,b in zip(bottom,counts)]
        axes[1].set_xticks(range(len(conditions)),[label(c) for c in conditions],rotation=24,ha="right")
        axes[1].set_ylabel("记录数");axes[1].set_title("(b) 实际末态分布（含缺失）");axes[1].legend(fontsize=8)
        save(fig,"fig4_repair_states.png","每条记录为一轮用户反馈；前三组预置 repairing，正式文体组无预置事项。不是自然诱发受伤到修复的完整路径。区间仅作描述。")
    if facts:
        fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
        ordered=sorted(facts,key=lambda v:v["condition"])
        x=list(range(len(ordered)))
        for ax,key,title in [(axes[0],"elapsed_seconds","(a) 每条脚本的前台耗时"),(axes[1],"say_chars","(b) 每轮实际回复字符数")]:
            for i,v in enumerate(ordered):
                info=v[key];point=info["mean"]
                if point is None:continue
                a,b=info["bootstrap95"]
                ax.errorbar(i,point,yerr=[[point-a],[b-point]],fmt="o",capsize=4,color="#286b91")
            ax.set_xticks(x,[label(v["condition"]) for v in ordered],rotation=24,ha="right")
            ax.set_title(title);ax.grid(axis="y",alpha=.2)
        axes[0].set_ylabel("秒");axes[1].set_ylabel("字符（包含空格与标点）")
        save(fig,"fig5_latency_and_length.png","case 等权均值及 case-cluster bootstrap 95% 区间。耗时同时受协议调用数、输出长度和服务负载影响。")
    checker = [v for v in summary["groups"].values() if v["experiment"]=="checker"]
    if checker:
        checker=sorted(checker,key=lambda v:v["condition"])
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
        rate_panel(axes[0],["无冲突候选" if v["condition"]=="correct" else "有冲突候选" for v in checker],
                   [v["evaluation"]["checker_correct_including_errors"] for v in checker],
                   "(a) 独立候选核查任务的标签匹配率")
        columns=["consistent","inconsistent","invalid","error"]
        matrix=[]
        for condition in ("correct","incorrect"):
            counts=Counter()
            for record in records:
                if record.get("experiment")!="checker" or record.get("condition")!=condition:continue
                predicted=(record.get("checker_output") or {}).get("verdict")
                predicted="error" if record.get("status")=="error" else predicted if predicted in columns[:2] else "invalid"
                counts[predicted]+=1
            matrix.append([counts[c] for c in columns])
        image=axes[1].imshow(matrix,cmap="Blues",aspect="auto",vmin=0)
        for i,row in enumerate(matrix):
            for j,value in enumerate(row):axes[1].text(j,i,str(value),ha="center",va="center",color="black")
        axes[1].set_xticks(range(4),["一致","不一致","无效标签","执行错误"])
        axes[1].set_yticks(range(2),["真值：一致","真值：不一致"])
        axes[1].set_title("(b) 核查混淆计数");fig.colorbar(image,ax=axes[1],label="记录数")
        save(fig,"fig6_checker_controls.png","固定前提与人工指定候选；它检验局部核查器，不代表真实生成链路的综合安全性。点旁为正确/全部候选数，区间仅作描述。")
    if facts:
        ordered=sorted(facts,key=lambda v:FACT_ORDER.index(v["condition"]) if v["condition"] in FACT_ORDER else 999)
        fig,axes=plt.subplots(1,2,figsize=(14,4.8))
        rate_panel(axes[0],[v["condition"] for v in ordered],
                   [v["say_answer_marker_coverage"] for v in ordered],"(a) SAY 结论标记覆盖率")
        direct=[v for v in ordered if "direct_parse_coverage" in v]
        rate_panel(axes[1],[v["condition"] for v in direct],
                   [v["direct_parse_coverage"] for v in direct],"(b) direct 规则可评分覆盖率")
        save(fig,"fig7_answer_coverage.png","分母为相应条件全部记录，执行错误计未覆盖。缺少结论标记不能推出语义判断错误；direct 正确率只能在可评分子集解释。")
    supplement=summary.get("supplement")
    if supplement:
        fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
        conditions=["learned","empty"]
        for offset,metric,color,display in [(-.09,"recall_both","#286b91","严格结论槽双数字召回"),(.09,"body_target_pair_occurrence","#b36b26","全文双数字出现（事后核对）")]:
            for i,c in enumerate(conditions):
                v=supplement["conditions"][c][metric];point=v["rate"];a,b=v["wilson95"]
                if point is None:continue
                axes[0].errorbar(i+offset,point,yerr=[[max(0,point-a)],[max(0,b-point)]],fmt="o",capsize=4,color=color,label=display if i==0 else None)
                axes[0].annotate(f'{v["successes"]}/{v["n"]}',(i+offset,point),xytext=(0,10 if offset>0 else -16),textcoords="offset points",ha="center",fontsize=8)
        axes[0].set_xticks([0,1],["载入实际固化判断","空白记忆"]);axes[0].set_ylim(-.12,1.16)
        axes[0].set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
        axes[0].set_title("(a) 同次运行的新会话探针");axes[0].legend(fontsize=8);axes[0].grid(axis="y",alpha=.2)
        checks=[("judgment_target_pair_occurrence","判断保留两数字"),("evidence_ids_valid_recomputed","证据编号有效"),
                ("traits_unchanged_recomputed","traits 未改变"),("replay_judgments_equal_recomputed","判断重建一致"),("replay_traits_equal_recomputed","traits 重建一致")]
        diagnostics=supplement["diagnostics"]
        infos=[]
        for key,display in checks:
            count=sum(d[key] is True for d in diagnostics);n=len(diagnostics)
            infos.append({"successes":count,"n":n,"rate":count/n if n else None,"wilson95":wilson(count,n)})
        rate_panel(axes[1],[display for key,display in checks],infos,"(b) 固化与重建的已保存记录核对")
        save(fig,"fig8_consolidated_memory.png","四个独立重复；误差线为描述性 Wilson 95% 区间。全文出现为事后词面核对；参数初始 plasticity/drift 均为零，未改变不能独立验证成长禁用机制。")
    annotations=summary.get("annotations")
    if annotations:
        fig,axes=plt.subplots(1,2,figsize=(15,5.3))
        facts_order=[v for v in FACT_ORDER if v in annotations["facts"]]
        for offset,color,display in [(-.1,"#286b91","修正严格匹配（全部32条/组）"),(.1,"#b36b26","所问结论内容标签（rep0，8条/组）")]:
            for i,c in enumerate(facts_order):
                v=summary["groups"]["facts/"+c]["evaluation"]["say_correct_decimal_fixed"] if offset<0 else annotations["facts"][c]
                point=v["rate"];a,b=v["wilson95"]
                axes[0].errorbar(i+offset,point,yerr=[[max(0,point-a)],[max(0,b-point)]],fmt="o",capsize=3,color=color,label=display if i==0 else None)
                axes[0].annotate(f'{v["successes"]}/{v["n"]}',(i+offset,point),xytext=(0,-14 if offset<0 else 9),textcoords="offset points",ha="center",fontsize=7)
        axes[0].set_xticks(range(len(facts_order)),[label(c) for c in facts_order],rotation=28,ha="right")
        axes[0].set_ylim(-.05,1.18);axes[0].set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
        axes[0].set_title("(a) 严格格式与所问结论内容的描述性诊断");axes[0].legend(loc="lower left",fontsize=8);axes[0].grid(axis="y",alpha=.2)
        conditions=sorted(annotations["advice"],key=lambda c:float(c.rsplit("_",1)[-1]))
        rate_panel(axes[1],[c.rsplit("_",1)[-1] for c in conditions],[annotations["advice"][c] for c in conditions],"(b) intervention 与具体行动建议出现率")
        axes[1].set_xlabel("intervention 参数值")
        save(fig,"fig9_annotation_diagnostics.png","事后单 Codex 代理、评分时隐藏条件；内容标签仅评所问结论，不评全文。左两系列样本数不同，不作因果比较；右每档16条。区间仅作描述。")
    return names


def write_tables(summary, path):
    def fmt(value, digits=3):
        return "—" if value is None else f"{value:.{digits}f}"
    lines=["# 后续实验数据表", "", "下表由 analyze.py 从保存记录生成；空缺表示未评分或未保存，不代表成功。", "",
           "| 实验 | 条件 | 记录数 | API/执行错误 | 任务数 | SAY 字符均值（case 等权） |", "| :--- | :--- | ---: | ---: | ---: | ---: |"]
    for v in summary["groups"].values():
        lines.append(f'| {v["experiment"]} | {v["condition"]} | {v["n"]} | {v["status"].get("error",0)} | {v["cases"]} | {fmt(v["say_chars"]["mean"],1)} |')
    lines += ["", "| 实验/条件 | 指标 | 成功/已评分 | 缺失 | 比例 | Wilson 95% 区间 |", "| :--- | :--- | :--- | ---: | ---: | :--- |"]
    for key,v in summary["groups"].items():
        for metric,info in v["evaluation"].items():
            if info["n"]:
                lo,hi=info["wilson95"]
                lines.append(f'| {key} | {metric} | {info["successes"]}/{info["n"]} | {info["missing"]} | {fmt(info["rate"])} | [{fmt(lo)}, {fmt(hi)}] |')
    lines += ["", "| 事实条件 | SAY 结论标记覆盖/全部记录 | SAY 缺标记数（含执行失败） | direct 可评分/全部记录 | direct 可评分子集修正严格匹配率 |", "| :--- | :--- | ---: | :--- | :--- |"]
    for v in summary["groups"].values():
        if v["experiment"]!="facts":continue
        say=v["say_answer_marker_coverage"];direct=v.get("direct_parse_coverage")
        info=v["evaluation"]["direct_correct_decimal_fixed"]
        lines.append(f'| {v["condition"]} | {say["successes"]}/{say["n"]} | {say["n"]-say["successes"]} | '
                     +(f'{direct["successes"]}/{direct["n"]}' if direct else "不适用")
                     +f' | {info["successes"]}/{info["n"]} ({fmt(info["rate"])}) |')
    lines += ["", "direct 可评分指结论标记规则能够返回真/假；缺标记不能视为完整判断错误。SAY 主任务正确率将缺格式计失败，和完整语义正确性仍有区别。", "",
              "| 实验/条件 | 本地协议有效/有标记轮次 | 流式一致/有标记轮次 | 原始调用引用数 |", "| :--- | :--- | :--- | ---: |"]
    for key,v in summary["groups"].items():
        valid=v["protocol_valid"];stream=v["stream_match"]
        lines.append(f'| {key} | {valid["successes"]}/{valid["n"]} | {stream["successes"]}/{stream["n"]} | {v["exported_calls"]} |')
    lines += ["", "| 条件（相对 structured_base） | SAY 修正严格匹配率差 | 配对任务数 | case-cluster bootstrap 95% 区间 |", "| :--- | ---: | ---: | :--- |"]
    for condition,info in summary["facts_paired_decimal_fixed_vs_structured_base"].items():
        lo,hi=info["bootstrap95"]
        lines.append(f'| {condition} | {fmt(info["difference"])} | {info["paired_cases"]} | [{fmt(lo)}, {fmt(hi)}] |')
    lines += ["", "Wilson 区间为记录级描述，重复任务不满足完全独立假设；配对差值按任务聚合并重采样任务，不能外推为全部对话任务的置信范围。", ""]
    if "supplement" in summary:
        sup=summary["supplement"]
        lines += ["## 独立补充：实际固化记忆的新会话探针", "", "| 条件 | 指标 | 成功/全部分配重复 |", "| :--- | :--- | :--- |"]
        for condition,metrics in sup["conditions"].items():
            for name,v in metrics.items():lines.append(f'| {condition} | {name} | {v["successes"]}/{v["n"]} |')
        lines += ["", "| 重复 | 判断条数 | 判断保留两数字 | 证据编号有效 | traits 未改变 | 判断重建一致 | traits 重建一致 |", "| ---: | ---: | :--- | :--- | :--- | :--- | :--- |"]
        for v in sup["diagnostics"]:
            keys=["judgment_target_pair_occurrence","evidence_ids_valid_recomputed","traits_unchanged_recomputed","replay_judgments_equal_recomputed","replay_traits_equal_recomputed"]
            lines.append(f'| {v["replicate"]} | {v["judgment_count"]} | '+" | ".join(str(v[k]) for k in keys)+" |")
        lines += ["", "全文双数字出现是事后词面核对，不能替代语义正确性。该补充检验同次运行中的跨会话使用，没有跨日保持证据。", ""]
    if "annotations" in summary:
        lines += ["## 事后条件遮蔽的单代理内容标注", "", "| 条件 | 所问结论内容标签 correct | 相同子集修正严格匹配 | 全组修正严格匹配 |", "| :--- | :--- | :--- | :--- |"]
        for condition,v in summary["annotations"]["facts"].items():
            full=summary["groups"]["facts/"+condition]["evaluation"]["say_correct_decimal_fixed"]
            lines.append(f'| {condition} | {v["successes"]}/{v["n"]} | {v["matched_subset_strict_successes"]}/{v["n"]} | {full["successes"]}/{full["n"]} |')
        lines += ["", "| intervention 条件 | 具体建议数/已标注记录 |", "| :--- | :--- |"]
        for condition,v in sorted(summary["annotations"]["advice"].items()):lines.append(f'| {condition} | {v["successes"]}/{v["n"]} |')
        lines += ["", "这些标签来自事后单代理评定，既无独立人类真值，也无评审者一致性估计；仅作为格式与内容差异的诊断。", ""]
    path.write_text("\n".join(lines),encoding="utf-8")


def self_test():
    a,b=wilson(1,1)
    assert abs(a-.20654931437723745)<1e-10 and abs(b-1)<1e-10
    assert wilson(0,0)==[None,None]
    rec=lambda case,value: {"case_id":case,"status":"ok","evaluation":{"say_correct":value}}
    reference=[rec("a",False),rec("a",False),rec("b",True)]
    candidate=[rec("a",True),rec("a",True),rec("b",True)]
    paired=paired_case_difference(reference,candidate,"say_correct")
    assert paired["difference"]==.5 and paired["paired_cases"]==2
    assert evaluation({"evaluation":{"expected_state":"paused","actual_state":"healed"}},"expected_state") is False
    assert rate([{"status":"error","evaluation":{}}],"say_correct",True)["rate"]==0
    assert trait_setting({"condition":"brevity_0.9"})==("brevity",.9)
    decimal_case={"kind":"number","gold":["37.5"]}
    assert corrected_answer("结论：37.5%",decimal_case) is True
    assert corrected_answer("结论：375%",decimal_case) is False
    assert corrected_answer("结论：37.5。",decimal_case) is True
    assert corrected_answer("与问题无关的文字",decimal_case) is None
    assert corrected_answer("结论：<37.5>%",decimal_case) is False
    assert corrected_answer("结论：37.5%.",decimal_case) is True
    sample={"records":[{"experiment":"facts","condition":"structured_base","case_id":"a",
                         "trial_id":"one","status":"ok","evaluation":{"say_correct":False,
                         "direct_correct":None,"answer_format_present":False},"turns":[],"call_ids":["one/c1"]}]}
    info=summarize(sample)["groups"]["facts/structured_base"]
    assert info["direct_parse_coverage"]["rate"]==0
    assert info["evaluation"]["direct_correct"]["rate"] is None
    assert info["exported_calls"]==1 and info["say_answer_marker_coverage"]["rate"]==0
    print("Analysis self-test passed.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=BASE/"experiment_results.json")
    parser.add_argument("--out",type=Path,default=BASE)
    parser.add_argument("--supplement",type=Path,default=BASE/"supplement_results.json")
    parser.add_argument("--annotations",type=Path,default=BASE/"annotation_summary.json")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    payload=args.input.read_bytes();data=json.loads(payload)
    amendment=add_corrected_scores(data)
    summary=summarize(data)
    if args.supplement.exists():
        supplement_payload=args.supplement.read_bytes()
        summary["supplement"]=summarize_supplement(json.loads(supplement_payload))
        summary["supplement"]["input_sha256"]=hashlib.sha256(supplement_payload).hexdigest()
    if args.annotations.exists():
        annotation_payload=args.annotations.read_bytes()
        summary["annotations"]=summarize_annotations(json.loads(annotation_payload),data.get("records",[]))
        summary["annotations"]["input_sha256"]=hashlib.sha256(annotation_payload).hexdigest()
    args.out.mkdir(parents=True,exist_ok=True)
    amendment["input_sha256"]=hashlib.sha256(payload).hexdigest()
    (args.out/"scoring_amendment.json").write_text(json.dumps(amendment,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary["analysis"]["scoring_amendment"]={k:v for k,v in amendment.items() if k!="affected_trials"}
    summary["analysis"]["input_sha256"]=hashlib.sha256(payload).hexdigest()
    summary["analysis"]["analyzer_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["figures"]=make_figures(data,summary,args.out)
    (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_tables(summary,args.out/"results_tables.md")
    print(json.dumps({"records":summary["record_count"],"groups":len(summary["groups"]),"figures":summary["figures"]},ensure_ascii=False))


if __name__=="__main__":
    main()
