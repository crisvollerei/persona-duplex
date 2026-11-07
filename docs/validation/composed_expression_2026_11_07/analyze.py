"""Recompute material checks, descriptive results and figures without model calls."""
from dataclasses import asdict
import argparse
import gzip
import json
from pathlib import Path
import statistics

from benchmark import HERE, implementation, load, save, sha


def collect():
    m = implementation()
    checks = []
    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)
    data = load(HERE / "results.json"); records = data["records"]
    key = load(HERE / "review_key.json"); inputs = load(HERE / "review_inputs.json")
    ledger = json.loads(gzip.decompress((HERE / "runtime_ledgers.json.gz").read_bytes()))
    check("protocol digest", sha(HERE / "protocol.json") == data["protocol_sha256"])
    check("18 conditions", len(records) == 18 and len({r["id"] for r in records}) == 18)
    check("108 primary outputs", sum(len(r["turns"]) for r in records) == data["primary_outputs"] == 108)
    check("108 verification outputs", data["verification_outputs"] == 108 and all(r["restart_equal"] for r in records))
    by_id = {r["id"]:r for r in records}
    check("complete anonymous mapping", set(key.values()) == set(by_id) and {i["id"] for i in inputs} == set(key))
    for item in inputs:
        check(item["id"] + "/visible outputs", item["turns"] == [{k:t[k] for k in ("turn", "user", "sources", "say")} for t in by_id[key[item["id"]]]["turns"]])
    paired = {}; summaries = {}
    for row in records:
        id = row["id"]; history = []
        signatures = [t["signature"] for t in row["turns"]]
        events = ledger[id]
        check(id + "/two sessions", sum(e["kind"] == "policy_run_start" for e in events) == 2)
        check(id + "/no model events", not any("model_call" in e["kind"] for e in events))
        successes = [e for e in events if e["kind"] == "policy_turn"]
        check(id + "/six delivered turns", len(successes) == 6)
        for turn, success in zip(row["turns"], successes):
            prefix = id + "/" + str(turn["turn"])
            check(prefix + "/delivery", turn["delivery_match"] and success["say"] == turn["say"])
            check(prefix + "/zero calls", turn["calls_count"] == success["calls_count"] == 0)
            frame = m["policy"].ContentFrame.from_json(json.dumps(turn["frame"]))
            frame.validate_evidence(turn["user"], tuple(m["policy"].Source(**s) for s in turn["sources"]))
            state = m["policy"].PolicyState.from_dict(turn["state"])
            decision = m["policy"].decide(frame, state,
                m["policy"].PolicyConfig(intervention=0, practical=0, feeling=1, detail=1, seed="composed-study-v1"),
                topic=next(c["topic"] for c in load(HERE / "cases.json") if c["id"] == row["case_id"]),
                key=str(turn["turn"]), at="2026-11-07T21:00:00+08:00")
            check(prefix + "/recomputed decision", json.loads(json.dumps(asdict(decision))) == turn["decision"])
            commitments = state.for_topic(next(c["topic"] for c in load(HERE / "cases.json") if c["id"] == row["case_id"]))
            if row["method"] == "composed":
                plan = m["expression"].compose_reply(frame, decision, commitments,
                    config=m["expression"].ExpressionConfig(**row["expression"]), seed="composed-study-v1",
                    key=str(turn["turn"]), recent=history)
                check(prefix + "/recomputed plan", json.loads(json.dumps(asdict(plan))) == turn["expression_plan"])
                check(prefix + "/recomputed text", plan.text == turn["say"])
                check(prefix + "/recent template", plan.template_id not in history[-2:])
                history.append(plan.template_id)
                plan_event = next(e for e in events if e["kind"] == "policy_expression_plan" and e["turn_id"] == success["turn_id"])
                check(prefix + "/plan before completion", events.index(plan_event) < events.index(success))
            else:
                check(prefix + "/legacy text", m["policy_session"].bounded_reply(frame, decision, commitments) == turn["say"])
        causal = [{k:t[k] for k in ("user", "sources", "frame", "decision", "state")} for t in row["turns"]]
        pair = (row["case_id"], row["profile"])
        if pair in paired: check(id + "/matched condition", causal == paired[pair])
        paired[pair] = causal
        summaries[id] = dict(adjacent_same_signature=sum(a == b for a,b in zip(signatures, signatures[1:])),
                             transitions=5, mean_chars=statistics.mean(len(t["say"]) for t in row["turns"]))
    summary = dict(primary_outputs=108, mechanical_replays=108, api_calls=0,
        unique_dialogues=len({tuple(t["say"] for t in r["turns"]) for r in records}),
        methods={}, coders={}, agreement={}, per_condition=summaries)
    for method in ("bounded", "composed"):
        rows = [r for r in records if r["method"] == method]
        summary["methods"][method] = dict(conditions=len(rows), outputs=6*len(rows),
            unique_dialogues=len({tuple(t["say"] for t in r["turns"]) for r in rows}),
            adjacent_same_signature=sum(summaries[r["id"]]["adjacent_same_signature"] for r in rows),
            transitions=5*len(rows), mean_chars=statistics.mean(len(t["say"]) for r in rows for t in r["turns"]))
    labels = []
    metrics = ("naturalness", "repetition", "warmth", "formality", "brevity")
    for coder in (1, 2):
        ratings = load(HERE / f"ratings_{coder}.json")
        check(f"coder {coder} coverage", len(ratings) == 18 and {r["id"] for r in ratings} == set(key))
        mapping = {key[r["id"]]:r for r in ratings}; labels.append(mapping)
        for row in ratings:
            check(f"coder{coder}/{row['id']}/turns", len(row["turns"]) == 6 and {t["turn"] for t in row["turns"]} == set(range(1,7)))
            check(f"coder{coder}/{row['id']}/scores", all(type(row[k]) is int and 1 <= row[k] <= 5 for k in metrics))
            check(f"coder{coder}/{row['id']}/types", all(t["advice"] in ("yes", "no", "unclear") and type(t["unsupported"]) is bool for t in row["turns"]))
        groups = {}
        for method in ("bounded", "composed"):
            for profile in ("all", "reserved", "warm", "formal"):
                group = [mapping[r["id"]] for r in records if r["method"] == method and (profile == "all" or r["profile"] == profile)]
                groups[method + "/" + profile] = dict(dialogues=len(group), **{k:statistics.mean(r[k] for r in group) for k in metrics},
                    advice=sum(t["advice"] == "yes" for r in group for t in r["turns"]),
                    advice_unclear=sum(t["advice"] == "unclear" for r in group for t in r["turns"]),
                    unsupported=sum(t["unsupported"] for r in group for t in r["turns"]))
        summary["coders"][str(coder)] = groups
    for metric in metrics:
        summary["agreement"][metric] = dict(equal=sum(labels[0][id][metric] == labels[1][id][metric] for id in by_id), total=18)
    for metric in ("advice", "unsupported"):
        summary["agreement"][metric] = dict(equal=sum(a[metric] == b[metric] for id in by_id for a,b in zip(
            sorted(labels[0][id]["turns"], key=lambda t:t["turn"]), sorted(labels[1][id]["turns"], key=lambda t:t["turn"]))), total=108)
    return summary, dict(checks=len(checks), failed=0, details=checks)


def figures(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "figure.dpi":150, "savefig.dpi":150, "font.family":"DejaVu Sans"})
    colors = {"bounded":"#8a96a8", "composed":"#256e9b"}
    fig, axes = plt.subplots(1,2, figsize=(10,4.5))
    for ax, metric, title in zip(axes, ("naturalness", "repetition"), ("Naturalness (higher is better)", "Perceived repetition (lower is better)")):
        for coder, marker in (("1","o"),("2","s")):
            vals = [summary["coders"][coder][m + "/all"][metric] for m in ("bounded","composed")]
            ax.plot([0,1], vals, marker=marker, linewidth=1.5,
                    label=f"Coder {coder}: {vals[0]:.2f} to {vals[1]:.2f}")
        ax.set(xticks=[0,1],xticklabels=["Fixed", "Composed"],ylim=(.8,5.4),xlim=(-.25,1.45),title=title,ylabel="Dialogue rating (1-5)")
        ax.grid(axis="y", alpha=.15); ax.legend(loc="lower left")
    fig.suptitle("Multi-turn coding: 18 conditions, 12 distinct dialogues; two Codex tasks",fontsize=11)
    fig.tight_layout(); fig.savefig(out/"fig1_experience.png"); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,4.3))
    for ax, metric in zip(axes,("warmth","formality","brevity")):
        for method in ("bounded","composed"):
            for coder, marker in (("1","o"),("2","s")):
                vals=[summary["coders"][coder][method+"/"+p][metric] for p in ("reserved","warm","formal")]
                ax.plot(range(3), vals, color=colors[method],marker=marker,linestyle="-" if coder=="1" else "--",
                        label=("Fixed" if method=="bounded" else "Composed")+" / "+coder)
        ax.set(xticks=range(3),xticklabels=["Reserved","Warm","Formal"],ylim=(.8,5.4),title=metric.capitalize())
        ax.grid(axis="y",alpha=.15)
    axes[0].set_ylabel("Dialogue rating (1-5)"); axes[2].legend(loc="lower right",fontsize=8)
    fig.suptitle("Profile bundles; means over 3 synthetic scenarios, no causal single-trait claim",fontsize=11)
    fig.tight_layout();fig.savefig(out/"fig2_style.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    methods=("bounded","composed");values=[summary["methods"][m]["adjacent_same_signature"] for m in methods]
    ax.bar([0,1],[v/45 for v in values],color=[colors[m] for m in methods],width=.5)
    for x,n in enumerate(values):ax.text(x,n/45+.025,f"{n}/45",ha="center")
    ax.set(xticks=[0,1],xticklabels=["Fixed","Composed"],ylim=(0,1),ylabel="Adjacent authored-surface repetition",title="Mechanical variation is not semantic variation")
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis="y",alpha=.15)
    fig.text(.5,.02,"45 transitions per method; fixed baseline is only 3 distinct sequences (9/15 unique transitions).",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/"fig3_surface_repetition.png");plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    summary,audit=collect();args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/"summary.json",summary);save(args.out/"audit_report.json",audit)
    folder=args.out/"figures";folder.mkdir(exist_ok=True);figures(summary,folder)
    print(json.dumps(dict(checks=audit["checks"],failed=audit["failed"],outputs=summary["primary_outputs"],api_calls=0),ensure_ascii=False))


if __name__=="__main__":main()
