"""从同目录的公开观测复算阶段论文；不读取工作区其他实验文件，不联网。"""

import json
from collections import Counter
from pathlib import Path
from statistics import mean


def report(studies):
    def rows(name):
        return studies[name]["rows"]

    inquiry = rows("human_inquiry")
    print("表 1　追问：可评 / 计划", len(inquiry), "/", studies["human_inquiry"]["design"]["planned"])
    print("  适切性（程序、提示）：", *(mean(r[k] for r in inquiry) for k in ("program_rating", "prompt_rating")))
    print("  偏好：", dict(Counter(r["preference"] for r in inquiry)))
    edit = rows("local_edit")
    for name, pairs in [
        ("局部编辑", [r["human"] for r in edit]),
        ("能力承诺", list(studies["capability_commitments"]["design"]["human"].values())),
    ]:
        rated = [p for p in pairs if not p["skipped"]]
        print(name, "适切性（程序、提示）：",
              *(round(mean(p["scores"][method]["fit"] for p in rated), 2) for method in ("scoped", "strong")),
              "偏好：", dict(Counter(p["preference"] for p in rated)))
    materials = rows("conversation_materials")
    passed = sum(len(r["candidates"]) == 2 and r["different"] == "yes" and not r["reject"]
                 and all(c["fit"] >= 4 for c in r["candidates"])
                 and all(a["fact"] == a["boundary"] == "clear" for a in r["audit"]) for r in materials)
    print("材料筛选：", passed, "/", len(materials))
    target = [r for r in edit if r["group"] == "target"]
    for method in ("strong", "scoped"):
        def stable(r, coder):
            return [r["semantic"][method][phase][coder]["uptake"]
                    for phase in ("before", "resample", "after")] == ["plain", "plain", "light"]
        print("局部编辑", method, "两名编码者的稳定转变数：",
              *(sum(stable(r, coder) for r in target) for coder in ("jade", "amber")), "分母", len(target))

    print("\n表 2　能力承诺、归属与现态")
    for name, field, value, routes in (
        ("capability_commitments", "claim", "unsupported", ("usual", "strong", "scoped")),
        ("action_ownership", "ownership", "aligned", ("strong", "compiled")),
        ("action_state", "state", "grounded", ("ownership", "strong", "compiled")),
    ):
        for method in routes:
            group = [r for r in rows(name) if r["route"] == method]
            scored = [r for r in group if r["ratings"] is not None]
            assert all(len(r["ratings"]) == 2 and r["say"] for r in scored)
            sent = sum(r["status"] != "not_sent" for r in group)
            agree = sum(all(c[field] == value for c in r["ratings"]) for r in scored)
            print(name, method, "计划 / 实发 / 可评", len(group), sent, len(scored), f"{field}={value}（双方认可）", agree)
            if name == "action_state":
                joint = [sum(c["state"] == "grounded" and c["delivery"] == "met" and c["actor"] == "aligned"
                             for c in (r["ratings"][i] for r in scored)) for i in range(2)]
                print("  联合通过", joint, "输入 token", sum(r["usage"]["prompt_tokens"] for r in group))

    print("\n表 3　生成路径（正式单元）")
    for study in ("generation_ablation", "generation_confirmation"):
        data = rows(study)
        full = [r["usage"]["total_tokens"] for r in data if r["method"] == "full"]
        for method in ("full", "compact", "direct"):
            group = [r for r in data if r["method"] == method]
            ratio = mean(r["usage"]["total_tokens"] for r in group) / mean(full)
            conflicts = [sum(r["ratings"][i]["facts"] == "conflict" for r in group) for i in range(2)]
            print(study, method, f"非空 {sum(bool(r['say']) for r in group)}/{len(group)}",
                  f"相对用量 {ratio:.2%}", "事实冲突", conflicts)

    print("\n条件与数量：确定缺陷（两份编码）；缺失用量")
    for method in ("full", "compact", "direct"):
        for arm in ("base", "compiled"):
            group = [r for r in rows("constraint_reference") if (r["method"], r["arm"]) == (method, arm)]
            counts = [sum(c["facts"] == "conflict" or c["fulfillment"] == "failed"
                          for c in (r["ratings"][i] for r in group)) for i in range(2)]
            print(method, arm, counts, "分母", len(group), "用量缺失", sum(r["usage"] is None for r in group))


if __name__ == "__main__":
    data = json.loads(Path(__file__).with_name("observations.json").read_bytes())
    report(data["studies"])
