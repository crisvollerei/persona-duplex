"""Readable final replies with separate coder ratings; no new judgments."""
from benchmark import HERE, load

def main():
    records = load(HERE / "run/results.json")["records"]
    cases = {c["id"]: c for c in load(HERE / "cases.json")}
    profiles = {p["id"]: p for p in load(HERE / "profiles.json")}
    keys = load(HERE / "review_key.json")
    by_ref = {(ref["record_id"], ref["candidate_id"]): r["id"] for r in keys if r["kind"] == "online" for ref in r["references"]}
    ratings = [{r["id"]: r for r in load(HERE / f"reviews/ratings_{i}.json")} for i in (1,2)]
    lines = ["# 24条最终回复与分维度编码", "", "原文来自在线记录，评分来自两份独立Codex编码；斜线两侧依次为编码者1、2。此处不作额外裁决。候选池和校准项见原始材料。", ""]
    for case in cases.values():
        lines += [f"## {case['id']}", ""]
        for h in case["history"]: lines += ["> 历史：" + h["content"], ""]
        lines += ["> 用户：" + case["user"], ""]
        for row in (r for r in records if r["case"] == case["id"]):
            method = "直接生成" if row["method"] == "direct" else "候选选择"
            lines += [f"### {profiles[row['profile']]['name']} · {method}", ""]
            if not row["say"]: lines += ["输出缺失：" + row["status"], ""]; continue
            id = by_ref[(row["id"], row["selected_id"] or "direct")]
            a, b = [r[id] for r in ratings]
            lines += ["> " + row["say"].replace("\n", "\n> "), "",
                f"评审ID `{id}`；人格 {a['persona_fit']}/{b['persona_fit']}；情境 {a['context_fit']}/{b['context_fit']}；事实逻辑 {a['grounding']}/{b['grounding']}；边界 {a['boundary']}/{b['boundary']}。", "",
                "编码者1：" + a["evidence"]["reason"], "", "编码者2：" + b["evidence"]["reason"], ""]
    (HERE / "final_replies.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("Exported", len(records), "final reply conditions")

if __name__ == "__main__": main()
