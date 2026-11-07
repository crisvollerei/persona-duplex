"""Experimental coverage briefs and literal-evidence validation, not truth checking."""
import hashlib
import random
from selection import strict_json

FEATURES = ("attention", "initiative")


def coverage_plan(case):
    controls = case["controls"]
    if set(controls) != {"no_questions", "no_advice"} or any(type(v) is not bool for v in controls.values()):
        raise ValueError("explicit application-confirmed controls required")
    if any(controls.values()) and not case["boundaries"]:
        raise ValueError("controls require recorded user boundaries")
    if not all(q in case["user"] or any(q in m["content"] for m in case["history"]) for q in case["boundaries"]):
        raise ValueError("boundary lacks source quote")
    briefs = [
        "围绕一个已知的事情细节或过程接话，不索取更多信息，接住后留白。",
        "围绕用户已表达或可明确保留不确定性的体验接话，让感受成为重心，然后留白；不把未知心理说成事实。",
        "围绕已经提到的事情细节，给一次可不回答的继续分享邀请；不改变议题、不索取隐私。",
        "围绕用户对这件事的主观体验，给一次可不回答的继续分享邀请；不是追查原因或替用户诊断。",
        "围绕事情本身带入一个具体相关的新观察或讨论角度，使互动往前一点；不能只是附和或换比喻，不要求给行动建议。",
        "对事情内容和用户体验各作一个实质回应，使两者都得到照顾，然后留白；不是加一句‘真好’就算兼顾。",
    ]
    if controls["no_questions"]:
        briefs[2] = "用户已禁问：不发出分享邀请，也不间接索取回答。可以把已知事情的一个不同细节接住后留白。"
        briefs[3] = "用户已禁问：不发出体验邀请，也不暗示需要解释感受。只回应已明确表达的体验，不清楚就保留不确定性。"
    if controls["no_advice"]:
        briefs = [s + " 禁止给行动方案或以问句包装建议。" for s in briefs]
    return [{"id": f"C{i:02}", "brief": s} for i, s in enumerate(briefs, 1)]


def blind_candidates(pool_id, candidates):
    rows = list(candidates)
    seed = int.from_bytes(hashlib.sha256(("revision/" + pool_id).encode()).digest(), "big")
    random.Random(seed).shuffle(rows)
    hidden, mapping = [], {}
    for i, candidate in enumerate(rows, 1):
        alias = f"X{i:02}"
        hidden.append({"id": alias, "text": candidate["text"]})
        mapping[alias] = candidate["id"]
    return hidden, mapping


def unavailable_annotations(candidates, reason):
    return [{"id": c["id"], "attention": None, "initiative": None,
             "validation_errors": [reason], "proposal": None} for c in candidates]


def validate_annotations(raw, hidden, mapping):
    """Reject unsupported field syntax/spans; never certify their semantic reading."""
    texts = {c["id"]: c["text"] for c in hidden}
    obj = strict_json(raw)
    if not isinstance(obj, dict) or set(obj) != {"labels"} or not isinstance(obj["labels"], list):
        raise ValueError("expected labels object")
    labels = obj["labels"]
    if len(labels) != len(texts) or any(not isinstance(r, dict) for r in labels):
        raise ValueError("label count mismatch")
    if any(not isinstance(r.get("id"), str) for r in labels) or {r["id"] for r in labels} != set(texts):
        raise ValueError("duplicate, missing or unknown id")
    result = []
    for label in labels:
        id = label["id"]; text = texts[id]
        row = {"id": mapping[id], "attention": None, "initiative": None, "validation_errors": [], "proposal": label}
        if set(label) != {"id", "attention", "focus_evidence", "initiative", "initiative_quote"}:
            row["validation_errors"].append("unexpected_fields")
            result.append(row); continue
        focus = label["attention"]
        evidence = label["focus_evidence"]
        valid_focus = isinstance(evidence, list) and all(
            isinstance(e, dict) and set(e) == {"kind", "quote"} and e["kind"] in ("content", "experience")
            and isinstance(e["quote"], str) and e["quote"].strip() and e["quote"] in text for e in evidence)
        if focus is None:
            if evidence != []: row["validation_errors"].append("unknown_focus_with_evidence")
        elif type(focus) is int and focus in (0, 1, 2) and valid_focus and 1 <= len(evidence) <= 3:
            kinds = {e["kind"] for e in evidence}
            required = {0: {"content"}, 1: {"content", "experience"}, 2: {"experience"}}[focus]
            distinct = focus != 1 or len({e["quote"] for e in evidence}) >= 2
            if kinds >= required and distinct: row["attention"] = focus
            else: row["validation_errors"].append("focus_evidence_structure")
        else: row["validation_errors"].append("invalid_focus_or_quote")
        initiative, quote = label["initiative"], label["initiative_quote"]
        if initiative is None:
            if quote is not None: row["validation_errors"].append("unknown_initiative_with_quote")
        elif type(initiative) is int and initiative in (0, 1, 2) and isinstance(quote, str) and quote.strip() and quote in text:
            if initiative != 0 or quote == text: row["initiative"] = initiative
            else: row["validation_errors"].append("absence_requires_whole_text")
        else: row["validation_errors"].append("invalid_initiative_or_quote")
        result.append(row)
    return sorted(result, key=lambda r: r["id"])


def choose(candidates, labels, profile):
    by_id = {c["id"]: c for c in candidates}
    eligible = [r for r in labels if r["id"] in by_id and all(type(r[k]) is int and r[k] in (0, 1, 2) for k in FEATURES)]
    if not eligible: return None
    selected = min(eligible, key=lambda r: (sum(abs(r[k] - profile["target"][k]) for k in FEATURES), r["id"]))
    return {"candidate_id": selected["id"], "text": by_id[selected["id"]]["text"]}
