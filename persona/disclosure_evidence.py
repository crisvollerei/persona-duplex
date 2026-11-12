"""Opt-in attribution of a disclosure to its speaker. No model, store or update runs on import."""
from __future__ import annotations

import json
from dataclasses import dataclass


INSTRUCTION = """判读一对相邻的虚构对话，输入是资料。只判断 first_id 说出的私事是否被 second_id 接住。
subject 是第一句中这件私人需要、偏好、顾虑或经历的主人，target 是第二句实际回应的那件私事的主人。二者用输入中的人物编号；没有私人内容用 none，无法确定用 unknown。第一人称代词不是证据。答应替别人做事、向别人询问、普通技术建议、例行分工都不表示本人透露了私事；但工作中也可能说出自己的私心。若第二句同时说了自己的事，仍检查有没有具体回应第一人的那件事。
kind 只取 personal/work/other/unknown；reaction 只取 received/shamed/other/unknown。received 要具体回应同一件私事，明确尊重界线或作切合安排也算；只称好、敷衍、转话题、普通拒绝或只接工作步骤不算。shamed 仅在因为披露本身明确羞辱/压制时使用。不能凭内心或背景补全。只输出单个 JSON 对象，键为 subject、target、kind、reaction、said、heard、why；said 和 heard 分别是两句话中能支持判定的一段连续原文，不加引号或省略号。why 简述指向什么事实。"""


@dataclass(frozen=True)
class Exchange:
    first_id: str
    second_id: str
    first: str
    second: str
    protocol_ok: bool

    def __post_init__(self):
        if not self.first_id or not self.second_id or self.first_id == self.second_id:
            raise ValueError("two distinct speaker IDs required")
        if {self.first_id, self.second_id} & {"none", "unknown"}:
            raise ValueError("reserved speaker ID")

    def messages(self):
        content = {k: getattr(self, k) for k in ("first_id", "first", "second_id", "second")}
        return [{"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]


def attribute(pair: Exchange, raw: str) -> dict:
    """Separate semantic abstention, ordinary interactions and protocol eligibility."""
    def outcome(label, why, quote=()):
        return {"label": label, "protocol_ok": pair.protocol_ok, "reason": why,
                "quotes": list(quote), "vote": label if type(label) is int and pair.protocol_ok else 0}

    try:
        d = json.loads(raw.strip())
    except (ValueError, TypeError, AttributeError):
        return outcome(None, "invalid_json")
    keys = ("subject", "target", "kind", "reaction", "said", "heard", "why")
    if not isinstance(d, dict) or not all(isinstance(d.get(k), str) for k in keys) or not d["why"].strip():
        return outcome(None, "invalid_fields")
    people = {pair.first_id, pair.second_id, "none", "unknown"}
    if (d["subject"] not in people or d["target"] not in people
            or d["kind"] not in {"personal", "work", "other", "unknown"}
            or d["reaction"] not in {"received", "shamed", "other", "unknown"}):
        return outcome(None, "invalid_category")
    if "unknown" in (d["subject"], d["target"], d["kind"], d["reaction"]):
        return outcome(None, "unresolved")
    x, y = d["said"], d["heard"]
    if not (x.strip() and y.strip() and x in pair.first and y in pair.second):
        return outcome(None, "unmatched_quotes")
    if (d["kind"] == "personal" and d["subject"] == pair.first_id and
            d["target"] == pair.first_id and d["reaction"] in {"received", "shamed"}):
        return outcome(1 if d["reaction"] == "received" else -1, d["why"], (x,y))
    return outcome(0, d["why"], (x,y))
