"""Experimental two-stage evidence reader. The first decision cannot see the reply."""
import json

from .disclosure_evidence import Exchange


FIRST = """仅判读一人已经说出口的发言，不猜接下来别人怎么回应。有没有一句话在告诉听者关于说话者自己的一件事：其具体的需要、顾虑、偏好、个人往事或边界？工作场合也可以有私心，不准因话题是工作排除个人部分。技术建议、帮别人做事、询问别人的需要、纯交接及没说具体是什么的客套都不是。混合话中只选最清楚的一件属于本人的私事。输出JSON：{"decision":"own或none或unknown","quote":"一段一字不改的连续原文","claim":"用一句话指明本人透露的具体内容"}。无此事时quote与claim留空。"""
SECOND = """第一人已经说出的私事及原话在claim中，第二人的实际回答在reply中。只判断第二人是否明确接住同一件事。尊重边界、切合私情的安排、回应对方的具体感受或经历都算；普通拒绝不算羞辱。只接技术、泛泛肯定、转话题、改说自己的类似经历而未接住这件事，均不算。嘲弄或压制对方吐露这件事本身才是shamed。不要反过来把第二人的私事算给第一人。输出JSON：{"decision":"received或shamed或other或unknown","quote":"回复中能说明该关系的一段连续原文","why":"简述理由"}。"""


def _messages(system, data):
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]


def _parse(raw):
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except (ValueError, TypeError):
        return {}


def attribute_in_order(pair: Exchange, ask):
    """ask(stage, messages) returns raw JSON. Never request the reply stage for no/unknown claim."""
    def result(label, reason, quotes=()):
        return dict(label=label, reason=reason, quotes=list(quotes), protocol_ok=pair.protocol_ok,
                    vote=label if type(label) is int and pair.protocol_ok else 0)

    first = _parse(ask("first", _messages(FIRST, {"speaker": pair.first_id, "speech": pair.first})))
    if first.get("decision") == "none":
        if first.get("quote") or first.get("claim"):
            return result(None, "inconsistent_no_claim")
        return result(0, "no_personal_claim")
    x, claim = first.get("quote"), first.get("claim")
    if (first.get("decision") != "own" or not isinstance(x, str) or not x.strip()
            or x not in pair.first or not isinstance(claim, str) or not claim.strip()):
        return result(None, "unresolved_first")

    second = _parse(ask("second", _messages(SECOND, {"claim": {"speaker": pair.first_id,
                   "quote": x, "claim": claim}, "responder": pair.second_id, "reply": pair.second})))
    code=second.get("decision");y=second.get("quote");why=second.get("why")
    if (code not in ("received", "shamed", "other") or not isinstance(y,str) or not y.strip()
            or y not in pair.second or not isinstance(why,str) or not why.strip()):
        return result(None, "unresolved_response")
    return result({"received":1,"shamed":-1,"other":0}[code],claim+"；"+why,(x,y))
