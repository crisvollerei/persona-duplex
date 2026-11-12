"""Offline observer adapter; never sends a reply, queries relationship memory, or updates it."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("prior_observer",HERE.parent/"arrival_shadow"/"observer.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
BASELINE=old.BASE+"\n"+old.FORMS["evidence"]
TYPED="""你是会话旁路观察员。只读一段独立对话，以最后一轮为准，为每个指定人物分别记录两件事：说话的人传达了怎样的位置状态，以及这个判断是怎么来的。目标是对话传达的当前信息，不需要外部核实。
state：present表示已在目标岗位，future表示尚需到达岗位（不要求已经在路上），unknown表示无法落实到此人此刻的状态。
basis针对这个人的当前位置或到岗需求：explicit=说话者直接交代了地点、动向或在岗位发生的行动，含有明确来源的转述；pragmatic=没有直接定位，但本次语境足以让听者推断状态；none=没有可用的定位依据。仅明确说过某种意愿，不等于明确报告位置。否定一项报告不等于肯定相反的位置。不要一律把意愿写成unknown，也不要一律推断为future。不要把一个人的线索分给另一个人；留意所指地点、时间、后来更正。
只返回一个JSON对象，people数组按请求的姓名顺序各写一条；每条含person、evidence、basis、state。evidence为0至3个对象，各含turn（1起算的整数）和quote（此轮text的逐字连续非空片段，不含说话人前缀）。确定状态或依据非none时必须指出引文，若需上下文可引多轮。不要输出对话回复。"""
def messages(case,people,loc,method,order):
    user=dict(location=loc,people=list(people if order!="B" else people[::-1]),
              dialogue=[dict(turn=n,**r) for n,r in enumerate(case["turns"],1)])
    return [dict(role="system",content=BASELINE if method=="legacy" else TYPED),
            dict(role="user",content=json.dumps(user,ensure_ascii=False))]
def classify(raw,case,people,method):
    out={p:dict(state=None,basis=None,evidence=[],reason="invalid_schema") for p in people}
    def unique(items):
        if len(dict(items))!=len(items):raise ValueError("duplicate_field")
        return dict(items)
    try:
        entries=json.loads(raw,object_pairs_hook=unique)["people"]
        if not isinstance(entries,list) or len(entries)!=len(people) or {x["person"] for x in entries}!=set(people):return out
    except (TypeError,ValueError,KeyError):return out
    for x in entries:
        p=x["person"];state=x.get("state");basis=x.get("basis") if method=="typed" else None;ev=x.get("evidence")
        if state not in ("present","future","unknown") or (method=="typed" and basis not in ("explicit","pragmatic","none")):continue
        if not isinstance(ev,list) or len(ev)>3 or (not ev and (state!="unknown" or basis in ("explicit","pragmatic"))):
            out[p]["reason"]="missing_evidence";continue
        if not all(isinstance(z,dict) and type(z.get("turn")) is int and 1<=z["turn"]<=len(case["turns"])
                   and isinstance(z.get("quote"),str) and z["quote"].strip() and z["quote"] in case["turns"][z["turn"]-1]["text"] for z in ev):
            out[p]["reason"]="bad_span";continue
        out[p]=dict(state=state,basis=basis,evidence=ev,reason="accepted")
    return out
def selfcheck():
    pp=["甲","乙"];c={"turns":[dict(speaker="甲",text="楼下还有两个箱子。")]}
    x=dict(person="甲",state="future",basis="pragmatic",evidence=[{"turn":1,"quote":"楼下"}])
    y=dict(person="乙",state="unknown",basis="none",evidence=[])
    def test(row):return classify(json.dumps(dict(people=[row,y]),ensure_ascii=False),c,pp,"typed")["甲"]
    assert test(x)["basis"]=="pragmatic" and test(dict(x,evidence=[]))["state"] is None
    assert test(dict(x,evidence=[{"turn":1,"quote":"我下楼了"}]))["reason"]=="bad_span"
    assert test(dict(x,basis="inferred_fact"))["state"] is None and classify("",c,pp,"legacy")["甲"]["state"] is None
