"""Read-only arrival observer. Output is research data, not a reply or a routing decision."""
import json

BASE="""你是会话旁路观察员。只读一段独立对话，判断最后一轮之后，指定的两个人各自相对于目标岗位此刻的状态。
present：当前明确已在目标岗位。future：明确仍需到该岗位。unknown：当前证据不足，包括只表达意愿而没有确认位置。
发言者是谁、指代哪位、到达的地点、是今天还是过去、后面是否有更正，都需要对应。不要把一个人的信息推广给另一人；可以采用当事人自述或明确的现场转述，不需要外部核实。不能靠惯例补事实。
仅返回一个JSON对象，people数组按请求的人员顺序各写一条。不要输出对话回复。"""
FORMS={
    "direct":'每条只含person（原姓名）和state（present、future、unknown之一）。',
    "evidence":'每条按person、evidence、state给出。evidence是0至3个对象的数组，每项含turn（从1开始的整数）、quote（该轮text内逐字连续的非空片段，不含发言者前缀）。确定状态必须先指出相应证据；unknown可无引文。state为present、future、unknown之一。'
}
STATES=("present","future","unknown")
PEOPLE=("俞川","常鸣")
def messages(case,form,order):
    people=PEOPLE if order=="YC" else PEOPLE[::-1]
    user={"location":"展厅内的签收台","people":people,"dialogue":[dict(turn=i,**x) for i,x in enumerate(case["turns"],1)]}
    return [{"role":"system","content":BASE+"\n"+FORMS[form]},{"role":"user","content":json.dumps(user,ensure_ascii=False)}]

def group(states):
    s=list(states)
    if len(s)!=2 or any(x not in STATES for x in s):return "invalid"
    if s[0]==s[1]!="unknown":return s[0]
    return "mixed" if "unknown" not in s else "unknown"

def classify(raw,case,form):
    invalid={p:{"state":None,"reason":"schema","evidence":[]} for p in PEOPLE}
    try:
        def unique(items):
            d=dict(items)
            if len(d)!=len(items):raise ValueError("duplicate field")
            return d
        obj=json.loads(raw,object_pairs_hook=unique)
        if not isinstance(obj,dict):return invalid
        entries=obj["people"]
        if not isinstance(entries,list) or len(entries)!=2 or {x["person"] for x in entries}!=set(PEOPLE):return invalid
    except (ValueError,KeyError,TypeError):return invalid
    for x in entries:
        p=x["person"];state=x.get("state");quotes=x.get("evidence",[])
        if state not in STATES:continue
        if form=="evidence":
            if not isinstance(x.get("evidence"),list) or len(quotes)>3 or (state!="unknown" and not quotes):
                invalid[p]["reason"]="evidence_missing";continue
            if not all(isinstance(z,dict) and type(z.get("turn")) is int and 1<=z["turn"]<=len(case["turns"])
                       and isinstance(z.get("quote"),str) and z["quote"].strip()
                       and z["quote"] in case["turns"][z["turn"]-1]["text"] for z in quotes):
                invalid[p]["reason"]="evidence_not_verbatim";continue
        invalid[p]=dict(state=state,reason="accepted",evidence=quotes if form=="evidence" else [])
    return invalid

def selfcheck():
    c={"turns":[{"speaker":"甲","text":"我没到岗。"}]}
    def raw(quote):
        return json.dumps({"people":[{"person":"俞川","state":"future","evidence":[{"turn":1,"quote":quote}]},
                                    {"person":"常鸣","state":"unknown","evidence":[]}]},ensure_ascii=False)
    assert classify(raw("没到岗"),c,"evidence")["俞川"]["state"]=="future"
    assert classify(raw("已经到了"),c,"evidence")["俞川"]["state"] is None
    assert classify("",c,"direct")["常鸣"]["state"] is None
    assert group(["future","present"])=="mixed" and group(["present","unknown"])=="unknown"
