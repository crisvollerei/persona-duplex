"""Read original provider forms and responses independently of the session and action parser."""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
from hashlib import sha256
HERE=Path(__file__).resolve().parent
O=HERE/"observed"

def verify():
    load=lambda f:json.loads((O/f).read_bytes())
    reg,mem,rows,res=map(load,("registration.json","memory.json","choices.json","results.json"))
    with gzip.open(O/"calls.jsonl.gz","rb") as f:calls=[json.loads(z) for z in f]
    assert set(reg["order"])=={c["id"] for c in calls}==set(rows) and len(calls)==296
    ids=set(); models=Counter(); usage=Counter(); actions={}; status=Counter(); valid=Counter()
    for c in calls:
        k=c["id"];a,h,m,q,i=k.split(".")
        ev=mem[".".join((a,h,m))]
        msgs=c["messages"]
        assert c["params"]==reg["model"] and rows[k]["request"]==sha256(json.dumps(msgs,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        assert json.dumps(ev["visible"],ensure_ascii=False)[1:-1] in msgs[0]["content"]
        if c["ok"]:
            receipt=c["response"];assert receipt["id"] and receipt["id"] not in ids
            ids.add(receipt["id"]);models[receipt["model"]]+=1
            usage.update({kk:v for kk,v in receipt["usage"].items() if type(v) is int})
        # Decode role-public SAY, then the actual printed menu, not HEAD or prose recommendations.
        text=re.split(r"(?m)^\s*<<<SAY>>>\s*$",c["raw"])
        say=re.split(r"(?m)^\s*<<<TAIL>>>\s*$",text[1])[0].strip() if len(text)==2 else ""
        tickets=re.findall(r"(?m)^落签\s*[：:]\s*(青签|白签)[。.]*\s*$",say)
        terminal=re.search(r"(?:^|\n)\s*落签\s*[：:]\s*(青签|白签)[。.]?\s*$",say)
        hand=dict(re.findall(r"(?m)^(青签|白签)｜由(唐荔|俞川|池岚|常鸣)",msgs[-1]["content"]))
        named=("池岚" if q=="other_person" else {"tang":"俞川","yu":"唐荔"}[a])
        assert set(hand.values())=={named,"常鸣"} and hand[list(hand)[int(i)%2]]==named
        assert hand["青签" if int(i)<2 else "白签"]==named
        raw_chosen=hand[terminal[1]] if terminal and len(tickets)==1 else None
        chosen=rows[k]["event"]["chosen"]
        if chosen is not None:assert raw_chosen==chosen
        if chosen!=raw_chosen:status["raw_ticket_withheld"]+=1
        assert list(rows[k]["event"]["after"].values())==[chosen or "未定"]
        actions[a,h,m,q,int(i)]=None if chosen is None else int(chosen==named)
        status["executed"]+=int(chosen is not None)
        valid["by_arm_"+m]+=int(rows[k]["valid"])
    assert status["executed"]==res["executed"] and sum(valid.values())==res["protocol_valid"]
    for a,A in res["actors"].items():
        for q in ("relief","onsite","other_person"):
            for m in ("R","V","O","S"):
                for key,branch in (("order","fall"),("recovery","flat")):
                    pairs=[(actions[a,"rise",m,q,n],actions[a,branch,m,q,n]) for n in range(4)]
                    pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
                    z=A[q][m][key]
                    assert (len(pairs),sum(x for x,y in pairs),sum(y for x,y in pairs))==(z["n"],z["left"],z["right"])
    return dict(count=len(calls),unique_receipts=len(ids),returned_models=dict(models),actions=dict(status),usage=dict(usage),valid=dict(valid))

if __name__=="__main__":print(json.dumps(verify(),ensure_ascii=False,indent=2))
