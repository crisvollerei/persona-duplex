"""Independent checks directly on provider receipts; no network or project imports."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
O = HERE / "observed"
def load(name):
    return json.loads((O/name).read_bytes())

def verify():
    reg, choices, results = map(load, ("registration.json", "choices.json", "results.json"))
    with gzip.open(O/"calls.jsonl.gz","rb") as f:
        calls = [json.loads(line) for line in f]
    assert len(calls)==len(set(reg["order"]))==len(choices)==288
    assert set(reg["order"])==set(choices)=={c["id"] for c in calls}
    actors = {"tang":"俞川","yu":"唐荔"}
    ids, counts, outcomes, models, usage = set(), Counter(), {}, Counter(), Counter()
    faults=Counter()
    for c in calls:
        k=c["id"]; a,h,m,p,q,n=k.split("."); n=int(n); row=choices[k]
        assert c["params"]==reg["foreground"] and len(c["messages"])>=2
        blob=json.dumps(c["messages"],ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
        assert hashlib.sha256(blob).hexdigest()==row["request"]
        assert not re.search(rb"\bsk-[a-zA-Z0-9]{18,}",blob+c["raw"].encode())
        if c["ok"]:
            rid=c["response"]["id"]
            assert rid and rid not in ids
            ids.add(rid); models[c["response"]["model"]]+=1
            usage.update({kk:v for kk,v in (c["response"]["usage"] or {}).items() if type(v) is int})
        expected = actors[a] if p=="old" else "池岚"
        # Decode the commitment from the provider's public section, using the actual user-visible form.
        parts=re.split(r"(?m)^[ \t]*<<<SAY>>>[ \t]*\r?$",c["raw"])
        text=re.split(r"(?m)^[ \t]*<<<TAIL>>>[ \t]*\r?$",parts[1])[0].strip() if len(parts)==2 else ""
        form=c["messages"][-1]["content"].splitlines()
        names={}
        for line in form:
            hit=re.fullmatch(r"(青签|白签)｜由(俞川|唐荔|池岚|常鸣).+",line)
            if hit:names[hit[1]]=hit[2]
        assert set(names.values())=={expected,"常鸣"}
        assert list(names.values())[n%2]==expected and names["青签" if n//2==0 else "白签"]==expected
        event=row["event"]
        vote_lines=[l.strip() for l in text.splitlines() if re.match(r"^\s*落签\s*[：:]",l)]
        match=re.fullmatch(r"落签\s*[：:]\s*(青签|白签)[。.]?",vote_lines[0]) if len(vote_lines)==1 and text.endswith(vote_lines[0]) else None
        chosen=names[match[1]] if match else None
        # A protected delivery may discard even a literal raw vote. Keep such cases visible.
        if chosen!=event["chosen"]:
            assert chosen and event["chosen"] is None
            faults["raw_ballot_not_delivered"]+=1
        assert set(event["before"].values())=={"未定"} and set(event["after"].values())=={event["chosen"] or "未定"}
        z=None if not event["chosen"] else int(event["chosen"]==expected)
        assert z==event["target"]
        outcomes[a,h,m,p,q,n]=z
        counts["requests"]+=1; counts["executed"]+=int(z is not None)
        counts["valid"]+=row["validity"]["valid"]
    assert counts["executed"]==results["committed"] and counts["valid"]==results["protocol_valid"]
    for a in actors:
        for m in ("R","J","S"):
            for group in ("old_near","old_crop","new_near","new_crop"):
                p,family=group.split("_"); qs=["desk","audio"] if family=="near" else ["crop"]
                votes=[(outcomes[a,"kept",m,p,q,n],outcomes[a,"missed",m,p,q,n]) for q in qs for n in range(4)]
                votes=[(x,y) for x,y in votes if None not in (x,y)]
                target=results["actors"][a]["paired"][m]["groups"][group]
                assert [len(votes),sum(x for x,y in votes),sum(y for x,y in votes)]==[target[k] for k in ("n","kept","missed")]
    return {"counts":dict(counts),"unique_response_ids":len(ids),"returned_models":dict(models),
            "usage":dict(usage),"delivery_discrepancies":dict(faults)}

if __name__=="__main__":
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
