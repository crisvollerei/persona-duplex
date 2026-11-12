"""Independent checks of blinded inputs, first human ballot, receipt counts and headline statistics."""
import gzip
import json
from hashlib import sha256
from pathlib import Path
import re
from collections import Counter
HERE=Path(__file__).resolve().parent
def audit():
    def load(p):return json.loads(p.read_bytes())
    O=HERE/"observed"
    people=("俞川","常鸣")
    cases={c["id"]:c["turns"] for c in load(HERE/"cases.json")}
    reg=load(O/"registration.json");res=load(O/"results.json");pred=load(O/"predictions.json");human=load(O/"human.json")
    with gzip.open(O/"calls.jsonl.gz","rb") as fp:calls=[json.loads(l) for l in fp]
    assert len(set(reg["order"]))==len(calls)==64 and set(reg["order"])==set(pred)=={r["id"] for r in calls}
    for path,h in reg["inputs"].items():assert sha256((HERE.parents[3]/path).read_bytes()).hexdigest()==h
    symbols={"到":"present","赶":"future","?":"unknown"}
    gold={i:dict(zip(people,(symbols[x],symbols[y]))) for i,x,y in re.findall(r"(\d{2}) ([到赶?])([到赶?])",human["original"])}
    assert gold==human["labels"] and len(gold)==16
    cmodel=Counter();ids=set();counts={};clips=Counter()
    for r in calls:
        k=r["id"];i,f,o=k.split(".");m=r["messages"];x=json.loads(m[1]["content"])
        assert len(m)==2 and m[0]["role"]=="system" and m[1]["role"]=="user"
        assert set(x)=={"location","people","dialogue"} and x["location"]=="展厅内的签收台"
        assert x["people"]==list(people if o=="YC" else people[::-1])
        assert x["dialogue"]==[dict(turn=t,**msg) for t,msg in enumerate(cases[i],1)]
        assert sha256(json.dumps(m,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()==pred[k]["request"]
        if r["ok"]:
            rid=r["response"]["id"];assert rid and rid not in ids;ids.add(rid)
            cmodel[r["response"]["model"]]+=1
        if r["ok"]:
            try:raw={z["person"]:z for z in json.loads(r["raw"])["people"]}
            except (TypeError,KeyError,ValueError):raw={}
        else:raw={}
        observed={}
        for p,z in pred[k]["people"].items():
            observed[p]=z["state"]
            if z["state"] is not None:
                assert raw[p]["state"]==z["state"]
                if f=="evidence":
                    assert z["evidence"]==raw[p]["evidence"]
                    for e in z["evidence"]:
                        assert e["quote"] in cases[i][e["turn"]-1]["text"];clips[(f,o)]+=1
        metric=counts.setdefault(f+"."+o,Counter())
        metric["person_correct"]+=sum(observed[p]==gold[i][p] for p in people)
        metric["both_people_correct"]+=observed==gold[i]
        metric["false_both_present"]+=all(observed[p]=="present" for p in people) and not all(gold[i][p]=="present" for p in people)
    for k,v in counts.items():
        assert all(res["methods"][k]["overall"][field]==n for field,n in v.items())
    return dict(unique_response_ids=len(ids),models=dict(cmodel),evidence_quotes={o:v for (_,o),v in clips.items()},headline={k:dict(v) for k,v in counts.items()})
if __name__=="__main__":print(json.dumps(audit(),ensure_ascii=False,indent=2))
