"""Exploratory stability of the unchanged ticket prompt across two study batches."""
from collections import Counter
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
def comparison():
    load=lambda p:json.loads(p.read_bytes())
    old=load(HERE.parent/"relief_revision/observed/choices.json")
    new=load(HERE/"observed/choices.json")
    cells=Counter();by_actor={}
    for k,r in new.items():
        a,h,m,q,f,n=k.split(".")
        if f!="T":continue
        past=old[".".join((a,h,m,q,n))]
        assert past["request"]==r["request"]
        p,c=past["event"]["chosen"],r["event"]["chosen"]
        cells["identical_prompt"]+=1
        if None not in (p,c):
            cells["both_delivered"]+=1;cells["different_person"]+=int(p!=c)
        if h=="rise" and m=="R" and q=="relief":
            g=by_actor.setdefault(a,Counter())
            for word,z in (("old",p),("new",c)):
                g[word+"_delivered"]+=int(z is not None)
                g[word+"_second"]+=int(z is not None and z==r["menu"][1]["value"])
    return {"all":dict(cells),"raw_improvement":{a:dict(z) for a,z in by_actor.items()}}
if __name__=="__main__":print(json.dumps(comparison(),ensure_ascii=False,indent=2))
