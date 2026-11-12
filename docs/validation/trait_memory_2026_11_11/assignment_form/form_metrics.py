"""Layout-paired measurements. Missing choices never mean 'reserve'."""
import json
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
FORMS=("T","B","N")

def summarize(rows):
    def y(a,h,m,q,f,n,strict=False):
        r=rows[".".join(map(str,(a,h,m,q,f,n)))]
        if r["event"]["chosen"] is None or (strict and not r["validity"]["valid"]):return None
        return int(r["event"]["chosen"]==r["goal"])
    def diff(pairs): # first condition minus second one
        ok=[(u,v) for u,v in pairs if None not in (u,v)]
        l=sum(u for u,v in ok);r=sum(v for u,v in ok);n=len(ok)
        lo=sum((u or 0)-(1 if v is None else v) for u,v in pairs)
        hi=sum((1 if u is None else u)-(v or 0) for u,v in pairs)
        return dict(left=l,right=r,n=n,d=(l-r)/n if n else None,bounds=[lo/len(pairs),hi/len(pairs)])
    def swaps(a,f,common=False,strict=False,repeat=False):
        pairs=[];second=choices=0;known=0;flips_all=0
        for h in ("rise","fall"):
            for indices in (((0,2),(1,3)) if repeat else ((0,1),(2,3))):
                u,v=[y(a,h,"R","relief",f,n,strict) for n in indices]
                if None not in (u,v):
                    known+=1;flips_all+=int(u!=v)
                for n,z in zip(indices,(u,v)):
                    if z is not None:
                        choices+=1;second+=int(z==n%2)
                if common and any(y(a,h,"R","relief",ff,n,strict) is None for n in indices for ff in ("T","N")):continue
                if None not in (u,v):pairs.append((u,v))
        flips=sum(u!=v for u,v in pairs);n=len(pairs)
        return dict(n=n,flips=flips,F=flips/n if n else None,
                    Bpos=None if repeat or not n else sum(v-u for u,v in pairs)/n,
                    second=second,choices=choices,all_planned_flip_bounds=[flips_all/4,(flips_all+4-known)/4])
    actors={}
    for a in ("tang","yu"):
        native={f:swaps(a,f) for f in FORMS}
        common={f:swaps(a,f,True) for f in ("T","N")}
        common_strict={f:swaps(a,f,True,True) for f in ("T","N")}
        repeat=swaps(a,"N",repeat=True)
        c,n=common["T"],common["N"]
        replace=n["n"]>=3 and c["F"]-n["F"]>=.25 and abs(n["Bpos"])<=.25 and repeat["n"]>=3 and repeat["F"]<=.25
        table={}
        for m in ("R","V"):
            for f in FORMS:
                line={}; arm=m+"."+f
                for q,other,label in (("relief","fall","order"),("relief","flat","recovery"),("other_person","fall","identity"),("onsite","fall","domain")):
                    for strict in (False,True):
                        pairs=[(y(a,"rise",m,q,f,i,strict),y(a,other,m,q,f,i,strict)) for i in range(4)]
                        line[label+("_valid" if strict else "")]=diff(pairs)
                line["scope_gate"]=all(line[g]["n"]>=3 for g in ("order","recovery","identity","domain")) and (
                    min(line[g]["d"] for g in ("order","recovery"))>=.5 and max(abs(line[g]["d"]) for g in ("identity","domain"))<=.25)
                table[arm]=line
        actors[a]=dict(native=native,common=common,common_valid=common_strict,same_order=repeat,replace=replace,memory=table)
    delivery={}
    for f in FORMS:
        rr=[r for k,r in rows.items() if k.split(".")[-2]==f]
        delivery[f]=dict(n=len(rr),signed=sum(r["event"]["chosen"] is not None for r in rr),valid=sum(r["validity"]["valid"] for r in rr))
    return dict(planned=len(rows),delivery=delivery,states=dict(Counter(r["event"]["status"] for r in rows.values())),
                actors=actors,replace=all(x["replace"] for x in actors.values()))
if __name__=="__main__":
    out=HERE/"observed";r=summarize(json.loads((out/"choices.json").read_bytes()))
    assert r==json.loads((out/"results.json").read_bytes())
    print(r["delivery"],r["replace"])
    for a,z in r["actors"].items():print(a,z["common"],z["same_order"])
