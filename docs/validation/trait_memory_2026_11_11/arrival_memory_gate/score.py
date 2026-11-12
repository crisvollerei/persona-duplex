"""Descriptive paired order effects; in Z the contrast is between identical-input replicates."""
from collections import Counter
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=json.loads((HERE/"design.json").read_bytes())

def summarize(rows):
    def value(a,j,s,h,m,n,strict):
        r=rows[".".join(map(str,(a,j,s,h,m,n)))]
        if r["event"]["chosen"] is None or strict and not r["validity"]["valid"]:return None
        return int(r["event"]["chosen"]==r["goal"])
    def measure(a,jobs,s,m,common=False,strict=False):
        pairs=[];lo=hi=0;total=4*len(jobs)
        for j in jobs:
            for n in range(4):
                u,v=[value(a,j,s,h,m,n,strict) for h in ("rise","fall")]
                lo+=(u or 0)-(1 if v is None else v);hi+=(1 if u is None else u)-(v or 0)
                if common and any(value(a,j,s,h,f,n,strict) is None for h in ("rise","fall") for f in ("V","Z")):continue
                if u is not None and v is not None:pairs.append((u,v))
        n=len(pairs);l=sum(u for u,v in pairs);r=sum(v for u,v in pairs)
        return dict(n=n,planned=total,left=l,right=r,d=(l-r)/n if n else None,
                    discordant=sum(u!=v for u,v in pairs),full_bounds=[lo/total,hi/total])
    actors={}
    for a in D["actors"]:
        cells={}
        for j,jobs in [("pooled",list(D["jobs"])),*((j,[j]) for j in D["jobs"])]:
            cells[j]={}
            for s in D["states"]:
                cells[j][s]={m:measure(a,jobs,s,m) for m in D["arms"]}
                for m in ("V","Z"):
                    cells[j][s][m+"_common"]=measure(a,jobs,s,m,True)
                    cells[j][s][m+"_valid_common"]=measure(a,jobs,s,m,True,True)
        def criteria(suffix):
            coverage=all(cells[j][s]["V"+suffix]["n"]>=3 for j in D["jobs"] for s in ("future","present"))
            future=cells["pooled"]["future"];present=cells["pooled"]["present"]
            v,z=future["V"+suffix]["d"],future["Z"+suffix]["d"];pv,pz=present["V"+suffix]["d"],present["Z"+suffix]["d"]
            use=coverage and all(cells[j]["future"]["V"+suffix]["d"]>=.5 for j in D["jobs"]) and v-abs(z)>=.25
            avoid=coverage and abs(pz)<=.25 and abs(pv)-abs(pz)>=.25
            return dict(coverage=coverage,use=use,avoidable_spillover=avoid,passed=use and avoid)
        actors[a]=dict(cells=cells,gate=criteria("_common"),strict_gate=criteria("_valid_common"))
    return dict(n=len(rows),committed=sum(r["event"]["chosen"] is not None for r in rows.values()),
                protocol_valid=sum(r["validity"]["valid"] for r in rows.values()),
                status=dict(Counter(r["event"]["status"] for r in rows.values())),actors=actors,
                supported=all(v["gate"]["passed"] for v in actors.values()))

if __name__=="__main__":
    out=HERE/"observed";result=summarize(json.loads((out/"choices.json").read_bytes()))
    assert result==json.loads((out/"results.json").read_bytes())
    for a,z in result["actors"].items():
        print(a,z["gate"])
        for s,g in z["cells"]["pooled"].items():print(s,[(m,g[m]["d"],g[m]["n"]) for m in ("V","R","Z")])
