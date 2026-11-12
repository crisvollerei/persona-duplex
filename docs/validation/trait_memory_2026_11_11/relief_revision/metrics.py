"""Paired real assignments, common-layout incremental contrasts, missing-data bounds."""
import json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent

def results(rows):
    def gap(a,m,q,left,right,arms=(),valid=False):
        def value(branch,arm,n):
            r=rows[".".join(map(str,(a,branch,arm,q,n)))]
            if r["event"]["status"]!="executed" or (valid and not r["valid"]):return None
            return int(r["event"]["chosen"]==r["goal"])
        pairs=[];lo=hi=0
        for n in range(4):
            u,v=value(left,m,n),value(right,m,n)
            lo+=(u or 0)-(1 if v is None else v);hi+=(1 if u is None else u)-(v or 0)
            if u is None or v is None or any(value(br,arm,n) is None for br in (left,right) for arm in arms):continue
            pairs.append((u,v))
        i=sum(u for u,v in pairs);j=sum(v for u,v in pairs);n=len(pairs)
        return dict(left=i,right=j,n=n,d=(i-j)/n if n else None,full_bounds=[lo/4,hi/4])
    actors={}
    for a in ("tang","yu"):
        A={}
        for q in ("relief","other_person","onsite"):
            A[q]={}
            for m in ("R","S","V","O"):
                A[q][m]={name:gap(a,m,q,"rise",other) for name,other in (("order","fall"),("recovery","flat"))}
                A[q][m]["valid_order"]=gap(a,m,q,"rise","fall",valid=True)
                if m in ("R","S","V"):A[q][m]["common"]=gap(a,m,q,"rise","fall",arms=("R","S","V"))
                if m in ("V","O"):A[q][m]["version_common"]=gap(a,m,q,"rise","fall",arms=("V","O"))
        A["early_change"]=gap(a,"R","relief","rise","early")
        signal=[A["relief"]["R"][k] for k in ("order","recovery")]
        A["raw_update"]=all(x["n"]>=3 and x["d"]>=.5 for x in signal)
        A["raw_specific"]=all(A[q]["R"]["order"]["n"]>=3 and abs(A[q]["R"]["order"]["d"])<=.25 for q in ("other_person","onsite"))
        z=[A["relief"][m]["common"] for m in ("V","R","S")]
        inc=(z[0]["d"]-z[1]["d"]) if z[0]["n"] else None
        noise=abs(z[1]["d"]-z[2]["d"]) if z[0]["n"] else None
        passed=z[0]["n"]>=3 and inc>=.25 and inc>noise and all(
            A[q]["V"]["common"]["n"]>=3 and abs(A[q]["V"]["common"]["d"])<=abs(A[q]["R"]["common"]["d"])+.25
            for q in ("other_person","onsite"))
        A["increment"]=dict(n=z[0]["n"],v_minus_r=inc,resampling=noise,passed=passed)
        actors[a]=A
    return dict(count=len(rows),executed=sum(r["event"]["status"]=="executed" for r in rows.values()),
                protocol_valid=sum(r["valid"] for r in rows.values()),
                status=dict(Counter(r["event"]["status"] for r in rows.values())),actors=actors,
                expand=all(A["increment"]["passed"] for A in actors.values()))

if __name__=="__main__":
    out=HERE/"observed";data=results(json.loads((out/"choices.json").read_bytes()))
    assert data==json.loads((out/"results.json").read_bytes())
    for a,r in data["actors"].items():
        print(a,r["increment"],r["raw_update"],r["raw_specific"])
        for q in ("relief","other_person","onsite"):
            print(q,[(m,r[q][m]["order"]["d"],r[q][m]["order"]["n"]) for m in ("R","S","V","O")])
