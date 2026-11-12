"""Compute only paired action contrasts. No text grading or provider access."""
from collections import Counter
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=json.loads((HERE/"design.json").read_bytes())
GROUPS={"old_near":("old",D["near"]), "new_near":("new",D["near"]),
        "old_crop":("old",D["unrelated"]), "new_crop":("new",D["unrelated"])}


def summarize(rows):
    def get(a,h,m,p,q,n):
        return rows[".".join((a,h,m,p,q,str(n)))]
    def val(r,strict):
        return int(r["event"]["target"]) if (r["event"]["status"]=="executed" and
                                             (not strict or r["validity"]["valid"])) else None
    def gap(a,mode,p,qs,common=False,strict=False):
        votes=[]; lo=hi=0;counts=Counter()
        for q in qs:
            for n in D["layouts"]:
                u,v=(val(get(a,h,mode,p,q,n),strict) for h in D["outcomes"])
                lo+=(0 if u is None else u)-(1 if v is None else v)
                hi+=(1 if u is None else u)-(0 if v is None else v)
                if common and any(val(get(a,h,m,p,q,n),strict) is None for m in D["memory"] for h in D["outcomes"]):continue
                if u is not None and v is not None: votes.append((u,v));counts[q]+=1
        total=len(qs)*len(D["layouts"]);n=len(votes)
        kept=sum(u for u,v in votes);missed=sum(v for u,v in votes)
        return dict(n=n,planned=total,kept=kept,missed=missed,
                    gap=(kept-missed)/n if n else None, per_task=dict(counts),
                    coverage=all(counts[q]>=3 for q in qs),
                    all_layout_bounds=[lo/total,hi/total])
    out={}
    for a in D["roles"]:
        person={}
        for kind,common,strict in (("paired",False,False),("shared",True,False),("protocol_valid",False,True)):
            groups={}
            for m in D["memory"]:
                gs={g:gap(a,m,p,qs,common,strict) for g,(p,qs) in GROUPS.items()}
                parts=[gap(a,m,"old",[q],common,strict) for q in D["near"]]
                gate=(all(x["coverage"] for x in gs.values()) and gs["old_near"]["gap"]>=.5
                      and all(x["gap"]>0 for x in parts) and all(abs(gs[g]["gap"])<=.25 for g in list(GROUPS)[1:]))
                groups[m]=dict(groups=gs,old_near_tasks=dict(zip(D["near"],parts)),selective=gate)
            person[kind]=groups
        groups=person["shared"]
        def effect(m,g):return groups[m]["groups"][g]["gap"]
        enough=all(x["coverage"] for y in groups.values() for x in y["groups"].values())
        near=[effect(m,"old_near") for m in D["memory"]]
        inc=noise=None
        if None not in near:
            inc=near[1]-near[0];noise=abs(near[2]-near[0])
        gate=enough and inc>=.25 and inc>noise and all(abs(effect("J",g))<=abs(effect("R",g))+.25 for g in list(GROUPS)[1:])
        person["increment"]=dict(coverage=enough,J_minus_R=inc,raw_resample_abs=noise,passed=gate)
        out[a]=person
    return dict(planned=len(rows),committed=sum(r["event"]["status"]=="executed" for r in rows.values()),
                protocol_valid=sum(r["validity"]["valid"] for r in rows.values()),
                statuses=dict(Counter(r["event"]["status"] for r in rows.values())),actors=out,
                expand_all=all(x["increment"]["passed"] for x in out.values()))

if __name__=="__main__":
    out=HERE/"observed"
    data=summarize(json.loads((out/"choices.json").read_bytes()))
    assert data==json.loads((out/"results.json").read_bytes())
    for a,r in data["actors"].items():
        print(a, r["increment"])
        for g in GROUPS:
            print(g,[(m,r["paired"][m]["groups"][g]["gap"],r["paired"][m]["groups"][g]["n"]) for m in D["memory"]])
