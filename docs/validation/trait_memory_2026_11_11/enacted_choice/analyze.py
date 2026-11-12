"""Compute registered choice contrasts from story events, never from explanations."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
import trial as t
from persona.parse import parse

def avg(xs):return round(mean(xs),6) if xs else None

def count(rows):
    known=[r for r in rows if r["event"]["ticket"] is not None]
    return dict(planned=len(rows),committed=len(known),target=sum(r["event"]["target"] for r in known),
                other=sum(not r["event"]["target"] for r in known),missing=len(rows)-len(known),
                executed=sum(r["event"]["status"]=="executed" for r in rows),
                valid=sum(r["validity"]["valid"] for r in rows),
                valid_and_committed=sum(r["validity"]["valid"] for r in known))

def contrasts(by_scene, ids, layouts=range(4), valid_pair=False):
    ds={};exact=[];take=("G","F") if valid_pair else t.ARMS
    for q in ids:
        cases=[]
        for i in layouts:
            z=by_scene[q][i]
            if all(z[b]["event"]["ticket"] is not None and (not valid_pair or z[b]["validity"]["valid"]) for b in take):
                cases.append({b:int(z[b]["event"]["target"]) for b in take})
        vals={b:avg([v[b] for v in cases]) for b in take}
        if cases:exact.append({b:mean(v[b] for v in cases) for b in take})
        ds[q]=dict(n=len(cases),rates=vals)
    filled=[v for v in ds.values() if v["n"]]
    rates={b:avg([v[b] for v in exact]) for b in take}
    diffs={"GF":None,"SF":None,"GS":None,"G_pooled":None}
    if filled:
        for name,b,c in (("GF","G","F"),("SF","S","F"),("GS","G","S")):
            if b in take and c in take:diffs[name]=avg([r[b]-r[c] for r in exact])
        if not valid_pair:diffs["G_pooled"]=avg([r["G"]-(r["F"]+r["S"])/2 for r in exact])
    return dict(n=sum(x["n"] for x in filled),scenes=len(filled),rates=rates,
                wins=None if valid_pair else sum(x["rates"]["G"]>(x["rates"]["F"]+x["rates"]["S"])/2 for x in filled),
                diff=diffs,by_scene=ds)

def analyze(p,calibration=False):
    rows=t.read(p/"choices.json");logs=t.base.lines(p/"calls.jsonl.gz");seen=set()
    scenes=[q for q in t.SCENES if not calibration or q["family"]!="transfer"]
    assert set(rows)=={t.key(j) for j in t.jobs() if j[2] in scenes}=={r["id"] for r in logs} and len(logs)==24*len(scenes)
    for r in logs:
        z=rows[r["id"]];assert t.base.digest(r["messages"])==z["request"]
        assert parse(r["raw"]).say.strip()==z["say"].strip()
        if r["ok"]:
            k=r["response"]["id"];assert k not in seen;seen.add(k)
    ids={f:[q["id"] for q in scenes if q["family"]==f] for f in ("relation","transfer","control")}
    results={}
    for a in t.base.ROLES:
        cells={};scount={}
        for q in scenes:
            k=q["id"];cells[k]={i:{b:rows[t.key((a,b,q,i))] for b in t.ARMS} for i in range(4)}
            scount[k]={b:count([z[b] for z in cells[k].values()]) for b in t.ARMS}
        totals={f:{b:count([z[b] for k in ks for z in cells[k].values()]) for b in t.ARMS} for f,ks in ids.items()}
        main=contrasts(cells,ids["relation"]);vf=contrasts(cells,ids["relation"],valid_pair=True)
        sub={k:contrasts(cells,ids["relation"],v) for k,v in
             {"target_blue":[0,1],"target_white":[2,3],"target_first":[0,2],"target_last":[1,3]}.items()}
        control=sum(x["target"] for x in totals["control"].values())
        diff=main["diff"];g=totals["relation"]["G"];f=totals["relation"]["F"]
        bounds=[(g["target"]-f["target"]-f["missing"])/16,(g["target"]+g["missing"]-f["target"])/16]
        sufficient=main["n"]>=12 and all(x["n"]>=2 for x in main["by_scene"].values()) and control>=10
        criteria=dict(coverage_and_control=sufficient,
            magnitude=bool(main["n"] and diff["GF"]>=.125 and diff["GF"]>abs(diff["SF"]) and diff["GS"]>0),
            breadth=main["wins"]>=3,layout_robust=all(x["scenes"]==4 and x["diff"]["G_pooled"]>0 for x in sub.values()),
            valid_sensitivity=vf["n"]>=4 and vf["scenes"]>=2 and vf["diff"]["GF"]>0)
        results[a]=dict(main=main,valid_pair=vf,layouts=sub,controls=control,
                    gf_missing_bounds=bounds,by_scene=scount,by_family=totals,
                    transfer=contrasts(cells,ids["transfer"]) if ids["transfer"] else None,criteria=criteria,support=all(criteria.values()))
    return dict(roles=results,expand=all(v["support"] for v in results.values()),delivery=count(list(rows.values())),
                statuses=dict(Counter(v["event"]["status"] for v in rows.values())),calls=len(logs),success=len(seen),
                models=dict(Counter(r["response"]["model"] for r in logs if r["ok"])),
                tokens=sum(r["response"]["usage"]["total_tokens"] for r in logs if r["ok"]))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path);ap.add_argument("--handoff",action="store_true");ap.add_argument("--write",action="store_true")
    args=ap.parse_args();folder=args.out or (t.HERE/("handoff" if args.handoff else "observed"));res=analyze(folder,args.handoff);f=folder/"results.json"
    if args.write:t.write(f,res)
    else:assert t.read(f)==res
    print("Support:",{a:z["support"] for a,z in res["roles"].items()},"; device:",res["statuses"])
