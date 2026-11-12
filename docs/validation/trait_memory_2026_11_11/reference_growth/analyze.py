"""Unblind the fixed-budget samples, retain missingness, and tabulate the preset decision."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
import experiment as e
from persona.parse import parse

def avg(xs):
    return round(mean(xs),6) if xs else None

def tables(rows,score):
    def get(a,b,q,r):
        k=f"{a}.{b}.{q}.{r}"
        return score[k] if rows[k]["say"].strip() else None
    out={}
    def delta(a,first,second,scopes,complete=False):
        diffs=[];by={}
        for q in scopes:
            vs=[]
            for r in (1,2):
                x,y=[get(a,b,q,r) for b in (first,second)]
                if x is None or y is None:continue
                if complete and not all(rows[f"{a}.{b}.{q}.{r}"]["validity"]["valid"] for b in (first,second)):continue
                vs.append(x-y)
            by[q]=avg(vs);diffs.extend(vs)
        return dict(n=len(diffs),mean=avg(diffs),scenes=sum(x is not None for x in by.values()),by_scene=by)
    qs=[q["id"] for q in e.scenes]
    for a,arms in e.ARMS.items():
        arm_stats={}
        for b in arms:
            vs=[get(a,b,q,r) for q in qs for r in (1,2)];counts=Counter(vs)
            arm_stats[b]=dict(mean=avg([v for v in vs if v is not None]),levels=[counts[i] for i in range(3)],
                              missing=counts[None],valid=sum(rows[f"{a}.{b}.{q}.{r}"]["validity"]["valid"] for q in qs for r in (1,2)))
        gf=delta(a,"G","F",qs);sf=delta(a,"S","F",qs);valid=delta(a,"G","F",qs,True)
        by_scene={}
        for q in qs:
            data={b:[get(a,b,q,r) for r in (1,2)] for b in arms}
            def combined():
                vals=data["F"]+data["S"]+data["G"]
                return None if None in vals else avg(data["G"])-avg(data["F"]+data["S"])
            by_scene[q]={"values":data,"vs_two_frozen":combined()}
        wins=sum(v["vs_two_frozen"] is not None and v["vs_two_frozen"]>0 for v in by_scene.values())
        primary_ok=(gf["n"]==sf["n"]==12 and gf["mean"]>abs(sf["mean"]) and wins>=4 and
                    valid["n"]>=3 and valid["scenes"]>=2 and valid["mean"]>0)
        out[a]=dict(arms=arm_stats,growth_minus_frozen=gf,sham_minus_frozen=sf,valid_growth_minus_frozen=valid,
                    by_scene=by_scene,wins=wins,primary_ok=primary_ok)
        if a=="tang":
            m={b:v["mean"] for b,v in arm_stats.items()}
            if all(not v["missing"] for v in arm_stats.values()):
                # Compute using raw levels rather than rounded cell means.
                m={b:mean(get(a,b,q,r) for q in qs for r in (1,2)) for b in arms}
                f=(m["F"]+m["S"])/2
                out[a]["factorial"]={k:round(v,6) for k,v in
                  {"number_hint_on":m["N"]-f,"number_hint_off":m["G"]-m["H"],
                   "remove_hint_low_number":m["H"]-f,"remove_hint_high_number":m["G"]-m["N"]}.items()}
            else:out[a]["factorial"]=None
    return out

def analyze(p):
    rows=e.read(p/"probes.json");blind=e.read(p/"blind.json");key=e.read(p/"key.json")
    assert set(rows)=={e.key(j) for j in e.suite()} and set(key)==set(blind) and set(key.values())==set(rows)
    calls=e.base.lines(p/"calls.jsonl.gz")
    assert len(calls)==len({r["id"] for r in calls})==len({r["response"]["id"] for r in calls})==120
    for r in calls:
        if r["id"].startswith("sample."):
            v=rows[r["id"][7:]];assert v["request"]==e.base.digest(r["messages"])
            assert v["say"].strip()==parse(r["raw"]).say.strip()
    res={};ratings={};err={}
    for method,name in (("reader","reading.json"),("model","model_coding.json")):
        raw=e.read(p/name);assert set(raw)==set(blind)
        for b,z in raw.items():
            assert (z["score"] is None or (type(z["score"]) is int and 0<=z["score"]<=2)) and z["why"]
            assert z["error"] is None or type(z["error"]) is bool
            assert rows[key[b]]["say"]==blind[b]["say"]
        ratings[method]={key[b]:v["score"] for b,v in raw.items()}
        err[method]=[b for b,z in raw.items() if z["error"] is True]
        res[method]=tables(rows,ratings[method])
    agree=[ratings["reader"][k]==ratings["model"][k] for k in rows
           if ratings["reader"][k] is not None and ratings["model"][k] is not None]
    confusion=[[sum(ratings["reader"][k]==i and ratings["model"][k]==j for k in rows)
                for j in range(3)] for i in range(3)]
    passed={}
    for a in e.ARMS:
        g=res["model"][a]["growth_minus_frozen"]
        passed[a]=res["reader"][a]["primary_ok"] and g["n"]==12 and g["mean"]>0
    return dict(tables=res,passes=passed,expand=all(passed.values()),total=len(rows),
            valid=sum(v["validity"]["valid"] for v in rows.values()),nonempty=sum(bool(v["say"].strip()) for v in rows.values()),
            agreement=[sum(agree),len(agree)],reader_by_model_counts=confusion,material_errors=err,
            calls=dict(Counter(r["id"].split(".")[0] for r in calls)),models=dict(Counter(r["response"]["model"] for r in calls)),
            tokens=sum(r["response"]["usage"]["total_tokens"] for r in calls))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=e.HERE/"observed");p.add_argument("--write",action="store_true")
    args=p.parse_args();d=analyze(args.out);dest=args.out/"results.json"
    if args.write:e.write(dest,d)
    else:assert e.read(dest)==d
    print("Support by role:",d["passes"],"; valid deliveries:",d["valid"],"/",d["total"])
