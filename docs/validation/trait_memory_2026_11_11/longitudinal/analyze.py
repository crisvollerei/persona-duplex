"""Unblind and tabulate the paired stage measurements. No network or synthetic imputation."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
from run import (ARMS,BASE,D,GRADE,HERE,NAME,P,ROOT,ROLES,TEST,blind_item,digest,lines,obj,read,write)
from persona.parse import parse

def mean(x):return round(sum(x)/len(x),4) if x else None
def contrasts(ratings,key,probes):
    data={k:ratings[b] for b,k in key.items()}
    out={}
    for a in ROLES:
        stages={}
        for cp in D["checkpoints"]:
            pairs=[]
            for q in TEST:
                if q["stage"] not in (cp,"anchor"):continue
                for rep in range(1,D["repeats"]+1):
                    ids=[f'test.{cp}.{a}.{arm}.{q["id"]}.{rep}' for arm in ARMS]
                    vals=[data[k]["score"] for k in ids]
                    pairs.append(dict(task=q["id"],anchor=q["stage"]=="anchor",rep=rep,scores=vals,
                                valid=all(probes[k]["validity"]["valid"] for k in ids),
                                valid_each=[probes[k]["validity"]["valid"] for k in ids]))
            def report(ps):
                ps=[p for p in ps if None not in p["scores"]];v=[p["scores"] for p in ps]
                diff=[g-f for g,f in v]
                return dict(n=len(v),growing_mean=mean([g for g,f in v]),frozen_mean=mean([f for g,f in v]),
                     difference=mean(diff),higher=sum(d>0 for d in diff),equal=diff.count(0),lower=sum(d<0 for d in diff),
                     growing_any=sum(g>=1 for g,f in v),frozen_any=sum(f>=1 for g,f in v),
                     growing_specific=sum(g==2 for g,f in v),frozen_specific=sum(f==2 for g,f in v))
            stages[str(cp)]={k:report(ps) for k,ps in (("all",pairs),("valid",[p for p in pairs if p["valid"]]),
                    ("new",[p for p in pairs if not p["anchor"]]),("anchor",[p for p in pairs if p["anchor"]]))}
            stages[str(cp)]["pairs"]=pairs
        out[a]=stages
    return out

def analyze(out):
    cs=lines(out/"calls.jsonl");req=lines(out/"requests.jsonl.gz");calls={x["id"]:x for x in cs}
    inputs={x["hash"]:x["messages"] for x in req}; reg=read(out/"registration.json")
    assert len(cs)==len(calls)==204 and len(inputs)==len(req)
    assert all(x["status"]=="ok" for x in cs) and len({x["response"]["id"] for x in cs})==len(cs)
    assert all(digest(m)==h for h,m in inputs.items()) and {x["request"] for x in cs}==set(inputs)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in reg["files"].items())
    for a in ROLES:
        mat=reg["material"][a]["self_disclosure"]
        assert mat["anchor"]==BASE[a] and all(mat[k]==P[k] for k in ("plasticity","drift"))
    hist=read(out/"trajectory.json");probes=read(out/"probes.json");snaps=read(out/"checkpoints.json")
    assert len(hist)==24 and [e["episode"] for e in hist]==list(range(1,25))
    assert set(probes)=={k for group in reg["measure_order"].values() for k in group}
    assert all(parse(calls[t["call"]]["raw"]).say.strip()==t["say"].strip() for e in hist for t in e["turns"])
    assert all(parse(calls[k]["raw"]).say.strip()==v["say"].strip() for k,v in probes.items())
    key=read(out/"key.json");blind=read(out/"blind.json");primary=read(out/"coding.json")
    assert set(primary)==set(blind)==set(key) and set(key.values())==set(probes) and len(key)==72
    for b,k in key.items():assert blind[b]==blind_item(k,probes[k]["say"])
    secondary={}
    for k,c in calls.items():
        expected=D["judge"] if k.startswith(("gate.","grade.")) else D["foreground"]
        assert c["params"]==expected
        if k.startswith("grade."):
            msgs=inputs[c["request"]];assert msgs[0]["content"]==GRADE
            sent=json.loads(msgs[1]["content"]);rated=obj(c["raw"])
            assert set(sent)==set(rated) and not(set(rated)&set(secondary))
            assert all(blind[b]==v for b,v in sent.items());secondary.update(rated)
    assert set(secondary)==set(key)
    for coder in (primary,secondary):
        for b,v in coder.items():
            assert (type(v["score"]) is int and 0<=v["score"]<=2) if blind[b]["say"].strip() else v["score"] is None
            assert type(v["error"]) is bool and v["why"]
    first,second=(contrasts(c,key,probes) for c in (primary,secondary))
    audit=read(out/"evidence_audit.json")
    accepted={f'{e["episode"]}.{a}' for e in hist for a in ROLES if e["evidence"][a]["verified"]}
    assert set(audit)==accepted and all(v[0] in ("支持","不支持") and v[1] for v in audit.values())
    roles={}
    for a in ROLES:
        votes=[e["evidence"][a] for e in hist]
        v=[BASE[a]]+[e["growing"]["value"] for e in votes]
        assert all(z["frozen"]["value"]==BASE[a] for z in votes)
        for ep in D["checkpoints"]:assert abs(snaps[str(ep)]["values"][a]["growing"]-v[ep])<1e-7
        assert all(BASE[a]-P["drift"]-1e-7<=x<=BASE[a]+P["drift"]+1e-7 for x in v)
        counts=[sum(bool(x["growing"]["changes"]) for x in votes[:n]) for n in D["checkpoints"]]
        stage=[first[a][str(t)]["all"]["difference"] for t in D["checkpoints"]]
        f=first[a]["24"];s=second[a]["24"]
        trajectory_ok=all(x is not None for x in stage) and stage[-1]>0 and stage[-1]>max(stage[:2])
        ok=bool(counts[1]>=2 and counts[2]>=4 and trajectory_ok and all(
            f[group]["difference"] is not None and f[group]["difference"]>=0 for group in ("new","anchor")) and all(
            g["difference"] is not None and g["difference"]>0 for g in (f["valid"],s["all"])))
        roles[a]=dict(values=v,checkpoint_pushes=counts,positive=sum(z["verified"]==1 for z in votes),
            negative=sum(z["verified"]==-1 for z in votes),zero=sum(not z["verified"] for z in votes),
            explicit_zero=sum(type(z["proposed"]) is int and z["proposed"]==0 for z in votes),
            positive_audit_supported=sum(audit[k][0]=="支持" for k in accepted if k.endswith("."+a)),
            withheld_nonzero=sum(z["proposed"] in (-1,1) and not z["verified"] for z in votes),
            unknown_labels=sum(type(z["proposed"]) is not int for z in votes),
            crossed_hint=[i for i in range(1,len(v)) if (v[i-1]<=.35)!=(v[i]<=.35)],supports_accumulation=ok)
    return dict(roles=roles,primary=first,secondary=second,second_coding=secondary,
       support=all(v["supports_accumulation"] for v in roles.values()),calls=len(cs),
       call_types=dict(Counter(c["id"].split(".")[0] for c in cs)),
       disagreements=[b for b in primary if primary[b]["score"]!=secondary[b]["score"]],
       material_errors={"primary":[b for b in primary if primary[b]["error"]],"secondary":[b for b in secondary if secondary[b]["error"]]},
       plot_valid=sum(t["validity"]["valid"] for e in hist for t in e["turns"]),
       probe_valid=sum(t["validity"]["valid"] for t in probes.values()),
       models=dict(Counter(c["response"]["model"] for c in cs)),usage={k:sum(c["response"]["usage"][k] for c in cs)
            for k in ("prompt_tokens","completion_tokens","total_tokens")})

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--observed",type=Path,default=HERE/"observed")
    ap.add_argument("--write",action="store_true");args=ap.parse_args()
    res=analyze(args.observed);p=args.observed/"results.json"
    if args.write:write(p,res)
    else:assert read(p)==res
    for a in ROLES:print(NAME[a],res["roles"][a],{cp:x["all"]["difference"] for cp,x in res["primary"][a].items()})
