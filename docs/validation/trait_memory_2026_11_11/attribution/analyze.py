"""Tabulate all predeclared comparisons; scores and responses are never imputed."""
from collections import Counter
import argparse
import statistics

import study as s
from confirm import metrics


def avg(vals):
    return round(statistics.mean(vals),4) if vals else None


def behavior(probes,ratings):
    def diff(rows,i,j,valid=False):
        vals=[row[i][1]-row[j][1] for row in rows if row[i][1] is not None and row[j][1] is not None
              and (not valid or (row[i][0] and row[j][0]))]
        return [avg(vals),len(vals)]
    out={}
    for a in s.prev.ROLES:
        for family in ("asked","room"):
            rows=[]
            for q in s.qs:
                if q["type"]!=family:continue
                for r in (1,2):
                    row=[]
                    for arm in s.arms:
                        k=f"{a}.{arm}.{q['id']}.{r}"
                        row.append([probes[k]["validity"]["valid"],ratings[k] if probes[k]["say"].strip() else None])
                    rows.append(row)
            out[a+"."+family]={"G-F":diff(rows,0,1),"S-F":diff(rows,2,1),"valid_G-F":diff(rows,0,1,True),
                "by_arm":[[x[i][1] for x in rows] for i in range(3)],"paired":rows}
    return out

def outcome(p,co):
    h,ref=s.exchanges();base=s.read(p/"probes.json")
    hs=[k for k in h if k[0]=="H"];eligible=[k for k in hs if h[k].protocol_ok and ref[k] is not None]
    old={k:s.hist[int(k[1:3])-1]["evidence"][k.split(".")[1]] for k in hs}
    orig={k:v["verified"] for k,v in old.items()}
    # Exploratory mechanical sensitivity: apply only the pre-labelled, eligible reference.
    # Never generate or score dialogue under this counterfactual.
    reference_states=s.evolve({k:{"reason":"预先固定的文本参考",
        "quotes":[h[k].first,h[k].second],"vote":ref[k] if h[k].protocol_ok and ref[k] is not None else 0} for k in hs})
    def book(v):
        m=metrics(eligible,ref,v);m["fp"]=m["booked"]-m["tp"];m["fn"]=m["reference"]-m["tp"]
        return m
    gates={"original":book(orig)}
    stats={};checks={};traces={};data={};calls={};usage={};role_snaps={}
    for name,folder,pref in (("single",p,"C"),("ordered",co,"N")):
        rows=s.read(folder/"attribution.json");votes={k:r["vote"] for k,r in rows.items()}
        gates[name]=book(votes)
        controls=(s.read(s.HERE/"controls.json") if pref=="C" else s.read(s.HERE/"confirmation.json")["controls"])
        ids=[f"{pref}{i:02}" for i in range(1,17)];refs={k:item[-1] for k,item in zip(ids,controls)}
        checks[name]={"correct":sum(rows[k]["label"]==refs[k] for k in ids),
              "unknown":[k for k in ids if rows[k]["label"] is None],
              "incorrect":[k for k in ids if rows[k]["label"] is not None and rows[k]["label"]!=refs[k]],
              "wrong_owner_votes":sum(bool(votes[ids[i]]) for i in (2,6,10,14)),
              "old_false_positive_retained":[k for k in eligible if orig[k] and orig[k]!=ref[k] and votes[k]]}
        traces[name]=dict(missing=sum(rows[k]["label"] is None for k in hs),valid_pairs=sum(h[k].protocol_ok for k in hs),
                          disputed_booked=[k for k in hs if ref[k] is None and votes[k]],
                          missed=[k for k in eligible if ref[k] and votes[k]!=ref[k]],
                          false_positive=[k for k in eligible if votes[k] and votes[k]!=ref[k]],
                          blocked_positive=[k for k in hs if not h[k].protocol_ok and rows[k]["label"]==1],
                          gain=[k for k in eligible if ref[k] and votes[k]==ref[k] and orig[k]!=ref[k]])
        tr=s.read(folder/"trajectories.json")
        role_snaps[name]={a:{"values":[s.prev.BASE[a],vs[11]["growing"],vs[-1]["growing"]],
                             "pushed":[i for i,x in enumerate(vs,1) if x["push"]]} for a,vs in tr.items()}
        logs=s.prev.lines(folder/"calls.jsonl.gz")
        assert len({r["id"] for r in logs})==len(logs)==len({r["response"]["id"] for r in logs})
        calls[name]=dict(Counter(r["id"].split(".")[0] for r in logs))
        usage[name]={"tokens":sum(r["response"]["usage"]["total_tokens"] for r in logs),
                    "models":dict(Counter(r["response"]["model"] for r in logs))}
    def decode(folder,keyfile,coding):
        key=s.read(folder/keyfile);r=s.read(folder/coding)
        assert set(key)==set(r) and len(set(key.values()))==len(key)
        assert all((type(v["score"]) is int and 0<=v["score"]<=2 or v["score"] is None) and v["why"] for v in r.values())
        return {k:r[b]["score"] for b,k in key.items()}
    data1={who:decode(p,"key.json",file) for who,file in (("reading","reading.json"),("model","model_coding.json"))}
    decision=s.read(co/"decision.json");new=dict(base)
    data2={who:dict(z) for who,z in data1.items()}
    if decision["retest"]:
        new.update(s.read(co/"probes.json"))
        for who,file in (("reading","reading.json"),("model","model_coding.json")):
            data2[who].update(decode(co,"blind_key.json",file))
    for name,b,c in (("single",base,data1),("ordered",new,data2)):
        if name=="ordered" and not decision["passes"]:continue
        stats[name]={who:behavior(b,z) for who,z in c.items()}
        data[name]={"protocol_valid":sum(z["validity"]["valid"] for z in b.values()),"denominator":len(b),
                    "grade_agreement":sum(x is not None and x==c["model"][k] for k,x in c["reading"].items())}
        stats[name]["passes"]={who:all(x["G-F"][1]==4 and x["S-F"][1]==4 and
                 x["G-F"][0]>abs(x["S-F"][0]) and x["valid_G-F"][1]>0 and x["valid_G-F"][0]>0
                 for x in stats[name][who].values()) for who in c}
    return dict(gates=gates,controls=checks,traces=traces,snapshots=role_snaps,expression=stats,
                reference_only_states={a:{"values":[s.prev.BASE[a],v[11]["growing"],v[-1]["growing"]],
                   "pushed":[i for i,x in enumerate(v,1) if x["push"]]} for a,v in reference_states.items()},
                delivery=data,decision=decision,calls=calls,usage=usage,history=dict(opportunities=len(hs),
                eligible=len(eligible),ambiguous=sum(ref[k] is None for k in hs),protocol_ok=sum(h[k].protocol_ok for k in hs)))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--write",action="store_true");args=ap.parse_args()
    res=outcome(s.HERE/"observed",s.HERE/"confirmed");dst=s.HERE/"results.json"
    if args.write:s.write(dst,res)
    else:assert s.read(dst)==res
    print(res["gates"],res["controls"],res["snapshots"])
    print({n:{k:(v["G-F"],v["S-F"]) for k,v in z["reading"].items()} for n,z in res["expression"].items()})
