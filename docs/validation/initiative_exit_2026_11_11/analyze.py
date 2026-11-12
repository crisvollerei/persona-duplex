"""Rejoin sealed, method-hidden observations. Recompute without keys or model calls."""
from collections import Counter
import hashlib
import json
import random
from statistics import median
from study import HERE, ROOT, PARAMS, baseline, config, digest, load, write


def joined(stage):
    reg = load(HERE/f"{stage}_registration.json")
    calls=load(HERE/f"{stage}_calls.json")
    assert [r["id"] for r in calls]==[j["id"] for j in reg["jobs"]]
    assert len(set(j["id"] for j in reg["jobs"]))==len(calls)
    packet=load(HERE/f"{stage}_blind.json")
    coding=load(HERE/f"{stage}_coding.json")
    assert coding["packet_sha256"] == hashlib.sha256((HERE/f"{stage}_blind.json").read_bytes()).hexdigest()
    codes={r["id"]:r for r in coding["items"]}
    assert len(codes)==len(coding["items"])
    combined=[{**j,**r} for j,r in zip(reg["jobs"],calls)]
    delivered=[r for r in combined if r["say"]]
    random.Random(91871).shuffle(delivered)
    assert len(delivered)==len(packet)
    used=set()
    for i,(r,p) in enumerate(zip(delivered,packet),1):
        assert p==dict(id=f"{stage[0]}{i:02d}",user=r["user"],say=r["say"])
        if r["mode"]=="open":
            assert p["id"] not in codes
        else:
            label=codes[p["id"]]["behavior"]
            assert label in ("met","failed","unclear")
            r["behavior"]=label
            used.add(p["id"])
        r["blind_id"]=p["id"]
    assert used==set(codes)
    turns={}
    for r in combined:
        body=reg["requests"][r["request"]]
        assert digest(body)==r["request"] and {k:v for k,v in body.items() if k!="messages"}==PARAMS
        assert body["messages"][-1]==dict(role="user",content=r["user"])
        previous=turns.setdefault((r["case"],r["mode"],r["route"]),body)
        assert previous==body
    for (case,mode,route),body in turns.items():
        if route=="full":
            assert baseline(config(),body["messages"])==turns[case,mode,"strong"]["messages"]
    return combined


def summarize():
    cases={r["id"]:r for r in load(HERE/"cases.json")}
    probes={r["id"]:r for r in load(HERE/"trajectory.json")["probes"]}
    output={}
    for stage in ("short","long"):
        rr=joined(stage); metrics={}
        for route in ("full","strong"):
            rows=[r for r in rr if r["route"]==route]
            modes={}
            for mode in sorted({r["mode"] for r in rows}):
                subs=[r for r in rows if r["mode"]==mode]
                modes[mode]=dict(planned=len(subs), delivered=sum(bool(r["say"]) for r in subs),
                    behavior=dict(Counter(r["behavior"] for r in subs if "behavior" in r)))
            exact=[(r,(probes if stage=="long" else cases)[r["case"]]) for r in rows
                   if stage=="long" and "expected" in probes[r["case"]] or
                   stage=="short" and r["mode"]=="bounded" and "exact" in cases[r["case"]]]
            n_exact=sum(r["say"]==c.get("expected",c.get("exact")) for r,c in exact)
            assert all((r["behavior"]=="met")==(r["say"]==c.get("expected",c.get("exact")))
                       for r,c in exact if "behavior" in r)
            usage=[r["usage"]["total_tokens"] for r in rows if r["usage"]]
            metrics[route]=dict(planned=len(rows),delivered=sum(bool(r["say"]) for r in rows),
                provider_models=dict(Counter(r["response_model"] for r in rows if r["response_model"])),
                full_protocol_invalid=sum((r.get("validity") or {}).get("valid") is False for r in rows)
                if route=="full" else None, reported_usage=len(usage),total_tokens=sum(usage),
                median_call_seconds=median(r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None),
                exact_delivery=[n_exact,len(exact)],modes=modes)
        output[stage]=metrics
        if stage=="short":
            # Two failures of one type either occur across themes or repeat within one theme.
            def hit(labels):
                return any(sum(r.get("behavior") in labels for r in rr
                    if r["route"]=="full" and r["mode"]==m)>=2 for m in ("bounded","choice"))
            output["decision"]=dict(edit_triggered=hit({"failed"}),
                treating_unclear_as_failure_would_trigger=hit({"failed","unclear"}),
                missing_or_empty=sum(not r.get("say") for r in rr),real_human_ratings_collected=0)
    registration=load(HERE/"history_registration.json")
    calls=load(HERE/"history_calls.json")
    assert [r["id"] for r in calls]==[j["id"] for j in registration["jobs"]]
    original=load(HERE/"long_registration.json")
    parent={j["case"]:original["requests"][j["request"]] for j in original["jobs"] if j["route"]=="full"}
    check={}
    for arm in ("body","native"):
        rows=[(j,r) for j,r in zip(registration["jobs"],calls) if j["arm"]==arm]
        for j,r in rows:
            body=registration["requests"][j["request"]]
            assert digest(body)==j["request"]
            if arm=="body":assert body==parent[j["case"]]
            else:
                a=baseline(config(),body["messages"])
                b=baseline(config(),parent[j["case"]]["messages"])
                assert a==b and body["messages"][0]==parent[j["case"]]["messages"][0]
        exact=[(r,probes[j["case"]]["expected"]) for j,r in rows if "expected" in probes[j["case"]]]
        check[arm]=dict(planned=len(rows),delivered=sum(bool(r["say"]) for _,r in rows),
            protocol_valid=sum((r["validity"] or {}).get("valid") is True for _,r in rows),
            exact_delivery=[sum(r["say"]==v for r,v in exact),len(exact)],
            exact_misses=[r["id"] for r,v in exact if r["say"]!=v],
            tokens=sum(r["usage"]["total_tokens"] for _,r in rows if r["usage"]),
            reported_usage=sum(r["usage"] is not None for _,r in rows))
    output["layout_followup"]=check
    return output


if __name__=="__main__":
    result=summarize()
    write(HERE/"results.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
