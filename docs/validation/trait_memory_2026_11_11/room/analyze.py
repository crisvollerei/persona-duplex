"""Offline integrity checks and unblinding for the conversational-room pilot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from study import D,HERE,JUDGE,SC,VAL,blind_item,check_inputs,check_sources,old,read,sha,tasks,write
from persona.parse import parse


def results(rows, coding, mapping):
    scored={k:dict(valid=rows[k]["validity"]["valid"],score=coding[b]["score"],
                  error=coding[b]["material_error"]) for b,k in mapping.items()}
    groups={}
    for m,arms in D["cells"].items():
        for a in arms:
            group=a+"."+m
            def summarize(xs):
                obs=[x for x in xs if x["score"] is not None]
                return dict(planned=len(xs),delivered=len(obs),valid=sum(x["valid"] for x in xs),
                    any=sum(x["score"]>=1 for x in obs),specific=sum(x["score"]==2 for x in obs),
                    errors=sum(x["error"] for x in obs),ordinal_sum=sum(x["score"] for x in obs))
            by_s={s:[scored[f"{a}.{s}.{m}.{r}"] for r in range(1,D["repeats"]+1)] for s in SC}
            groups[group]={"all":summarize([z for vs in by_s.values() for z in vs]),
                           "scenes":{s:summarize(vs) for s,vs in by_s.items()}}
    specs={"natural_room":("heard.step.room","passed.base.room"),
           "natural_ordinary":("heard.step.ordinary","passed.base.ordinary"),
           "history_at_base":("heard.base.room","passed.base.room"),
           "history_at_step":("heard.step.room","passed.step.room"),
           "step_after_heard":("heard.step.room","heard.base.room"),
           "step_after_passed":("passed.step.room","passed.base.room"),
           "room_after_heard":("heard.step.room","heard.step.ordinary"),
           "room_after_passed":("passed.base.room","passed.base.ordinary")}
    def compare(a,b,valid_only):
        def fetch(cell,s,r):
            h,n,m=cell.split(".");return scored[f"{h}.{n}.{s}.{m}.{r}"]
        pairs=[]
        for s in SC:
            for r in range(1,D["repeats"]+1):
                l,h=fetch(a,s,r),fetch(b,s,r)
                if l["score"] is None or h["score"] is None or (valid_only and not(l["valid"] and h["valid"])):continue
                pairs.append((s,l["score"],h["score"]))
        contrasts={}
        for name,threshold in (("any",1),("specific",2)):
            ds=[int(l>=threshold)-int(h>=threshold) for s,l,h in pairs]
            contrasts[name]={"higher":ds.count(1),"tied":ds.count(0),"lower":ds.count(-1),"difference":sum(ds),
                 "by_scene":{s:sum(int(l>=threshold)-int(h>=threshold) for t,l,h in pairs if s==t) for s in SC}}
        return dict(n=len(pairs),pair_scores=pairs,**contrasts)
    comparisons={k:{q:compare(a,b,q=="valid") for q in ("visible","valid")} for k,(a,b) in specs.items()}
    return {"groups":groups,"contrasts":comparisons}


def analyze(out):
    check_sources()
    reg=read(out/"registration.json"); turns=read(out/"turns.json");calls=read(out/"calls.json")
    inputs=read(out/"requests.json"); mapping=read(out/"key.json");blind=read(out/"blind.json")
    primary=read(out/"coding.json");external={};closed={}
    for path,h in reg["sha256"].items():
        assert hashlib.sha256((old.ROOT/path).read_bytes()).hexdigest()==h,path
    assert reg["plan"]==tasks() and reg["values"]==VAL and set(turns)==set(tasks())
    assert len({v["response"]["id"] for v in calls.values()})==len(calls)==42
    assert all(v["status"]=="ok" for v in calls.values())
    assert all(sha(m)==h for h,m in inputs.items()) and set(inputs)=={v["hash"] for v in calls.values()}
    assert set(mapping.values())==set(turns) and len(set(mapping.values()))==len(mapping)
    assert set(blind)==set(primary)==set(mapping)
    msgs={k:inputs[calls[k]["hash"]] for k in turns}
    check_inputs(msgs)
    for b,k in mapping.items():
        assert calls[k]["params"]==D["provider"] and parse(calls[k]["raw"]).say.strip()==turns[k]["say"].strip()
        assert blind[b]==blind_item(k,turns[k]["say"])
    for k in set(calls)-set(turns):
        call=calls[k];m=inputs[call["hash"]]
        assert k.startswith("judge.") and call["params"]==D["judge"] and m[0]["content"]==JUDGE
        batch=json.loads(m[1]["content"])
        text=re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$","",call["raw"].strip(),flags=re.I)
        try:
            rating=json.loads(text)
        except json.JSONDecodeError as e:
            # Accept only a missing OUTER closing brace after all individual objects.
            # IDs, field types, counts and complete reason strings are checked below.
            if e.pos!=len(text) or not text.endswith("}"):raise
            rating=json.loads(text+"}")
            closed[k]=list(rating)
        assert set(rating)==set(batch) and not(set(rating)&set(external))
        assert all(blind[b]==x for b,x in batch.items())
        external.update(rating)
    assert set(external)==set(primary)
    for coding in (primary,external):
        for b,x in coding.items():
            assert (type(x["score"]) is int and 0<=x["score"]<=2) if blind[b]["say"].strip() else x["score"] is None
            assert type(x["material_error"]) is bool and x["reason"].strip()
    counts=Counter(reason for x in turns.values() for reason in x["validity"]["reasons"] if not x["validity"]["valid"])
    return {"calls":len(calls),"foreground":len(turns),"judge_calls":len(calls)-len(turns),
       "delivered":sum(bool(x["say"].strip()) for x in turns.values()),
       "valid":sum(x["validity"]["valid"] for x in turns.values()),"invalid_reasons":dict(counts),
       "models":dict(Counter(v["response"]["model"] for v in calls.values())),
       "usage":{key:sum(v["response"]["usage"][key] for v in calls.values()) for key in
                ("prompt_tokens","completion_tokens","total_tokens")},
       "closed_outer_brace":closed,"secondary_coding":external,
       "primary":results(turns,primary,mapping),"secondary":results(turns,external,mapping),
       "different_grade":[b for b in primary if primary[b]["score"]!=external[b]["score"]],
       "different_any":[b for b in primary if bool(primary[b]["score"])!=bool(external[b]["score"])]}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--write",action="store_true")
    ap.add_argument("--observed",type=Path,default=HERE/"observed");args=ap.parse_args()
    s=analyze(args.observed);p=args.observed/"results.json"
    if args.write:write(p,s)
    else:assert read(p)==json.loads(json.dumps(s))
    print(f'{s["calls"]} calls; {s["valid"]}/{s["foreground"]} protocol valid.')
    for who in ("primary","secondary"):
        print(who,s[who]["contrasts"]["natural_room"]["visible"])
