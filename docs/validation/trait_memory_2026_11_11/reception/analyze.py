"""Recalculate all contrasts from the saved calls; never contacts a model."""
from collections import Counter
import argparse
import hashlib
import json
import re
from run import (ARMS, CELLS, D, F, HERE, MODES, ROOT, SCENES, STATES, T, cell_id,
                 read, rubric, sha, verify_inputs, write)
from persona.parse import parse


def decode_rating(raw):
    cleaned=re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$","",raw.strip(),flags=re.I)
    return json.loads(cleaned)


def summarize(rows, scores, key):
    info={k:{"value":scores[b]["score"],"error":scores[b]["error"],"b":b,
             "valid":rows[k]["validity"]["valid"],"visible":bool(rows[k]["say"].strip())}
          for b,k in key.items()}
    output={}
    def contrast(left,right,mode,valid):
        by_scene={}; diffs=[]
        for s in SCENES:
            pairs=[]
            for r in range(1,D["repeats"]+1):
                a,b=(info[cell_id((*arm,s,mode,r))] for arm in (left,right))
                if not a["visible"] or not b["visible"] or (valid and not(a["valid"] and b["valid"])):
                    continue
                pairs.append([a["value"],b["value"]])
                diffs.append(a["value"]-b["value"])
            by_scene[s]=pairs
        return dict(pairs=len(diffs),higher=sum(x>0 for x in diffs),equal=diffs.count(0),
                    lower=sum(x<0 for x in diffs),sum_difference=sum(diffs),by_scene=by_scene)
    contrasts={"natural":(("heard","step"),("passed","base")),
               "text_at_base":(("heard","base"),("passed","base")),
               "text_at_step":(("heard","step"),("passed","step")),
               "number_after_heard":(("heard","step"),("heard","base")),
               "number_after_passed":(("passed","step"),("passed","base"))}
    for mode in MODES:
        cells={}
        for a in ARMS:
            for st in STATES:
                group=[info[cell_id((a,st,s,mode,r))] for s in SCENES for r in range(1,D["repeats"]+1)]
                visible=[x for x in group if x["visible"]]
                cells[a+"."+st]={"n":len(visible),"valid":sum(x["valid"] for x in group),
                                "histogram":dict(Counter(str(x["value"]) for x in visible)),
                                "total":sum(x["value"] for x in visible),
                                "errors":sum(x["error"] for x in visible)}
        output[mode]={"cells":cells,"comparisons":{k:{name:contrast(*v,mode,valid)
                            for name,valid in (("visible",False),("valid_only",True))} for k,v in contrasts.items()}}
    return output


def analyze(obs):
    reg=read(obs/"registration.json"); calls=read(obs/"calls.json"); inputs=read(obs/"requests.json")
    for fname,h in reg["sha256"].items():
        assert hashlib.sha256((ROOT/fname).read_bytes()).hexdigest()==h,fname
    episodes=read(obs/"episodes.json"); turns=read(obs/"turns.json"); key=read(obs/"key.json")
    blind=read(obs/"blind.json"); coder=read(obs/"coding.json"); external={}
    assert len(key)==len(set(key.values()))==len(CELLS)
    assert set(blind)==set(key)==set(coder) and set(key.values())==set(turns)==set(reg["order"])
    assert set(inputs)=={v["request"] for v in calls.values()} and all(sha(m)==h for h,m in inputs.items())
    assert all(v["status"]=="ok" for v in calls.values())
    assert len(set(v["response"]["id"] for v in calls.values()))==len(calls)
    for a in ARMS:
        timeline=episodes[a]
        assert timeline["before"]==T["base"]
        for i,e in enumerate(timeline["receipts"]):
            ep=D["episodes"][i]
            assert e["source_say"]==F[ep["source"]]["say"] and F[ep["source"]]["validity"]["valid"]
            msgs=inputs[calls[e["call"]]["request"]]
            assert msgs[-2]["content"]==e["source_say"] and ep[a]==e["receipt"] and ep[a] in msgs[-1]["content"]
            assert parse(calls[e["call"]]["raw"]).say.strip()==e["actor_reply"].strip()
    assert abs(episodes["heard"]["after"]-.325)<1e-8 and episodes["passed"]["after"]==.31
    verify_inputs({k:inputs[calls[k]["request"]] for k in turns},episodes["heard"]["after"])
    for k,t in turns.items():
        assert parse(calls[k]["raw"]).say.strip()==t["say"].strip(),k
    for b,k in key.items():
        assert blind[b]["say"]==turns[k]["say"] and blind[b]["endpoint"]==k.split(".")[3]
    for kid,v in calls.items():
        if not kid.startswith("coding."):continue
        _,mode,_=kid.split("."); msgs=inputs[v["request"]]
        assert msgs[0]["content"]==rubric(mode)
        sent=json.loads(msgs[1]["content"]); rating=decode_rating(v["raw"])
        assert set(rating)==set(sent) and not(set(external)&set(sent))
        assert all(x==blind[b] and x["endpoint"]==mode for b,x in sent.items())
        external.update(rating)
    assert set(external)==set(blind)
    for b,x in blind.items():
        for c in (external,coder):
            if x["say"].strip():
                assert type(c[b]["score"]) is int and 0<=c[b]["score"]<=(2 if x["endpoint"]=="asked" else 1)
            assert type(c[b]["error"]) is bool and c[b]["reason"].strip()
    out={"provider_calls":len(calls),"experience":sum(k.startswith("experience.") for k in calls),
         "probe":len(turns),"coding_calls":sum(k.startswith("coding.") for k in calls),
         "valid_probes":sum(v["validity"]["valid"] for v in turns.values()),
         "valid_experience":sum(e["validity"]["valid"] for a in ARMS for e in episodes[a]["receipts"]),
         "models":dict(Counter(v["response"]["model"] for v in calls.values())),
         "usage":{n:sum(v["response"]["usage"][n] for v in calls.values())
                  for n in ("prompt_tokens","completion_tokens","total_tokens")},
         "study_coding":summarize(turns,coder,key),"second_coding":summarize(turns,external,key),
         "agreement":{mode:{"n":sum(x["endpoint"]==mode for x in blind.values()),
            "disagreements":[b for b,x in blind.items() if x["endpoint"]==mode and coder[b]["score"]!=external[b]["score"]]}
                      for mode in MODES}}
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--write",action="store_true")
    ap.add_argument("--observed",type=type(HERE),default=HERE/"observed");args=ap.parse_args()
    summary=analyze(args.observed)
    if args.write:write(args.observed/"results.json",summary)
    else:assert summary==read(args.observed/"results.json")
    print(f'{summary["provider_calls"]} calls; {summary["valid_probes"]}/{summary["probe"]} valid probes.')
    for m in MODES:
        print(m,summary["study_coding"][m]["comparisons"]["natural"])
