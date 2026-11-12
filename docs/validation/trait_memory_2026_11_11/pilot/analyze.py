"""Audit paired inputs, independently replay authored votes, and tabulate sealed SAY coding."""
from collections import Counter
import hashlib
from run import BASE, D, HERE, OTHER, TRAIT, W, digest, read, save
from persona.config import Trait
from persona.parse import parse

NEW = {"tang":"榛果四十","yu":"杏核五十"}
OLD = {"tang":"松针十二","yu":"苇叶十三"}

def audit(observed):
    calls, turns, learned, blind, key = [read(observed/f"{n}.json")
                                         for n in ("calls","turns","learned","blind","key")]
    assert all(v["status"]=="ok" and v["request_hash"]==digest(v["messages"]) for v in calls.values())
    assert len({v["response"]["id"] for v in calls.values()}) == len(calls)
    assert read(observed/"design_snapshot.json")["design"] == digest(D)
    assert read(observed/"design_snapshot.json")["world"] == digest(W)
    assert read(observed/"design_snapshot.json")["runner"] == hashlib.sha256((HERE/"run.py").read_bytes()).hexdigest()
    for k,row in turns.items():
        assert parse(calls[k]["raw"]).say.strip() == row["say"].strip()
    assert set(blind) == set(key) and len(set(key.values())) == len(blind)
    assert {key[k] for k in blind} == {k for k in turns if k.split(".")[1] in {"lamp","byline"}}
    for k,v in blind.items():
        assert v["say"] == turns[key[k]]["say"]
    result = {"calls":len(calls),"foreground":len(turns),"protocol_valid":sum(
        v["validity"]["valid"] for v in turns.values()),"memory_calls":len(calls)-len(turns),
        "models":dict(Counter(c["response"]["model"] for c in calls.values())),
        "usage":{f:sum(c["response"]["usage"][f] for c in calls.values())
                 for f in ("prompt_tokens","completion_tokens","total_tokens")},
        "paired_requests":0,"facts":{},"actors":{}}
    coding = read(observed/"coding.json")
    assert set(coding)==set(blind)
    for b,v in coding.items():
        assert v["score"] in (0,1,2) and type(v["error"]) is bool and v["reason"]
    for a in OTHER:
        t = Trait(anchor=BASE[a],plasticity=D["plasticity"],drift=D["drift"])
        votes=[]
        for i in range(2):
            f = read(observed/f"feedback_{i+1}.json")[a]
            cue,reply = (turns[f"{a}.{i}.{r}"] for r in ("cue","reply"))
            assert reply["user"] == f["reply"] + D["training"][a][i]["receipt"]
            if f["eligible"] and cue["validity"]["valid"] and reply["validity"]["valid"]:
                assert f["quote"] and f["quote"] in cue["say"]
                votes.append([cue["turn_id"],reply["turn_id"],f["quote"],f["reply"]])
        assert learned[a]["eligible"] == votes
        if len(votes)*D["weight"]>=1.4:
            t.nudge(1)
        assert abs(t.value - learned[a]["after"]) < 1e-5
        val = float(learned[a]["after"])
        rows = result["actors"][a] = {"before":BASE[a],"after":val,"eligible":len(votes),"scenes":{}}
        for p in D["probes"]:
            cells=rows["scenes"][p["id"]] = {}
            for j in range(D["repeats"]):
                prefix=f"{a}.{p['id']}.{j}."
                fixed=calls[prefix+"fixed"]["messages"]
                if val != BASE[a]:
                    changed=calls[prefix+"changed"]["messages"]
                    src=f"{TRAIT}={BASE[a]:.4f}"
                    dst=f"{TRAIT}={val:.4f}"
                    assert fixed[0]["content"].count(src)==changed[0]["content"].count(dst)==1
                    assert src not in changed[0]["content"] and dst not in fixed[0]["content"]
                    assert fixed[0]["content"].replace(src,dst)==changed[0]["content"]
                    assert fixed[1:]==changed[1:]
                    result["paired_requests"]+=1
                for arm in ("fixed","changed"):
                    k=prefix+arm
                    if k in turns:
                        b=next(i for i,v in key.items() if v==k)
                        cells.setdefault(arm,[]).append(dict(blind=b,valid=turns[k]["validity"]["valid"],**coding[b]))
        facts = result["facts"][a]={}
        for arm in ("empty","memory"):
            k=f"{a}.fact.{arm}"
            msgs=calls[k]["messages"]
            r=turns[k]
            assert len(msgs)==2 and NEW[a] not in msgs[-1]["content"] and OLD[a] not in msgs[-1]["content"]
            facts[arm]={"new_in_system":NEW[a] in msgs[0]["content"],"old_in_system":OLD[a] in msgs[0]["content"],
                        "delivered":r["say"],"valid":r["validity"]["valid"],
                        "exact_new_only":NEW[a] in r["say"] and OLD[a] not in r["say"]}
    pairs = []
    for a,v in result["actors"].items():
        for scene,cells in v["scenes"].items():
            for f,g in zip(cells["fixed"],cells.get("changed",())):
                pairs.append((a,scene,f,g))
    def contrast(items):
        diff=[g["score"]-f["score"] for _,_,f,g in items]
        return {"pairs":len(diff),"higher":sum(x>0 for x in diff),"equal":diff.count(0),
                "lower":sum(x<0 for x in diff)}
    result["visible_pairs"]=contrast(pairs)
    result["valid_pairs"]=contrast([p for p in pairs if p[2]["valid"] and p[3]["valid"]])
    result["behavior_histograms"]={
        name:[sum(pair[idx]["score"]==q for pair in pairs) for q in (0,1,2)]
        for name,idx in (("fixed",2),("changed",3))}
    return result

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--write",action="store_true")
    args=ap.parse_args()
    obs=HERE/"observed"
    result=audit(obs)
    if args.write:
        save(obs/"summary.json",result)
    else:
        assert result==read(obs/"summary.json")
    print(f'{result["calls"]} provider calls; {result["paired_requests"]} matched pairs; '
          f'{result["protocol_valid"]}/{result["foreground"]} protocol valid.')
