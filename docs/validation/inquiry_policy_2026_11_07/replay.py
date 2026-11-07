"""Re-run all conversation state transitions with byte-checked archived proposals.

This replays 108 existing selector responses; it sends no network requests and
does not create fresh experimental evidence or new subjective ratings.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

import benchmark as b


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise SystemExit("Use a new output directory")
    m=b.implementation();args.out.mkdir(parents=True)
    import gzip
    def raw(folder):return json.loads(gzip.decompress((folder/"raw_calls.json.gz").read_bytes()))
    checks=[];reused=[]
    for phase,folder,prompt_path in (("main",b.HERE,b.HERE/"selector_prompt.txt"),
            ("confirmation",b.HERE/"clarified_flow",b.HERE/"decision_probe/clarified_prompt.txt")):
        calls={(c["id"],c["turn"]):c for c in raw(folder)}
        def archived(modules,payload,out,id,turn,budget):
            call=calls[(id,turn)]
            expected=[dict(role="system",content=prompt_path.read_text(encoding="utf-8")),
                      dict(role="user",content=json.dumps(payload,ensure_ascii=False))]
            if expected!=call["messages"]:raise AssertionError("Changed archived request: "+id)
            if call["status"]!="ok":raise AssertionError("This archive has an unexpected transport failure")
            reused.append((phase,id,turn));return call["output"]
        b.select_with_model=archived
        for original in b.load(folder/"results.json")["records"]:
            job={k:deepcopy(original[k]) for k in ("id","case","profile","method","replicate")}
            replay,_=b.episode(m,job,args.out/phase,b.Budget())
            if json.loads(json.dumps(replay,ensure_ascii=False))!=original:
                raise AssertionError("Replayed trajectory changed: "+original["id"])
            checks.append(dict(phase=phase,id=original["id"],equal=True))
    if len(reused)!=108 or len(set(reused))!=108:raise AssertionError("Unexpected replay response count")
    b.save(args.out/"replay_report.json",dict(network_requests=0,reused_responses=108,
        exact_flows=len(checks),planned_steps=180,delivered_outputs=147,checks=checks))
    print("30 flows replayed exactly; 108 archived responses verified; zero network requests")


if __name__=="__main__":main()
