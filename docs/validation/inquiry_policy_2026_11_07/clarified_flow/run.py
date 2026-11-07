"""One full-flow confirmation per task/profile, using the clarified selector."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import getpass
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_inquiry_confirmation",HERE.parent/"benchmark.py")
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)


def select(m,payload,out,id,turn,budget):
    budget.reserve()
    fn=m["llm"].openai_compatible(b.MODEL,base_url=b.ENDPOINT,temperature=.3,max_tokens=128,timeout=60)
    messages=[dict(role="system",content=(HERE.parent/"decision_probe/clarified_prompt.txt").read_text(encoding="utf-8")),
              dict(role="user",content=json.dumps(payload,ensure_ascii=False))]
    row=dict(id=id,turn=turn,messages=messages,settings=deepcopy(fn.metadata),status="running",
             started_at=datetime.now(timezone.utc).isoformat())
    path=out/"calls"/f"{id}-{turn}.json";b.save(path,row);start=time.monotonic()
    try:
        value=fn(messages);row.update(status="ok",output=value);return value
    except Exception as exc:
        with budget.lock:budget.errors+=1
        row.update(status="error",error_type=type(exc).__name__);raise
    finally:
        row.update(elapsed_s=time.monotonic()-start,response=deepcopy(fn.observations.response));b.save(path,row)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise SystemExit("Use a new output directory")
    for name,digest in b.load(HERE/"protocol.json")["hashes"].items():
        if b.sha(HERE/name)!=digest:raise ValueError("Changed confirmation input")
    m=b.implementation();args.out.mkdir(parents=True)
    if not os.environ.get("PERSONA_API_KEY"):os.environ["PERSONA_API_KEY"]=getpass.getpass("Temporary API key (not saved): ")
    b.select_with_model=select;budget=b.Budget();records=[];ledgers={}
    jobs=[dict(id=f"{case['id']}-{profile['id']}-clarified",case=case["id"],profile=profile["id"],method="prompt",replicate=1)
          for case in b.load(HERE.parent/"cases.json") for profile in b.load(HERE.parent/"profiles.json")]
    b.save(args.out/"execution_plan.json",jobs)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(b.episode,m,job,args.out,budget) for job in jobs]
        for future in as_completed(futures):
            row,ledger=future.result();records.append(row);ledgers[row["id"]]=ledger
            print(f"Completed {len(records)}/6 confirmation flows",flush=True)
    records.sort(key=lambda r:r["id"])
    b.save(args.out/"results.json",dict(protocol_sha256=b.sha(HERE/"protocol.json"),calls_count=budget.calls,planned_outputs=36,records=records))
    calls=[b.load(p) for p in sorted((args.out/"calls").glob("*.json"))]
    (args.out/"raw_calls.json.gz").write_bytes(gzip.compress(json.dumps(calls,ensure_ascii=False).encode(),mtime=0))
    (args.out/"runtime_ledgers.json.gz").write_bytes(gzip.compress(json.dumps(ledgers,ensure_ascii=False).encode(),mtime=0))
    print(json.dumps(dict(requests=budget.calls,delivered=sum(t["status"]=="ok" for r in records for t in r["turns"]),
                         known=sum(r["known_slots"] for r in records),complete=sum(r["complete"] for r in records))))


if __name__=="__main__":main()
