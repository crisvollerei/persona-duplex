"""Paired prompt descriptions at frozen shared states, not new dialogue turns."""
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
import random
import sys
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_inquiry_benchmark_helpers",HERE.parent/"benchmark.py")
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise SystemExit("Use a new output folder")
    protocol=b.load(HERE/"protocol.json")
    for name,digest in protocol["hashes"].items():
        if b.sha(HERE/name)!=digest:raise ValueError("Changed paired-probe input")
    m=b.implementation();args.out.mkdir(parents=True)
    if not os.environ.get("PERSONA_API_KEY"):os.environ["PERSONA_API_KEY"]=getpass.getpass("Temporary API key (not saved): ")
    inputs=b.load(HERE/"inputs.json");jobs=[(item,method) for item in inputs for method in ("original","clarified")]
    random.Random(111072).shuffle(jobs);budget=b.Budget()
    def trial(item,method):
        id=item["id"]+"-"+method
        prompt=(HERE.parent/"selector_prompt.txt" if method=="original" else HERE/"clarified_prompt.txt").read_text(encoding="utf-8")
        messages=[dict(role="system",content=prompt),dict(role="user",content=json.dumps(item["payload"],ensure_ascii=False))]
        row=dict(id=id,state_id=item["id"],method=method,expected_id=item["expected_id"],eligible_ids=item["eligible_ids"],
                 messages=messages,started_at=datetime.now(timezone.utc).isoformat(),status="pending")
        budget.reserve();fn=m["llm"].openai_compatible(b.MODEL,base_url=b.ENDPOINT,temperature=.3,max_tokens=128,timeout=60)
        row["settings"]=deepcopy(fn.metadata);b.save(args.out/"calls"/(id+".json"),row);start=time.monotonic()
        try:
            row["output"]=fn(messages);row["status"]="ok"
        except Exception as exc:
            with budget.lock:budget.errors+=1
            row.update(status="error",error_type=type(exc).__name__)
        finally:
            row.update(elapsed_s=time.monotonic()-start,response=deepcopy(fn.observations.response))
        if row["status"]=="ok":
            try:
                chosen=b.parse_choice(m,row["output"]).slot_id
                row.update(selected_id=chosen,valid_schema=True,legal=chosen is None or chosen in item["eligible_ids"],matches=chosen==item["expected_id"])
            except Exception as exc:row.update(valid_schema=False,legal=False,matches=False,parse_error=type(exc).__name__)
        b.save(args.out/"calls"/(id+".json"),row)
        return row
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(trial,*job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%12==0:print(f"Completed {len(records)}/72 decision probes",flush=True)
    records.sort(key=lambda r:r["id"])
    b.save(args.out/"results.json",dict(protocol_sha256=b.sha(HERE/"protocol.json"),calls_count=budget.calls,
        records=[{k:v for k,v in r.items() if k not in ("messages","response","settings","output")} for r in records]))
    (args.out/"raw_calls.json.gz").write_bytes(gzip.compress(json.dumps(records,ensure_ascii=False).encode(),mtime=0))
    print(json.dumps({method:dict(n=36,legal=sum(r.get("legal",False) for r in records if r["method"]==method),
         matches=sum(r.get("matches",False) for r in records if r["method"]==method)) for method in ("original","clarified")}))


if __name__=="__main__":main()
