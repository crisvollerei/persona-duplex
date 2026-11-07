"""Frozen 2x2 component study. One pool plus one blind labeling call, no retries."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import threading
import time
import urllib.error
import urllib.request
from selection import parse_pool
from mechanism import coverage_plan, blind_candidates, validate_annotations, unavailable_annotations, choose

HERE = Path(__file__).resolve().parent
MODEL = "deepseek-chat"
ENDPOINT = "https://api.deepseek.com/v1/chat/completions"


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()


def save(path, value):
    s = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    key = os.environ.get("PERSONA_API_KEY")
    if key and key in s: raise ValueError("credential in output")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp"); temp.write_text(s, encoding="utf-8", newline="\n"); os.replace(temp, path)


def verify():
    p = load(HERE / "protocol.json")
    for name, digest in p["hashes"].items():
        if sha(HERE / name) != digest: raise ValueError("frozen input changed: " + name)
    return p


def context(case): return {k: case[k] for k in ("history", "user", "facts", "boundaries")}


def pool_messages(case, generation):
    suffix = "baseline" if generation == "baseline" else "coverage"
    system = (HERE / "common_prompt.txt").read_text(encoding="utf-8") + "\n" + (HERE / (suffix + "_prompt.txt")).read_text(encoding="utf-8")
    payload = {"context": context(case)}
    if generation == "coverage": payload["coverage_plan"] = coverage_plan(case)
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def label_messages(case, pool_id, candidates):
    hidden, mapping = blind_candidates(pool_id, candidates)
    payload = {"context": context(case), "candidates": hidden}
    return ([{"role": "system", "content": (HERE / "label_prompt.txt").read_text(encoding="utf-8")},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], hidden, mapping)


def body_for(stage, messages):
    return {"model": MODEL, "messages": messages, "temperature": .7,
            "max_tokens": 2400 if stage == "pool" else 3000, "stream": False}


class Client:
    def __init__(self, out):
        self.out = out; self.calls = 0; self.errors = 0; self.lock = threading.Lock()

    def call(self, id, stage, messages):
        with self.lock:
            if self.calls >= 32 or self.errors >= 3: return {"id": id, "status": "not_called", "reason": "budget_or_service_stop"}
            self.calls += 1
        row = {"id": id, "stage": stage, "started_at": now(), "endpoint": ENDPOINT,
               "request": body_for(stage, messages), "status": "running", "timeout_s": 90}
        path = self.out / "calls" / (id + ".json"); save(path, row)
        req = urllib.request.Request(ENDPOINT, data=json.dumps(row["request"]).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["PERSONA_API_KEY"]}, method="POST")
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=90) as response: payload = json.loads(response.read().decode())
            row["response"] = payload; choice = payload["choices"][0]; output = choice["message"].get("content")
            if choice.get("finish_reason") != "stop" or not isinstance(output, str) or not output.strip(): raise ValueError("incomplete_generation")
            row.update(status="ok", output=output)
        except Exception as exc:
            with self.lock: self.errors += 1
            row.update(status="error", error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError): row["http_status"] = exc.code; exc.close()
        finally:
            row.update(elapsed_s=time.monotonic()-start, finished_at=now()); save(path, row)
        return row


def trial(case, generation, client, profiles):
    id = case["id"] + "-" + generation
    raw = client.call(id + "-pool", "pool", pool_messages(case, generation))
    row = {"id": id, "case": case["id"], "generation": generation, "pool_call": raw["id"],
           "status": raw["status"], "candidates": [], "label_call": None, "label_status": "not_attempted", "labels": [], "selections": []}
    if raw["status"] == "ok":
        try: row["candidates"] = parse_pool(raw["output"])
        except (ValueError, TypeError, KeyError): row["status"] = "invalid_pool"
    if row["status"] != "ok": return row
    msgs, hidden, mapping = label_messages(case, id, row["candidates"])
    tagged = client.call(id + "-labels", "labels", msgs)
    row.update(label_call=tagged["id"], label_status=tagged["status"], label_id_mapping=mapping)
    if tagged["status"] == "ok":
        try: row["labels"] = validate_annotations(tagged["output"], hidden, mapping)
        except (ValueError, TypeError, KeyError): row["label_status"] = "invalid_labels"
    if not row["labels"]: row["labels"] = unavailable_annotations(row["candidates"], row["label_status"])
    for source, labels in (("self", row["candidates"]), ("evidence", row["labels"])):
        for profile in profiles:
            selected = choose(row["candidates"], labels, profile)
            row["selections"].append({"label_source": source, "profile": profile["id"],
                "status": "ok" if selected else "abstained", "selected_id": selected["candidate_id"] if selected else None,
                "say": selected["text"] if selected else None})
    return row


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True, type=Path); p.add_argument("--run", action="store_true"); args = p.parse_args()
    verify()
    if args.out.exists(): raise SystemExit("Use a new output directory")
    cases = load(HERE / "cases.json"); profiles = load(HERE / "profiles.json")
    jobs = [(case, generation) for case in cases for generation in ("baseline", "coverage")]
    random.Random(2026110727).shuffle(jobs)
    args.out.mkdir(parents=True)
    save(args.out / "manifest.json", {"started_at": now(), "protocol_sha256": sha(HERE / "protocol.json"),
        "python": platform.python_version(), "platform": platform.platform(), "online": args.run, "planned_calls": 32,
        "server_seed": None, "retries": 0})
    save(args.out / "execution_plan.json", [{"case": c["id"], "generation": g} for c,g in jobs])
    if not args.run:
        print("Prepared only; no requests. Use a new output folder with --run to execute."); return
    if not os.environ.get("PERSONA_API_KEY"): os.environ["PERSONA_API_KEY"] = getpass.getpass("Temporary API key (not saved): ")
    client = Client(args.out); rows = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(trial, case, generation, client, profiles) for case,generation in jobs]
        for future in as_completed(futures):
            row = future.result(); rows.append(row); save(args.out / "records" / (row["id"] + ".json"), row)
            print(f"Completed {len(rows)}/16 pools; actual requests {client.calls}", flush=True)
    rows.sort(key=lambda r: r["id"])
    calls = [load(f) for f in sorted((args.out / "calls").glob("*.json"))]
    raw = json.dumps(calls, ensure_ascii=False).encode()
    if os.environ["PERSONA_API_KEY"].encode() in raw: raise ValueError("credential in archive")
    (args.out / "raw_calls.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    save(args.out / "results.json", {"protocol_sha256": sha(HERE / "protocol.json"), "planned_pools": 16,
         "actual_calls": client.calls, "records": rows})
    print(json.dumps({"calls": client.calls, "pools_ok": sum(r["status"] == "ok" for r in rows),
        "label_calls_ok": sum(r["label_status"] == "ok" for r in rows)}, ensure_ascii=False))


if __name__ == "__main__": main()
