"""Persona-visible pilot; experimental routes, not the default Session pipeline."""
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
from selection import parse_pool, random_select
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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(s, encoding="utf-8", newline="\n"); os.replace(temp, path)

def verify():
    p = load(HERE / "protocol.json")
    for name, digest in p["hashes"].items():
        if sha(HERE / name) != digest: raise ValueError("frozen input changed: " + name)
    return p

def context(case): return {k: case[k] for k in ("history", "user", "facts", "boundaries")}
def generation_messages(case, profile, method):
    system = (HERE / "common_prompt.txt").read_text(encoding="utf-8") + "\n" + (HERE / (method + "_prompt.txt")).read_text(encoding="utf-8")
    payload = {"context": context(case), "persona": profile}
    if method == "coverage": payload["coverage_plan"] = coverage_plan(case)
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

def label_messages(case, pool_id, candidates):
    hidden, mapping = blind_candidates(pool_id, candidates)
    payload = {"context": context(case), "candidates": hidden}
    return ([{"role": "system", "content": (HERE / "label_prompt.txt").read_text(encoding="utf-8")},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], hidden, mapping)

def body_for(stage, messages):
    return {"model": MODEL, "messages": messages, "temperature": .7,
            "max_tokens": {"direct": 600, "pool": 2400, "labels": 3000}[stage], "stream": False}

class Client:
    def __init__(self, out):
        self.out = out; self.calls = 0; self.errors = 0; self.lock = threading.Lock()

    def call(self, id, stage, messages):
        with self.lock:
            if self.calls >= 36 or self.errors >= 3: return {"id": id, "status": "not_called", "reason": "budget_or_service_stop"}
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



def trial(case, profile, method, client):
    id = case["id"] + "-" + profile["id"] + "-" + method
    stage = "direct" if method == "direct" else "pool"
    raw = client.call(id, stage, generation_messages(case, profile, method))
    row = {"id": id, "case": case["id"], "profile": profile["id"], "method": method,
           "generation_call": raw["id"], "status": raw["status"], "say": None,
           "candidates": [], "labels": [], "label_call": None, "label_status": "not_attempted", "selected_id": None,
           "random_selected_id": None}
    if raw["status"] != "ok": return row
    if method == "direct":
        row["say"] = raw["output"].strip(); return row
    try: row["candidates"] = parse_pool(raw["output"])
    except (ValueError, TypeError, KeyError): row["status"] = "invalid_pool"; return row
    row["random_selected_id"] = random_select(row["candidates"], case["id"], profile["id"])
    msgs, hidden, mapping = label_messages(case, id, row["candidates"])
    tagged = client.call(id + "-labels", "labels", msgs)
    row.update(label_call=tagged["id"], label_status=tagged["status"], label_id_mapping=mapping)
    if tagged["status"] == "ok":
        try: row["labels"] = validate_annotations(tagged["output"], hidden, mapping)
        except (ValueError, TypeError, KeyError): row["label_status"] = "invalid_labels"
    if not row["labels"]: row["labels"] = unavailable_annotations(row["candidates"], row["label_status"])
    selected = choose(row["candidates"], row["labels"], profile)
    if selected: row.update(say=selected["text"], selected_id=selected["candidate_id"])
    else: row["status"] = "abstained"
    return row

def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--run", action="store_true"); a = p.parse_args()
    verify()
    if a.out.exists(): raise SystemExit("Use a new output directory")
    cases = load(HERE / "cases.json"); profiles = load(HERE / "profiles.json")
    jobs = [(c, r, m) for c in cases for r in profiles for m in ("direct", "coverage")]
    random.Random(2026110731).shuffle(jobs); a.out.mkdir(parents=True)
    save(a.out / "manifest.json", {"started_at": now(), "protocol_sha256": sha(HERE / "protocol.json"),
         "online": a.run, "planned_calls": 36, "python": platform.python_version(), "platform": platform.platform(),
         "server_seed": None, "retries": 0, "workers": 3})
    save(a.out / "execution_plan.json", [{"case": c["id"], "profile": r["id"], "method": m} for c,r,m in jobs])
    if not a.run: print("Prepared only; no requests."); return
    if not os.environ.get("PERSONA_API_KEY"): os.environ["PERSONA_API_KEY"] = getpass.getpass("Temporary API key (not saved): ")
    client = Client(a.out); rows = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(trial, c, r, m, client) for c,r,m in jobs]
        for future in as_completed(futures):
            row = future.result(); rows.append(row); save(a.out / "records" / (row["id"] + ".json"), row)
            print(f"Completed {len(rows)}/24 routes; requests {client.calls}", flush=True)
    rows.sort(key=lambda r: r["id"])
    calls = [load(f) for f in sorted((a.out / "calls").glob("*.json"))]
    raw = json.dumps(calls, ensure_ascii=False).encode()
    if os.environ["PERSONA_API_KEY"].encode() in raw: raise ValueError("credential in archive")
    (a.out / "raw_calls.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    save(a.out / "results.json", {"protocol_sha256": sha(HERE / "protocol.json"), "actual_calls": client.calls, "records": rows})
    print(json.dumps({"actual_calls": client.calls, "completed_outputs": sum(r["say"] is not None for r in rows)}, ensure_ascii=False))

if __name__ == "__main__": main()
