"""Explicit opt-in online pilot; no retries, credentials or mutable core imports."""
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

from selection import parse_pool, select, random_select, parse_choice, distance

HERE = Path(__file__).resolve().parent
MODEL = "deepseek-chat"
ENDPOINT = "https://api.deepseek.com/v1/chat/completions"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    key = os.environ.get("PERSONA_API_KEY")
    if key and key in text:
        raise ValueError("credential in output")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def verify():
    protocol = load(HERE / "protocol.json")
    for name, digest in protocol["hashes"].items():
        if sha(HERE / name) != digest:
            raise ValueError("frozen file changed: " + name)
    return protocol


def context(case):
    return {k: case[k] for k in ("history", "user", "facts", "boundaries")}


def messages(stage, case, profile=None, pool=None):
    suffix = {"pools": "pool", "prompt": "selector", "direct": "direct"}[stage]
    system = (HERE / "common_prompt.txt").read_text(encoding="utf-8") + "\n" + (HERE / (suffix + "_prompt.txt")).read_text(encoding="utf-8")
    payload = {"context": context(case)}
    if profile is not None:
        payload["profile"] = profile
    if pool is not None:
        payload["candidates"] = pool
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


class Client:
    def __init__(self, out, limit):
        self.out, self.limit = out, limit
        self.calls, self.errors = 0, 0
        self.lock = threading.Lock()

    def call(self, call_id, stage, msgs):
        with self.lock:
            if self.calls >= self.limit or self.errors >= 3:
                return {"id": call_id, "status": "not_called", "error": "budget_or_service_stop"}
            self.calls += 1
        body = {"model": MODEL, "messages": msgs, "temperature": .7,
                "max_tokens": 2400 if stage == "pools" else 128 if stage == "prompt" else 600,
                "stream": False}
        row = {"id": call_id, "stage": stage, "started_at": now(), "endpoint": ENDPOINT,
               "request": body, "status": "running", "timeout_s": 90}
        path = self.out / "calls" / (call_id + ".json")
        save(path, row)
        req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["PERSONA_API_KEY"]}, method="POST")
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                payload = json.loads(response.read().decode())
            row["response"] = payload
            choice = payload["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("incomplete_generation")
            text = choice["message"].get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty_generation")
            row.update(status="ok", output=text)
        except Exception as exc:
            with self.lock:
                self.errors += 1
            row.update(status="error", error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                row["http_status"] = exc.code
                exc.close()
        finally:
            row.update(elapsed_s=time.monotonic() - start, finished_at=now())
            save(path, row)
        return row


def run_pools(case, client):
    raw = client.call(case["id"], "pools", messages("pools", case))
    row = {"case": case["id"], "call_id": raw["id"], "status": raw["status"], "candidates": []}
    if raw["status"] == "ok":
        try:
            row["candidates"] = parse_pool(raw["output"])
        except (ValueError, TypeError, KeyError):
            row["status"] = "invalid_pool"
    return row


def run_output(job, cases, profiles, pools, client):
    case_id, profile_id, method = job
    case, profile, pool_row = cases[case_id], profiles[profile_id], pools[case_id]
    pool = pool_row["candidates"]
    row = {"id": f"{case_id}-{profile_id}-{method}", "case": case_id, "profile": profile_id,
           "method": method, "status": "pending", "say": None, "selected_id": None, "call_id": None}
    if method != "direct" and pool_row["status"] != "ok":
        row["status"] = "pool_unavailable"
        return row
    if method in ("prompt", "direct"):
        raw = client.call(row["id"], method, messages(method, case, profile, pool if method == "prompt" else None))
        row.update(call_id=raw["id"], status=raw["status"])
        if raw["status"] != "ok":
            return row
        if method == "direct":
            row.update(say=raw["output"], status="ok")
            return row
        try:
            chosen = parse_choice(raw["output"], pool)
        except (ValueError, TypeError):
            row["status"] = "invalid_choice"
            return row
    elif method == "program":
        chosen = select(pool, profile)
    elif method == "random":
        chosen = random_select(pool, case_id, profile_id)
    else:
        raise ValueError("unknown method")
    if chosen is None:
        row["status"] = "abstained"
        return row
    candidate = next(c for c in pool if c["id"] == chosen)
    row.update(status="ok", say=candidate["text"], selected_id=chosen,
               self_label_distance=distance(candidate, profile))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pools", "main"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pools", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    protocol = verify()
    if args.out.exists():
        raise SystemExit("Use a new output directory")
    cases = {c["id"]: c for c in load(HERE / "cases.json")}
    profiles = {p["id"]: p for p in load(HERE / "profiles.json")}
    manifest = {"phase": args.phase, "created_at": now(), "protocol_sha256": sha(HERE / "protocol.json"),
                "python": platform.python_version(), "platform": platform.platform(), "online": args.run,
                "server_seed": None, "retries": 0}
    if args.phase == "main":
        if not args.pools or not args.gate:
            raise SystemExit("Main stage requires archived pools and gate result")
        pool_data, gate = load(args.pools), load(args.gate)
        if pool_data["protocol_sha256"] != manifest["protocol_sha256"] or gate["pools_sha256"] != sha(args.pools) or not gate["proceed"]:
            raise ValueError("incompatible pools or failed gate")
        pools = {r["case"]: r for r in pool_data["records"]}
        manifest.update(pools_sha256=sha(args.pools), gate_sha256=sha(args.gate))
        jobs = [(c, p, m) for c in cases for p in profiles for m in ("program", "prompt", "direct", "random")]
    else:
        jobs = list(cases.values())
    random.Random(2026110719).shuffle(jobs)
    args.out.mkdir(parents=True)
    save(args.out / "manifest.json", manifest)
    save(args.out / "execution_plan.json", jobs)
    if not args.run:
        print("Prepared only; no API calls. Use another new folder with --run to execute.")
        return
    if not os.environ.get("PERSONA_API_KEY"):
        os.environ["PERSONA_API_KEY"] = getpass.getpass("Temporary API key (not saved): ")
    client = Client(args.out, 12 if args.phase == "pools" else 72)
    records = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_pools, job, client) if args.phase == "pools" else
                   executor.submit(run_output, job, cases, profiles, pools, client) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            save(args.out / "records" / ((row.get("id") or row["case"]) + ".json"), row)
            if len(records) % 6 == 0:
                print(f"Completed {len(records)}/{len(jobs)} units; actual calls {client.calls}", flush=True)
    records.sort(key=lambda r: r.get("id") or r["case"])
    calls = [load(p) for p in sorted((args.out / "calls").glob("*.json"))]
    raw = json.dumps(calls, ensure_ascii=False, allow_nan=False).encode()
    if os.environ["PERSONA_API_KEY"].encode() in raw:
        raise ValueError("credential in raw data")
    (args.out / "raw_calls.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    save(args.out / "results.json", {"protocol_sha256": manifest["protocol_sha256"], "phase": args.phase,
        "calls_count": client.calls, "planned_units": len(jobs), "records": records})
    print(json.dumps({"phase": args.phase, "calls": client.calls, "units": len(records),
        "ok": sum(r["status"] == "ok" for r in records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
