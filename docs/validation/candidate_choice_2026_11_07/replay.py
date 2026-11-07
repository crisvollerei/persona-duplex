"""Replay archived requests only after exact request comparison; never networking."""
import argparse
import gzip
import json
from pathlib import Path
from benchmark import HERE, MODEL, load, save, verify, run_pools, run_output


class ArchiveClient:
    def __init__(self, calls):
        self.by_id = {c["id"]: c for c in calls}
        assert len(self.by_id) == len(calls)
        self.used = set()

    def call(self, call_id, stage, messages):
        raw = self.by_id[call_id]
        expected = {"model": MODEL, "messages": messages, "temperature": .7,
                    "max_tokens": 2400 if stage == "pools" else 128 if stage == "prompt" else 600,
                    "stream": False}
        if raw["request"] != expected:
            raise AssertionError("Archived request does not match: " + call_id)
        self.used.add(call_id)
        return raw


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); args = p.parse_args()
    if args.out.exists(): raise SystemExit("Use a new output directory")
    verify()
    cases = {r["id"]: r for r in load(HERE / "cases.json")}
    profiles = {r["id"]: r for r in load(HERE / "profiles.json")}
    reports = []
    pools = {}
    for phase in ("pools", "main"):
        folder = HERE / phase
        if not (folder / "results.json").exists(): continue
        data = load(folder / "results.json")
        client = ArchiveClient(json.loads(gzip.decompress((folder / "raw_calls.json.gz").read_bytes())))
        checked = 0
        for old in data["records"]:
            if old["status"] == "not_called":
                raise ValueError("Service-stop replay requires explicit archived skipped-call handling")
            if phase == "pools":
                new = run_pools(cases[old["case"]], client)
                pools[old["case"]] = new
            else:
                new = run_output((old["case"], old["profile"], old["method"]), cases, profiles, pools, client)
            if new != old: raise AssertionError("Replay mismatch: " + str(old.get("id", old["case"])))
            checked += 1
        if client.used != set(client.by_id): raise AssertionError("Unused archived requests")
        reports.append({"phase": phase, "reproduced_units": checked, "exact_requests_matched": len(client.used)})
    save(args.out / "replay_report.json", {"network_calls": 0, "new_independent_samples": 0,
                                           "passed": True, "stages": reports})
    print(json.dumps(reports))


if __name__ == "__main__": main()
