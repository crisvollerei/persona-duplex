"""No-network replay verifies exact requests and resulting final replies."""
import gzip
import json
from benchmark import HERE, load, save, verify, body_for, trial

class ArchiveClient:
    def __init__(self, rows):
        self.calls = {r["id"]: r for r in rows}; self.used = set()
        assert len(self.calls) == len(rows)
    def call(self, id, stage, messages):
        row = self.calls[id]
        assert row["stage"] == stage and row["request"] == body_for(stage, messages), id
        self.used.add(id); return row

def main():
    verify(); data = load(HERE / "run/results.json")
    client = ArchiveClient(json.loads(gzip.decompress((HERE / "run/raw_calls.json.gz").read_bytes())))
    cases = {c["id"]: c for c in load(HERE / "cases.json")}; profiles = {p["id"]: p for p in load(HERE / "profiles.json")}
    for row in data["records"]:
        assert trial(cases[row["case"]], profiles[row["profile"]], row["method"], client) == row, row["id"]
    assert client.used == set(client.calls)
    report = {"passed": True, "network_calls": 0, "new_samples": 0, "requests_matched": len(client.used), "routes_replayed": len(data["records"])}
    save(HERE / "replay_report.json", report); print(json.dumps(report))

if __name__ == "__main__": main()
