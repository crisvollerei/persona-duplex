"""Exact-request archived replay, without keys, network or new observations."""
import argparse
import gzip
import json
from pathlib import Path
from benchmark import HERE, load, save, verify, body_for, trial


class ArchiveClient:
    def __init__(self,calls):
        self.calls={c["id"]:c for c in calls};self.used=set()
        assert len(self.calls)==len(calls)
    def call(self,id,stage,messages):
        row=self.calls[id]
        if row["request"]!=body_for(stage,messages):raise AssertionError("Request mismatch: "+id)
        self.used.add(id);return row


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True,type=Path);args=p.parse_args()
    if args.out.exists():raise SystemExit("Use a new output directory")
    verify();data=load(HERE/"run/results.json")
    calls=json.loads(gzip.decompress((HERE/"run/raw_calls.json.gz").read_bytes()))
    client=ArchiveClient(calls)
    cases={c["id"]:c for c in load(HERE/"cases.json")};profiles=load(HERE/"profiles.json")
    for row in data["records"]:
        if trial(cases[row["case"]],row["generation"],client,profiles)!=row:raise AssertionError("Result mismatch: "+row["id"])
    assert client.used==set(client.calls)
    report={"passed":True,"network_calls":0,"new_samples":0,"pools_replayed":len(data["records"]),
            "requests_exactly_matched":len(client.used),"offline_choices_replayed":sum(len(r["selections"]) for r in data["records"])}
    save(args.out/"replay_report.json",report);print(json.dumps(report))


if __name__=="__main__":main()
