"""Replay every provider response through the registered delivery path, with no network."""
from pathlib import Path
import hashlib
import json
import tempfile

from analyze import summarize
from history_check import snapshots
from review import pairs
from shadow_report import report
from study import HERE, ROOT, PARAMS, digest, load, trial


def main():
    assert summarize()==load(HERE/"results.json")
    with tempfile.TemporaryDirectory() as td:
        dirs=snapshots(Path(td))
        for stage in ("short","long","history"):
            reg=load(HERE/f"{stage}_registration.json")
            for name,h in reg.get("source_sha256",reg.get("source_hashes",{})).items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
            rows=load(HERE/f"{stage}_calls.json")
            for j,r in zip(reg["jobs"],rows,strict=True):
                assert j["id"]==r["id"] and r["raw"] is not None
                source=dirs[j["arm"]] if stage=="history" else dirs["body"] if stage=="long" else None
                msgs,say,val=trial(j["user"],j.get("route","full"),lambda _:r["raw"],source)
                assert digest({**PARAMS,"messages":msgs})==j["request"]
                assert r["say"]==say and r["validity"]==val
        path=dirs["body"]/"ledger.jsonl"
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        stats=report(path)
        assert stats["counts"]["run_start"]==4
        assert stats["counts"]["model_call_end"]==stats["counts"]["calls_without_usage"]==32
        assert sum(x["committed_turns"] for x in stats["groups"])==32
        assert sum(x.get("nonempty_say",0) for x in stats["groups"])==32
        assert stats["exact_delivery"]=="not_assessed_without_authorized_reference"
        assert before==hashlib.sha256(path.read_bytes()).hexdigest()
        red=Path(td)/"broken.jsonl"
        secret="PRIVATE_TEXT_DO_NOT_EXPORT"
        events=[{"kind":"turn","say":secret,"protocol":secret,"input":{"history":[{"role":"assistant","content":secret}]}},
                {"kind":"model_call_end","response":{"usage":{"total_tokens":42}}}]
        red.write_bytes((json.dumps({"kind":"commit","events":events})+"\n{broken").encode())
        bad=report(red)
        assert bad["counts"]["unreadable_rows"]==1 and bad["counts"]["tokens_known"]==42
        assert secret not in json.dumps(bad)
    assert len(pairs())==6
    print("88 registered responses replayed; extraction, histories, metrics and read-only boundaries verified.")


if __name__=="__main__":
    main()
