"""Deterministic multi-turn comparison; uses only frozen sources, never a model."""
from dataclasses import asdict
import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation():
    for path, digest in load(HERE / "protocol.json")["hashes"].items():
        if sha(HERE / path) != digest:
            raise ValueError("Changed frozen input: " + path)
    package = HERE / "implementation_snapshot/persona"
    alias = "_composed_expression_study"
    spec = importlib.util.spec_from_file_location(alias, package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec); sys.modules[alias] = module; spec.loader.exec_module(module)
    return {key: importlib.import_module(alias + "." + key) for key in ("policy", "policy_session", "expression", "config", "store")}


def run_sequence(m, case, profile, method, root, *, restart):
    cfg = m["config"].PersonaConfig.load(HERE / "persona_config.toml")
    cfg.role.identity = ""; cfg.role.voice = "本研究不使用自由表达模型"
    policy = m["policy"].PolicyConfig(intervention=0, practical=0, feeling=1, detail=1, seed="composed-study-v1")
    attempted = []
    def forbidden(messages):
        attempted.append(1)
        raise AssertionError("No model call is permitted in this deterministic study")
    def session():
        return m["policy_session"].PolicySession(cfg, m["store"].Store(root), forbidden,
            policy=policy, expression=m["expression"].ExpressionConfig(**profile["expression"]),
            realization=method, clock=lambda: "2026-11-07T21:00:00+08:00")
    current = session()
    current.boundary("deny", user="以后不要给建议", quote="以后不要给建议", topic=case["topic"])
    for item in case["commitments"]:
        current.commitment("confirm", id=item["id"], text=item["text"], user=item["text"], quote=item["text"], topic=case["topic"])
    outputs = []
    for number, item in enumerate(case["turns"], 1):
        if restart and number == 4:
            current.close(); current = session()
        frame = m["policy"].ContentFrame.from_json(json.dumps(item["frame"], ensure_ascii=False))
        sources = tuple(m["policy"].Source(**s) for s in item["sources"])
        chunks = []
        result = current.say(item["user"], frame=frame, sources=sources, topic=case["topic"], on_delta=chunks.append)
        plan = result.expression_plan
        if plan is None:
            signature = result.say
            for focus in frame.focuses:
                signature = signature.replace(json.dumps(focus.evidence.text, ensure_ascii=False), "<quote>")
            for commitment in case["commitments"]:
                signature = signature.replace(json.dumps(commitment["text"], ensure_ascii=False), "<commitment>")
        else:
            signature = "".join(p["text"] if p["kind"] == "atom" else "<" + p["kind"] + ">" for p in plan["parts"])
        outputs.append(dict(turn=number, user=item["user"], sources=item["sources"], frame=asdict(frame),
            say=result.say, decision=asdict(result.decision), expression_plan=plan, signature=signature,
            realization=result.realization, semantic_execution=result.semantic_execution,
            delivery_match="".join(chunks) == result.say, calls_count=result.calls_count,
            state=asdict(current.state)))
    current.close()
    if attempted:
        raise AssertionError("Unexpected model request")
    return outputs, current.store.read_ledger()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    if out.exists():
        raise SystemExit("Use a new output directory; published evidence is never overwritten")
    m = implementation(); out.mkdir(parents=True)
    records = []; ledgers = {}; checks = []
    with tempfile.TemporaryDirectory(prefix="composed-study-") as temp:
        for case in load(HERE / "cases.json"):
            for profile in load(HERE / "profiles.json"):
                for method in ("bounded", "composed"):
                    id = "/".join((case["id"], profile["id"], method))
                    rows, ledger = run_sequence(m, case, profile, method, Path(temp) / id / "restart", restart=True)
                    continuous, _ = run_sequence(m, case, profile, method, Path(temp) / id / "continuous", restart=False)
                    if rows != continuous:
                        raise AssertionError("Restart changed output: " + id)
                    records.append(dict(id=id, case_id=case["id"], profile=profile["id"], expression=profile["expression"],
                        method=method, turns=rows, restart_equal=True))
                    ledgers[id] = ledger
                    checks.append(dict(id=id, restarted_equals_continuous=True))
    paired = {}
    for row in records:
        key = (row["case_id"], row["profile"])
        causal = [{k:t[k] for k in ("user", "sources", "frame", "decision", "state")} for t in row["turns"]]
        if key in paired and paired[key] != causal:
            raise AssertionError("Unmatched inputs, decisions or states")
        paired[key] = causal
    order = list(records); random.Random(2701107).shuffle(order)
    inputs = []; key = {}
    cases = {c["id"]:c for c in load(HERE / "cases.json")}
    for number, row in enumerate(order, 1):
        id = f"D{number:02d}"; key[id] = row["id"]
        inputs.append(dict(id=id, confirmed_boundary="已确认本主题不主动给建议",
            confirmed_commitments=[c["text"] for c in cases[row["case_id"]]["commitments"]],
            turns=[{k:t[k] for k in ("turn", "user", "sources", "say")} for t in row["turns"]]))
    save(out / "results.json", dict(protocol_sha256=sha(HERE / "protocol.json"),
        primary_outputs=108, verification_outputs=108, api_calls=0, records=records, checks=checks))
    save(out / "review_inputs.json", inputs); save(out / "review_key.json", key)
    (out / "runtime_ledgers.json.gz").write_bytes(gzip.compress(json.dumps(ledgers, ensure_ascii=False).encode(), mtime=0))
    print("18 six-turn dialogues; 108 primary outputs, 108 verification outputs, zero model calls")


if __name__ == "__main__":
    main()
