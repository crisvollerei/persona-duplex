"""Read-only independent audit of model-facing material and provenance; no human files read."""
import ast
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / "arrival_shadow"
OUT = HERE / "observed"


def load(path):
    return json.loads(path.read_bytes())


def archive(path):
    with gzip.open(path, "rb") as src:
        return [json.loads(line) for line in src]


def digest(obj):
    return sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixed_literal(path, name):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def audit():
    design, reg = load(HERE / "design.json"), load(OUT / "registration.json")
    assert reg["model"] == design["model"]
    for path, checksum in reg["sources"].items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == checksum, path
    new = {c["id"]: c for c in load(HERE / "cases.json")}
    old = {c["id"]: c for c in load(OLD / "cases.json")}
    previous = {r["id"]: r for r in archive(OLD / "observed/calls.jsonl.gz")}
    typed = fixed_literal(HERE / "method.py", "TYPED")
    # Inspect the delimited system template itself, not a live generator or model extraction.
    assert "沈遥" not in typed and "许原" not in typed and "P1" not in typed
    expected = {}
    for partition, dataset, stages, n in [
        ("predictions", new, ("A", "B"), 48), ("repeats", new, ("A2",), 24),
        ("calibration", old, ("A",), 32), ("controls", new, ("A",), 24)
    ]:
        predictions = load(OUT / (partition + ".json"))
        assert len(predictions) == n
        stage = {"calibration": "old", "controls": "control"}.get(partition, "new")
        for i in dataset:
            for m in ("legacy", "typed"):
                for order in stages:
                    key = ".".join((i, m, order))
                    expected[stage+"."+key] = predictions[key]
    entries = archive(OUT / "calls.jsonl.gz")
    calls = {r["id"]: r for r in entries}
    assert len(reg["order"]) == len(calls) == len(entries) == len(expected) == 128
    assert set(calls) == set(reg["order"]) == set(expected)
    ids, models, failures, quotes, accepted = set(), Counter(), Counter(), Counter(), Counter()
    for k, r in calls.items():
        stage, i, method, order = k.split(".")
        assert r["params"] == reg["model"]
        person_order = ["俞川","常鸣"] if stage == "old" else list(design["people"])
        if order == "B":
            person_order.reverse()
        turns = list((old if stage == "old" else new)[i]["turns"])
        if stage == "control":
            turns.append({"speaker": "现场核验员", "text": design["updates"][(int(i)-1)%3]["text"]})
        location = "展厅内的签收台" if stage == "old" else design["location"]
        msg = r["messages"]
        # Equality with a reconstructed JSON user payload ensures neither human labels, expected
        # control states, case numbers nor pair IDs were appended to actual prompts.
        assert len(msg) == 2 and set(msg[0]) == {"role","content"} and msg[0]["role"] == "system", k
        assert msg[1] == {"role": "user", "content": json.dumps(
            {"location": location, "people": person_order, "dialogue": [
                {"turn": j, "speaker": t["speaker"], "text": t["text"]} for j,t in enumerate(turns,1)
            ]}, ensure_ascii=False)}, k
        assert msg[0]["content"] == (typed if method == "typed" else previous["01.evidence.YC"]["messages"][0]["content"]), k
        other = "legacy" if method == "typed" else "typed"
        assert msg[1] == calls[f"{stage}.{i}.{other}.{order}"]["messages"][1], k
        if order == "A2":
            assert msg == calls[f"{stage}.{i}.{method}.A"]["messages"], k
        if stage == "old" and method == "legacy":
            assert msg == previous[f"{i}.evidence.YC"]["messages"], k

        pred = expected[k]
        assert digest(msg) == pred["request"] and set(pred["people"]) == set(person_order), k
        if r["ok"]:
            receipt = r["response"]; rid = receipt["id"]
            assert rid and rid not in ids; ids.add(rid)
            models[receipt["model"]] += 1
        else:
            failures[r["error"]] += 1
            assert not r["raw"] and all(z["state"] is None for z in pred["people"].values()), k
        good = {p:z for p,z in pred["people"].items() if z["reason"] == "accepted"}
        if good:
            raw = {item["person"]:item for item in json.loads(r["raw"])["people"]}
            assert set(raw) == set(person_order)
        for p,z in good.items():
            out = raw[p]
            assert out["state"] == z["state"] and z["state"] in ("present","future","unknown"), k
            assert (out["basis"] if method == "typed" else None) == z["basis"], k
            if method == "typed":
                assert z["basis"] in ("explicit","pragmatic","none"), k
            assert out["evidence"] == z["evidence"] and 0 <= len(z["evidence"]) <= 3, k
            if z["state"] != "unknown" or z["basis"] in ("explicit","pragmatic"):
                assert z["evidence"], k
            for ev in z["evidence"]:
                n, q = ev["turn"], ev["quote"]
                assert type(n) is int and 1 <= n <= len(turns) and isinstance(q,str) and q.strip() and q in turns[n-1]["text"], k
                quotes[method] += 1
            accepted[method] += 1
    return {"slots":len(entries), "unique_successful_provider_ids":len(ids), "models":dict(models),
            "failed_calls":dict(failures),"accepted_people":dict(accepted),"verified_quotes":dict(quotes),
            "matched_method_user_pairs":64,"identical_repeat_pairs":24,"identical_prior_requests":16,
            "registered_file_hashes":len(reg["sources"])}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
