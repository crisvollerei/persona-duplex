"""Run once before online sampling. Frozen protocol is not rewritten."""
import subprocess
from benchmark import HERE, load, save, sha, now

def main():
    if (HERE / "protocol.json").exists(): raise SystemExit("Already frozen")
    repo = HERE.parents[2]
    source = repo / "personas/shorekeeper.toml"
    (HERE / "shorekeeper_source.toml").write_bytes(source.read_bytes())
    profiles = load(HERE / "profiles.json"); cases = load(HERE / "cases.json")
    assert len(profiles) == 2 and len(cases) == 6 and len(load(HERE / "fixtures.json")) == 12
    assert profiles[0]["source"]["sha256"] == sha(source)
    historical = {}
    for study in ("candidate_choice_2026_11_07", "candidate_revision_2026_11_07"):
        folder = HERE.parent / study; manifest = load(folder / "artifact_manifest.json")
        for row in manifest["files"]: assert sha(folder / row["path"]) == row["sha256"], row["path"]
        historical[study] = {"manifest_sha256": sha(folder / "artifact_manifest.json"), "file_count": len(manifest["files"])}
    save(HERE / "historical_integrity.json", historical)
    names = ["protocol.md", "cases.json", "profiles.json", "rubric.md", "fixtures.json", "fixture_key.json",
        "common_prompt.txt", "direct_prompt.txt", "coverage_prompt.txt", "label_prompt.txt", "selection.py", "mechanism.py",
        "benchmark.py", "prepare_reviews.py", "test_mechanism.py", "freeze.py", "shorekeeper_source.toml", "historical_integrity.json"]
    save(HERE / "protocol.json", {"study_id": HERE.name, "rubric_version": "persona-context-logic-v1",
        "fixed_at": now(), "planned_calls": 36, "planned_final_replies": 24, "planned_candidate_texts": 72,
        "planned_calibration_items": 12, "seed": 2026110731, "hashes": {n: sha(HERE / n) for n in names},
        "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "repository_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)),
        "implementation_scope": "Only the hashed study scripts execute; default Session is not used."})
    print("Frozen", len(names), "inputs; historical artifacts unchanged")

if __name__ == "__main__": main()
