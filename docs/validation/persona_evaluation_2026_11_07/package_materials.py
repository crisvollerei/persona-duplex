"""Create a reproducible delivery manifest after final material checks."""
import re
from urllib.parse import unquote
from benchmark import HERE, sha, load, save

def main():
    assert load(HERE / "audit_report.json")["passed"]
    assert load(HERE / "analysis_audit_report.json")["status"] == "passed"
    assert load(HERE / "replay_report.json")["passed"]
    links = 0
    for path in HERE.rglob("*.md"):
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"): continue
            rel = unquote(target.split("#",1)[0].strip("<>"))
            destination = (path.parent / rel).resolve()
            assert destination.exists() or destination == (HERE / "artifact_manifest.json").resolve(), (path.name, target)
            links += 1
    rows = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json" or "__pycache__" in path.parts: continue
        assert path.suffix not in {".tmp", ".pyc"}, path
        rows.append({"path": path.relative_to(HERE).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    value = {"study_id": HERE.name, "file_count": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
             "local_markdown_links_checked": links, "files": rows}
    save(HERE / "artifact_manifest.json", value)
    print({k:v for k,v in value.items() if k != "files"})

if __name__ == "__main__": main()
