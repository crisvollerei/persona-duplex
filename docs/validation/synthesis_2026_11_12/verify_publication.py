"""Verify the published byte manifest and live paper links without network access."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import runpy

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PATTERN=re.compile(rb"sk-[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{24,}")


def main():
    manifest=json.loads((HERE/"artifact_manifest.json").read_bytes())
    for record in manifest["files"]:
        path=(ROOT/record["path"]).resolve()
        assert path.is_relative_to(ROOT),record["path"]
        data=path.read_bytes()
        assert len(data)==record["bytes"] and hashlib.sha256(data).hexdigest()==record["sha256"],record["path"]
        expanded=gzip.decompress(data) if path.suffix==".gz" else data
        assert not PATTERN.search(expanded),"Credential-like value in "+record["path"]
    new_papers=[r["path"] for r in manifest["files"] if Path(r["path"]).name=="paper.md"]
    assert new_papers==[str((HERE/"paper.md").relative_to(ROOT)).replace("\\","/")],"Expected one new main paper"
    for p in (ROOT/"README.md",ROOT/"docs/README.md",ROOT/"docs/research.md",HERE/"paper.md"):
        for target in re.findall(r'\[[^]\n]+\]\(([^)]+)\)',p.read_text(encoding="utf-8")):
            if target.startswith(("https:","http:","#","mailto:")):continue
            assert (p.parent/target.split("#")[0]).exists(),(str(p.relative_to(ROOT)),target)
    # Run the primary numerical check in-process so it has no CLI side effects.
    module=runpy.run_path(str(HERE/"reproduce.py"))
    assert module["compute"]()==json.loads((HERE/"tables.json").read_bytes()),"Changed paper tables"
    print(len(manifest["files"]),"publication files verified; main links and numerical tables OK.")


if __name__=="__main__":main()
