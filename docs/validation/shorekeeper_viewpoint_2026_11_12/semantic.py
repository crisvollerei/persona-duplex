"""Prepare anonymous SAY-only inputs and compare two independent qualitative codings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
OUT=HERE/"observed"
FIELDS={"knowledge":"insufficient claimed_known withheld unspecified mixed",
        "identity":"none correct incorrect hint_only mixed",
        "pact":"active released historical unmentioned ambiguous",
        "shared":"recognized denied unmentioned ambiguous"}


def read(p): return json.loads(p.read_bytes())


def save(path, data, check):
    if check or path.exists():
        if read(path)!=data: raise ValueError("Changed artifact: "+path.name)
    else:
        path.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+"\n").encode())


def main(phase, check):
    archive=read(OUT/"turns.json"); scenes=read(HERE/"scenes.json")
    vs={v["id"]:v for v in scenes["variants"]}
    order=[e for e in archive if e["status"]=="delivered" and e["say"].strip()]
    random.Random(390241).shuffle(order)
    packet=[]; key={}
    for n,row in enumerate(order,1):
        ident=f"K{n:02d}";act=row["act"];v=vs[row["variant"]]
        packet.append(dict(id=ident,act=act,question=scenes["question"],reference_name=v["source"],
                           authorized_facts=[{k:e[k] for k in ("shorekeeper","traveler")} for e in v["events"][:act]],
                           say=row["say"]))
        key[ident]={k:row[k] for k in ("variant","condition","replicate","act")}
    save(OUT/"blind_semantic.json",packet,check or phase=="score")
    save(OUT/"semantic_key.json",key,check or phase=="score")
    if phase=="prepare": return
    from metrics import analyze
    literal={(r["variant"],r["condition"],r["replicate"],r["act"]):r["source_or_alias_hit"]
             for r in analyze(scenes,archive)["rows"]}
    coders={}; by_cell={}; contaminated={}
    text={e["id"]:e["say"] for e in packet}
    for label in ("C1","C2"):
        ratings=read(OUT/f"coding_{label}.json")
        rows={e["id"]:e for e in ratings}
        assert len(rows)==len(ratings) and set(rows)==set(key)
        for ident,e in rows.items():
            assert all(e[k] in opts.split() for k,opts in FIELDS.items())
            assert e["quote"].strip() and e["quote"] in text[ident],(label,ident)
        coders[label]=rows
        for method in ("scoped","full_notes"):
            for stage in (1,2,3):
                for rep in ("A","B","all"):
                    group=[k for k,r in key.items() if r["condition"]==method and r["act"]==stage and rep in ("all",r["replicate"])]
                    by_cell.setdefault(f"{method}.{stage}.{rep}",{})[label]=dict(visible=len(group),
                        fields={field:dict(sorted(Counter(rows[i][field] for i in group).items())) for field in FIELDS})
        # First/second-act visible leakage changes what the listener could know in later history.
        for k,item in key.items():
            cond=(item["variant"],item["condition"],item["replicate"],item["act"])
            if item["act"]<3 and (literal[cond] or rows[k]["identity"]=="correct"):
                contaminated.setdefault(label,[]).append(k)
    disagreement={k:[field for field in FIELDS if coders["C1"][k][field]!=coders["C2"][k][field]]
                  for k in key}
    tally={f:sum(f not in diffs for diffs in disagreement.values()) for f in FIELDS}
    result=dict(coding_type="two separate method-blind model-assisted readings; not human ratings",
                source_packet_sha256=hashlib.sha256((OUT/"blind_semantic.json").read_bytes()).hexdigest(),
                visible=len(key),planned=len(archive),by_cell=by_cell,
                agreement_of_visible=tally,disagreements={k:v for k,v in disagreement.items() if v},
                pre_public_name_or_confirmed_identity={c:contaminated.get(c,[]) for c in coders})
    save(OUT/"semantic_summary.json",result,check)
    print("coded",len(key),"; agreement",tally)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","score"));p.add_argument("--check",action="store_true")
    args=p.parse_args();main(args.phase,args.check)
