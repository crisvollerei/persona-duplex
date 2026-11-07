"""Create method-hidden, deduplicated text inputs; no model calls."""
import argparse
from pathlib import Path
import random
from benchmark import HERE, load, save, context


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=HERE);a=p.parse_args()
    data=a.data.resolve()
    if (data/"review_key.json").exists():raise SystemExit("Review inputs already exist; use a new data root")
    cases={c["id"]:c for c in load(HERE/"cases.json")}
    unique={}
    for row in load(data/"run/results.json")["records"]:
        for candidate in row["candidates"]:
            key=(row["case"],candidate["text"])
            if key not in unique:unique[key]={"case":row["case"],"text":candidate["text"],"references":[]}
            unique[key]["references"].append({"pool_id":row["id"],"candidate_id":candidate["id"]})
    rows=list(unique.values());random.Random(20261107271).shuffle(rows)
    inputs=[]
    for i,row in enumerate(rows,1):
        row["id"]=f"R{i:03}"
        inputs.append({"id":row["id"],**context(cases[row["case"]]),"say":row["text"]})
    save(data/"review_key.json",rows);save(data/"reviews/inputs.json",inputs)
    print(f"Prepared {len(rows)} unique scene-text records; generation and labels hidden")


if __name__=="__main__":main()
