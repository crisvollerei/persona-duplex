"""Read ONE caller-selected replica ledger; emit counts only. Never opens Store or sends traffic."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def history_shape(entry):
    inp=entry.get("input")
    hist=inp.get("history") if isinstance(inp,dict) else None
    if not isinstance(hist,list):return "unobserved"
    if not hist:return "empty"
    messages=[m.get("content") for m in hist if isinstance(m,dict) and m.get("role")=="assistant"]
    if not messages or not all(isinstance(s,str) for s in messages):return "unobserved"
    # These are signatures in stored history, not proof of historical semantic/protocol validity.
    matched=[s.startswith("<<<HEAD>>>\n") and "\n<<<SAY>>>\n" in s and "\n<<<TAIL>>>\n" in s for s in messages]
    return "three_markers" if all(matched) else "mixed" if any(matched) else "no_full_signature"


def report(path):
    turns=defaultdict(Counter)
    events=Counter()
    with path.open("rb") as f:
        for line in f:
            if not line.strip():continue
            try:
                root=json.loads(line)
            except (UnicodeError,ValueError):
                events["unreadable_rows"]+=1;continue
            if not isinstance(root,dict):
                events["unreadable_rows"]+=1;continue
            rows=root.get("events") if root.get("kind")=="commit" else [root]
            if not isinstance(rows,list):
                events["unreadable_rows"]+=1;continue
            for r in rows:
                if not isinstance(r,dict):
                    events["unreadable_events"]+=1;continue
                k=r.get("kind")
                if k in ("model_call_start","model_call_end","run_start","memory_recall"):
                    events[k]+=1
                if k=="model_call_end":
                    resp=r.get("response")
                    usage=resp.get("usage") if isinstance(resp,dict) else None
                    n=usage.get("total_tokens") if isinstance(usage,dict) else None
                    if type(n) is int and n>=0:
                        events["tokens_known"]+=n;events["calls_with_usage"]+=1
                    else:events["calls_without_usage"]+=1
                if k!="turn":continue
                route=r.get("generation_path")
                if route not in ("full","compact","direct"):route="unobserved"
                protocol=r.get("protocol")
                if protocol not in ("text","tool","persona","auto"):protocol="unobserved"
                cell=turns[(protocol,route,history_shape(r))]
                cell["committed_turns"]+=1
                say=r.get("say")
                cell["nonempty_say" if isinstance(say,str) and say.strip() else
                     "empty_say" if isinstance(say,str) else "say_unobserved"]+=1
                validity=r.get("validity")
                valid=validity.get("valid") if isinstance(validity,dict) else None
                cell["protocol_valid" if valid is True else "protocol_invalid" if valid is False
                     else "validity_unobserved"]+=1
    return dict(schema=1,counts=dict(events),groups=[dict(protocol=p,path=r,recent_history=h,**counts)
                for (p,r,h),counts in sorted(turns.items())],
                exact_delivery="not_assessed_without_authorized_reference",
                preference="not_measured")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replica",type=Path)
    args=parser.parse_args()
    print(json.dumps(report(args.replica),ensure_ascii=False,indent=2))
