"""Post-hoc matched history-layout check, frozen before these 20 requests."""
import argparse
from collections import Counter
from copy import deepcopy
import getpass
import hashlib
import os
from pathlib import Path
import random
import shutil
import tempfile

import trajectory
from study import HERE, ROOT, ENDPOINT, PARAMS, Session, SessionConfig, Store, config, digest, load, trial, write
from persona.llm import openai_compatible
from persona.protocol import visible_dialogue


def snapshots(td):
    body=td/"body"
    trajectory.build(body)
    native=td/"native"
    shutil.copytree(body,native)
    store=Store(native)
    runtime=store.load_view("runtime",{})
    old_history=runtime["history"]
    dialogue=visible_dialogue(old_history)
    assert len(dialogue)==12
    # Construct known-valid fixture metadata in an isolated Session; never infer real past HEADs.
    # Everything in direct/baseline is a duplicate of the already visible synthetic reply.
    def authored(_messages):
        return ("<<<HEAD>>>\nneed: 只是想说说话\nconfidence: 0.6\ndirect: "+line+
            "\nbaseline: "+line+"\ndisplace: none\n<<<SAY>>>\n"+line+
            "\n<<<TAIL>>>\nsettle: -\ndelta: -\npending: -")
    with Session(config(),Store(td/"format_fixture"),authored,
                 options=SessionConfig(memory_mode="frozen",consolidate_on_exit=False)) as s:
        for u,a in zip(dialogue[::2],dialogue[1::2]):
            line=a["content"]
            result=s.say(u["content"])
            assert result.validity["valid"] and result.say==line
        history=deepcopy(s.engine.history)
    assert visible_dialogue(history)==dialogue
    store.save_view("runtime",{**runtime,"history":history})
    return {"body":body,"native":native}


def design(dirs):
    reqs,jobs={},[]
    for q in load(HERE/"trajectory.json")["probes"]:
        histories=[]
        for arm,directory in dirs.items():
            msgs,_,_=trial(q["user"],"full",lambda _:"预备占位",directory)
            histories.append(msgs)
            h=digest({**PARAMS,"messages":msgs})
            reqs[h]={**PARAMS,"messages":msgs}
            for rep in (1,2):
                jobs.append(dict(id=f"{q['id']}-{arm}-{rep}",case=q["id"],user=q["user"],arm=arm,request=h))
        a,b=histories
        assert a[0]==b[0] and a[-1]==b[-1] and a[1:-1]!=b[1:-1]
        assert visible_dialogue(a[1:-1])==visible_dialogue(b[1:-1])
    random.Random(71931).shuffle(jobs)
    files=[HERE/n for n in ("study.py","trajectory.py","trajectory.json","history_check.py","long_preflight.json")]
    files+=sorted((ROOT/"persona").glob("*.py"))
    return dict(primary="complete full protocol",secondary="delivered text; exact-copy subset",
        scope="post-hoc: two layouts of the same six recent synthetic pairs; no true historical inner observations",
        endpoint=ENDPOINT,source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        requests=reqs,jobs=jobs)


def main(cmd):
    path=HERE/"history_registration.json";out=HERE/"history_calls.json"
    with tempfile.TemporaryDirectory() as temp:
        dirs=snapshots(Path(temp));reg=design(dirs)
        if cmd=="freeze":
            if path.exists():raise FileExistsError(path)
            write(path,reg);return
        if reg!=load(path):raise ValueError("frozen inputs changed")
        if cmd=="check":return
        key=os.environ.get("PERSONA_API_KEY") or getpass.getpass("临时实验凭据: ")
        with out.open("x") as f:f.write("[]")
        rows=[];errors=0
        for j in reg["jobs"]:
            r=dict(id=j["id"],status="not_sent",say=None,validity=None,usage=None)
            rows.append(r)
            if errors<2:
                fn=openai_compatible(**PARAMS,base_url=ENDPOINT,api_key=key,timeout=45)
                def send(msgs):
                    assert digest({**PARAMS,"messages":msgs})==j["request"]
                    r["status"]="started";write(out,rows)
                    r["raw"]=fn(msgs)
                    return r["raw"]
                try:
                    _,say,val=trial(j["user"],"full",send,dirs[j["arm"]])
                    r.update(status="ok" if say else "empty",say=say,validity=val)
                except (OSError,RuntimeError) as exc:
                    r.update(status="error",error_type=type(exc).__name__);errors+=1
                    if any(f"HTTP {c}" in str(exc) for c in (401,402,403)):errors=2
                obs=fn.observations.response or {}
                r.update(usage=obs.get("usage"),response_model=obs.get("model"))
            write(out,rows)
            print(j["id"],r["status"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("freeze","check","live"))
    main(parser.parse_args().command)
