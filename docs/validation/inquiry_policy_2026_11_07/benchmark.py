"""Frozen slot-selection comparison with a shared executor and scripted users."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import getpass
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
NOW="2026-11-07T21:00:00+08:00"
MODEL="deepseek-chat"
ENDPOINT="https://api.deepseek.com/v1"


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    secret=os.environ.get("PERSONA_API_KEY")
    if secret and secret in text:raise ValueError("Credential cannot enter study data")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(text,encoding="utf-8",newline="\n");os.replace(temp,path)


def implementation():
    for name,digest in load(HERE/"protocol.json")["hashes"].items():
        if sha(HERE/name)!=digest:raise ValueError("Frozen input changed: "+name)
    alias="_inquiry_study_frozen";folder=HERE/"implementation_snapshot/persona"
    spec=importlib.util.spec_from_file_location(alias,folder/"__init__.py",submodule_search_locations=[str(folder)])
    module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module)
    return {name:importlib.import_module(alias+"."+name) for name in ("policy","policy_session","inquiry","config","store","llm")}


class Budget:
    def __init__(self):self.calls=0;self.errors=0;self.lock=threading.Lock()
    def reserve(self):
        with self.lock:
            if self.calls>=72 or self.errors>=3:raise RuntimeError("Fixed call limit or service error stop")
            self.calls+=1


def select_with_model(m,payload,out,id,turn,budget):
    budget.reserve()
    fn=m["llm"].openai_compatible(MODEL,base_url=ENDPOINT,temperature=.3,max_tokens=128,timeout=60)
    messages=[dict(role="system",content=(HERE/"selector_prompt.txt").read_text(encoding="utf-8")),
              dict(role="user",content=json.dumps(payload,ensure_ascii=False))]
    call=dict(id=id,turn=turn,status="running",started_at=datetime.now(timezone.utc).isoformat(),
              messages=messages,settings=deepcopy(fn.metadata))
    path=out/"calls"/f"{id}-{turn}.json";save(path,call);start=time.monotonic()
    try:
        output=fn(messages);call.update(status="ok",output=output)
        return output
    except BaseException as exc:
        with budget.lock:budget.errors+=1
        call.update(status="error",error_type=type(exc).__name__)
        raise
    finally:
        call.update(elapsed_s=time.monotonic()-start,response=deepcopy(fn.observations.response));save(path,call)


def parse_choice(m,text):
    def pairs(values):
        result={}
        for key,value in values:
            if key in result:raise ValueError("Duplicate proposal key")
            result[key]=value
        return result
    obj=json.loads(text,object_pairs_hook=pairs)
    if not isinstance(obj,dict) or set(obj)!={"slot_id"}:raise ValueError("Expected one slot_id field")
    return m["inquiry"].QuestionChoice(**obj)


def episode(m,job,out,budget):
    case=next(c for c in load(HERE/"cases.json") if c["id"]==job["case"])
    profile=next(p for p in load(HERE/"profiles.json") if p["id"]==job["profile"])
    cfg=m["config"].PersonaConfig.load(HERE/"persona_config.toml")
    policy=m["policy"].PolicyConfig(**profile["policy"])
    inquiry=m["inquiry"].InquiryConfig(**load(HERE/"inquiry_config.json"))
    source=m["policy"].Source("task",case["task"])
    slots=tuple(m["inquiry"].QuestionSlot(s["id"],case["topic"],s["question"],s["kind"],
        m["policy"].Quote("task",s["evidence"]),s["purpose"]) for s in case["slots"])
    def forbidden(messages):raise AssertionError("No content or realization model is part of this study")
    def session():return m["policy_session"].PolicySession(cfg,m["store"].Store(out/"runtime"/job["id"]),
        forbidden,policy=policy,inquiry=inquiry,realization="bounded",clock=lambda:NOW)
    current=session();current.boundary("deny",user="不要给建议",quote="不要给建议",topic=case["topic"])
    rows=[];history=[];ambiguous_sent=False
    for turn in range(1,7):
        if turn==4:current.close();current=session()
        unsolicited=[]
        if case.get("late_answer_turn")==turn:
            for record in current.inquiry_state.records:
                if record.status=="pending":
                    spec=next(s for s in case["slots"] if s["id"]==record.slot.id)
                    reply=spec["answer"]
                    current.resolve_question(record.slot.id,operation="answer",user=reply,quote=reply,topic=case["topic"])
                    unsolicited.append(dict(slot_id=record.slot.id,text=reply));history.append(dict(role="user",content=reply))
        user=case["task"] if turn==1 else case["controls"].get(str(turn),"先记录这些信息。")
        frame=m["policy"].ContentFrame(request="listen",request_quote=user) if user=="只想你听听" else m["policy"].ContentFrame()
        before=current.inquiry_state;control=m["inquiry"].parse_control(user,case["topic"],at=NOW)
        effective=m["inquiry"].apply_control(before,control) if control else before
        request="listen" if control and control["mode"]=="deny" else "open" if control else frame.request
        gap=current._inquiry_gap(case["topic"])
        payload=dict(task=case["task"],user=user,request=request,topic=case["topic"],sources=[asdict(source)],
            candidates=[asdict(s) for s in slots],policy=asdict(policy),inquiry=asdict(inquiry),
            effective_permission=effective.permission(case["topic"],at=NOW),state=asdict(effective),
            gap=gap,current_control=control,advice_permission="deny",history=deepcopy(history))
        row=dict(turn=turn,user=user,frame=asdict(frame),sources=[asdict(source)],unsolicited=unsolicited,
            state_before=asdict(before),effective_state=asdict(effective),gap=gap,status="pending",payload=payload)
        choice=None;proposal_raw=None
        try:
            if job["method"]=="prompt":
                proposal_raw=select_with_model(m,payload,out,job["id"],turn,budget)
                choice=parse_choice(m,proposal_raw)
            elif job["method"]=="no_inquiry":choice=m["inquiry"].QuestionChoice(None)
            row["proposal"]=asdict(choice) if choice is not None else None
            chunks=[]
            result=current.say(user,topic=case["topic"],sources=(source,),frame=frame,questions=slots,
                               question_choice=choice,on_delta=chunks.append)
            row.update(status="ok",say=result.say,decision=asdict(result.decision),selection=result.question_selection,
                       delivery_match="".join(chunks)==result.say,session_calls=result.calls_count,realization=result.realization)
            history.extend([dict(role="user",content=user),dict(role="assistant",content=result.say)])
            selected=result.question_selection["selected_id"]
            if selected is not None:
                spec=next(s for s in case["slots"] if s["id"]==selected)
                response=spec["response"]
                if response=="ambiguous_first":
                    response="ambiguous" if not ambiguous_sent else "answer";ambiguous_sent=True
                reply="这个先不说" if response=="decline" else "还没决定" if response=="ambiguous" else spec["answer"]
                if response in ("answer","decline"):
                    current.resolve_question(selected,operation=response,user=reply,quote=reply,topic=case["topic"])
                row["user_reply"]=dict(slot_id=selected,text=reply,operation=response)
                history.append(dict(role="user",content=reply))
        except Exception as exc:
            row.update(status="error",error_type=type(exc).__name__,say=None,proposal_raw=proposal_raw)
            history.append(dict(role="user",content=user))
        row["state_after"]=asdict(current.inquiry_state);rows.append(row)
        save(out/"checkpoints"/(job["id"]+".json"),dict(**job,turns=rows))
    current.close()
    answers={r.slot.id:r.answer for r in current.inquiry_state.records if r.status=="answered"}
    artifact={s["id"]:answers.get(s["id"]) for s in case["slots"]}
    result=dict(**job,turns=rows,artifact=artifact,complete=all(v is not None for v in artifact.values()),
                known_slots=len(answers),final_state=asdict(current.inquiry_state))
    save(out/"episodes"/(job["id"]+".json"),result)
    return result,current.store.read_ledger()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--mode",choices=("offline","all"),default="offline");args=parser.parse_args()
    if args.out.exists():raise SystemExit("Use a new output directory")
    m=implementation();args.out.mkdir(parents=True)
    jobs=[]
    for case in load(HERE/"cases.json"):
        for profile in load(HERE/"profiles.json"):
            for method in ("program","no_inquiry","prompt"):
                for rep in range(2 if method=="prompt" else 1):
                    if args.mode=="offline" and method=="prompt":continue
                    jobs.append(dict(id=f"{case['id']}-{profile['id']}-{method}-{rep+1}",case=case["id"],profile=profile["id"],method=method,replicate=rep+1))
    random.Random(731107).shuffle(jobs);save(args.out/"execution_plan.json",jobs)
    if args.mode=="all" and not os.environ.get("PERSONA_API_KEY"):
        os.environ["PERSONA_API_KEY"]=getpass.getpass("Temporary API key (not saved): ")
    records=[];ledgers={};budget=Budget()
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(episode,m,job,args.out,budget):job for job in jobs}
        for future in as_completed(pending):
            result,ledger=future.result();records.append(result);ledgers[result["id"]]=ledger
            print(f"Completed {len(records)}/{len(jobs)} episodes; requests {budget.calls}",flush=True)
    records.sort(key=lambda r:r["id"])
    calls=[load(f) for f in sorted((args.out/"calls").glob("*.json"))] if (args.out/"calls").exists() else []
    save(args.out/"results.json",dict(protocol_sha256=sha(HERE/"protocol.json"),mode=args.mode,
        calls_count=budget.calls,planned_outputs=len(jobs)*6,records=records))
    (args.out/"raw_calls.json.gz").write_bytes(gzip.compress(json.dumps(calls,ensure_ascii=False).encode(),mtime=0))
    (args.out/"runtime_ledgers.json.gz").write_bytes(gzip.compress(json.dumps(ledgers,ensure_ascii=False).encode(),mtime=0))
    anonymous=[];key={};order=list(records);random.Random(1173).shuffle(order)
    for i,row in enumerate(order,1):
        id=f"Q{i:02d}";key[id]=row["id"]
        case=next(c for c in load(HERE/"cases.json") if c["id"]==row["case"])
        anonymous.append(dict(id=id,task=case["task"],initial_boundary="不主动建议，允许与任务相关的信息问题",
            turns=[dict(turn=t["turn"],user=t["user"],unsolicited=t["unsolicited"],say=t["say"],
                        user_reply=t.get("user_reply",{}).get("text")) for t in row["turns"]]))
    save(args.out/"review_inputs.json",anonymous);save(args.out/"review_key.json",key)
    print(json.dumps(dict(episodes=len(records),requests=budget.calls,complete_outputs=sum(t["status"]=="ok" for r in records for t in r["turns"])),ensure_ascii=False))


if __name__=="__main__":main()
