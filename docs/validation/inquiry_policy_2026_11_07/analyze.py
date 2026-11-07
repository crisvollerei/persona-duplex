"""Offline material audit, state-based metrics and figures for all three phases."""
from collections import Counter
from dataclasses import asdict
import argparse
import gzip
import json
from pathlib import Path
import statistics

from benchmark import HERE, NOW, implementation, load, save, sha


def raw(folder,name="raw_calls.json.gz"):
    return json.loads(gzip.decompress((folder/name).read_bytes()))


def collect():
    m=implementation();checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    cases={c["id"]:c for c in load(HERE/"cases.json")};profiles={p["id"]:p for p in load(HERE/"profiles.json")}
    inquiry=m["inquiry"].InquiryConfig(**load(HERE/"inquiry_config.json"))
    summary=dict(phases={},groups={},probe={},coders={},agreement={})
    evaluated={};all_calls=[]
    for phase,folder in (("main",HERE),("confirmation",HERE/"clarified_flow")):
        data=load(folder/"results.json");calls=raw(folder);all_calls+=calls
        check(phase+"/protocol",sha(folder/"protocol.json")==data["protocol_sha256"])
        check(phase+"/requests",len(calls)==data["calls_count"]==(72 if phase=="main" else 36))
        call_map={(c["id"],c["turn"]):c for c in calls};records=data["records"]
        check(phase+"/planned",sum(len(r["turns"]) for r in records)==data["planned_outputs"])
        for row in records:
            id=row["id"];case=cases[row["case"]];profile=profiles[row["profile"]];seq=0;row_checks=[]
            policy=m["policy"].PolicyConfig(**profile["policy"])
            slots=tuple(m["inquiry"].QuestionSlot(s["id"],case["topic"],s["question"],s["kind"],m["policy"].Quote("task",s["evidence"]),s["purpose"]) for s in case["slots"])
            for t in row["turns"]:
                label=id+"/"+str(t["turn"])
                frame=m["policy"].ContentFrame.from_json(json.dumps(t["frame"]))
                control=t["payload"]["current_control"]
                if control:frame=m["policy"].ContentFrame() if control["mode"]=="allow" else m["policy"].ContentFrame(request="listen",request_quote=t["user"])
                base_state=m["policy"].PolicyState(1,(m["policy"].Permission(case["topic"],"deny"),),())
                base=m["policy"].decide(frame,base_state,policy,topic=case["topic"],key=str(seq+1),at=NOW)
                state=m["inquiry"].InquiryState.from_dict(t["effective_state"])
                expected=m["inquiry"].select_question(frame,base,policy,inquiry,state,slots,
                    topic=case["topic"],key=str(seq+1),topic_turns_since_question=t["gap"],at=NOW)
                proposed=t.get("proposal",{});chosen=proposed.get("slot_id") if proposed is not None else expected["selected_id"]
                legal=chosen is None or chosen in expected["eligible_ids"]
                check(label+"/executor acceptance",legal==(t["status"]=="ok"))
                if row["method"]=="prompt":
                    call=call_map[(id,t["turn"])]
                    check(label+"/actual payload",json.loads(call["messages"][-1]["content"])==t["payload"])
                    check(label+"/parsed choice",json.loads(call["output"])==proposed)
                    check(label+"/after freeze",call["started_at"]>=load(folder/"protocol.json")["fixed_at"])
                if t["status"]=="ok":
                    seq+=1
                    check(label+"/selection",t["selection"]["selected_id"]==chosen and t["selection"]["program_selected_id"]==expected["selected_id"])
                    check(label+"/zero session models",t["session_calls"]==0)
                    check(label+"/delivery",t["delivery_match"])
                    if chosen is not None:
                        slot=next(s for s in slots if s.id==chosen)
                        check(label+"/reviewed question text",t["say"]==slot.question)
                row_checks.append(dict(turn=t["turn"],legal=legal,matches=chosen==expected["selected_id"],expected=expected,
                                       proposed_id=chosen,delivered=t["status"]=="ok"))
                if t.get("user_reply"):
                    response=t["user_reply"];spec=next(s for s in case["slots"] if s["id"]==response["slot_id"])
                    if response["operation"]=="answer":check(label+"/simulator answer",response["text"]==spec["answer"])
                    if response["operation"]=="ambiguous":check(label+"/unknown kept",any(r["slot"]["id"]==response["slot_id"] and r["status"]=="pending" for r in t["state_after"]["records"]))
            final=m["inquiry"].InquiryState.from_dict(row["final_state"])
            answers={r.slot.id:r.answer for r in final.records if r.status=="answered"}
            check(id+"/artifact",row["artifact"]=={s["id"]:answers.get(s["id"]) for s in case["slots"]})
            check(id+"/known",row["known_slots"]==len(answers))
            check(id+"/complete",row["complete"]==all(v is not None for v in row["artifact"].values()))
            check(id+"/no extra slots",len(final.records)<=2 and len({r.slot.id for r in final.records})==len(final.records))
            evaluated[(phase,id)]=row_checks
        summary["phases"][phase]=dict(requests=len(calls),planned=data["planned_outputs"],complete_outputs=sum(t["status"]=="ok" for r in records for t in r["turns"]),
            total_tokens=sum(c["response"]["usage"]["total_tokens"] for c in calls),models=dict(Counter(c["response"]["model"] for c in calls)))
        for method in sorted({r["method"] for r in records}):
            rs=[r for r in records if r["method"]==method];ev=[e for r in rs for e in evaluated[(phase,r["id"])]]
            failures=[r for r in rs if any(t["status"]!="ok" for t in r["turns"])]
            summary["groups"][phase+"/"+method]=dict(flows=len(rs),slots=2*len(rs),planned=6*len(rs),
                delivered=sum(e["delivered"] for e in ev),legal=sum(e["legal"] for e in ev),matches=sum(e["matches"] for e in ev),
                known=sum(r["known_slots"] for r in rs),complete=sum(r["complete"] for r in rs),
                questions=sum(len(r["final_state"]["records"]) for r in rs),
                declined=sum(s["status"]=="declined" for r in rs for s in r["final_state"]["records"]),
                rejected_flows=len(failures),first_rejections=len(failures),later_rejections=sum(not e["legal"] for e in ev)-len(failures),
                rejection_reasons=dict(Counter(e["expected"]["reason"] for e in ev if not e["legal"])))
    main=load(HERE/"results.json")["records"];confirmation=load(HERE/"clarified_flow/results.json")["records"]
    def visible(row):return [{k:t.get(k) for k in ("user","unsolicited","say","user_reply")} for t in row["turns"]]
    same=0
    for row in confirmation:
        matched=next(r for r in main if r["method"]=="program" and r["case"]==row["case"] and r["profile"]==row["profile"])
        same+=visible(row)==visible(matched)
    summary["confirmation_visible_equals_program"]=dict(equal=same,total=len(confirmation))
    folder=HERE/"decision_probe";probe=load(folder/"results.json");calls=raw(folder);all_calls+=calls
    check("probe/protocol",probe["protocol_sha256"]==sha(folder/"protocol.json"))
    check("probe/72 requests",len(calls)==probe["calls_count"]==72)
    inputs={r["id"]:r for r in load(folder/"inputs.json")};probe_map={r["id"]:r for r in probe["records"]}
    for row in calls:
        item=inputs[row["state_id"]];choice=json.loads(row["output"])["slot_id"]
        check(row["id"]+"/same payload",json.loads(row["messages"][-1]["content"])==item["payload"])
        check(row["id"]+"/legal",row["legal"]==(choice is None or choice in item["eligible_ids"]))
        check(row["id"]+"/match",row["matches"]==(choice==item["expected_id"]))
        check(row["id"]+"/result",all(probe_map[row["id"]][k]==row[k] for k in ("selected_id","legal","matches","status")))
    pure_cooldown={r["id"]+"-t"+str(e["turn"]) for r in main if r["method"]=="program" for e in evaluated[("main",r["id"])] if e["expected"]["reason"]=="question_cooldown"}
    for method in ("original","clarified"):
        rows=[r for r in calls if r["method"]==method];positive=[r for r in rows if inputs[r["state_id"]]["expected_id"] is not None];negative=[r for r in rows if inputs[r["state_id"]]["expected_id"] is None]
        cool=[r for r in rows if r["state_id"] in pure_cooldown]
        summary["probe"][method]=dict(n=len(rows),legal=sum(r["legal"] for r in rows),matches=sum(r["matches"] for r in rows),
            opportunity_n=len(positive),opportunity_asked=sum(r["selected_id"] is not None for r in positive),opportunity_matches=sum(r["matches"] for r in positive),
            abstention_n=len(negative),abstention_matches=sum(r["matches"] for r in negative),cooldown_n=len(cool),cooldown_matches=sum(r["matches"] for r in cool))
    summary["phases"]["probe"]=dict(requests=len(calls),total_tokens=sum(c["response"]["usage"]["total_tokens"] for c in calls),models=dict(Counter(c["response"]["model"] for c in calls)))
    summary["total_requests"]=len(all_calls);summary["total_tokens"]=sum(c["response"]["usage"]["total_tokens"] for c in all_calls)
    summary["models"]=dict(Counter(c["response"]["model"] for c in all_calls));summary["finish_reasons"]=dict(Counter(c["response"].get("finish_reason") for c in all_calls))
    key=load(HERE/"review_key.json");labels=[];main_by={r["id"]:r for r in main}
    for coder in (1,2):
        ratings=load(HERE/f"ratings_{coder}.json");check(f"coder{coder}/coverage",len(ratings)==24 and {r["id"] for r in ratings}==set(key))
        mapping={key[r["id"]]:r for r in ratings};labels.append(mapping)
        for id,r in mapping.items():
            check(f"coder{coder}/{id}/fields",len(r["turns"])==6 and {t["turn"] for t in r["turns"]}==set(range(1,7)) and type(r["coherence"]) is int and 1<=r["coherence"]<=5 and r["progress"] in ("none","partial","complete"))
            check(f"coder{coder}/{id}/labels",all(t["question"] in ("none","relevant","irrelevant","unclear") and type(t["pressure"]) is bool and type(t["unsupported"]) is bool for t in r["turns"]))
        summary["coders"][str(coder)]={method:dict(flows=len(rs),coherence=statistics.mean(mapping[r["id"]]["coherence"] for r in rs),
            progress=dict(Counter(mapping[r["id"]]["progress"] for r in rs)),questions=dict(Counter(t["question"] for r in rs for t in mapping[r["id"]]["turns"])),
            pressure=sum(t["pressure"] for r in rs for t in mapping[r["id"]]["turns"]),unsupported=sum(t["unsupported"] for r in rs for t in mapping[r["id"]]["turns"]))
            for method in ("program","no_inquiry","prompt") for rs in ([r for r in main if r["method"]==method],)}
    for name in ("coherence","progress"):
        summary["agreement"][name]=dict(equal=sum(labels[0][id][name]==labels[1][id][name] for id in main_by),n=24)
    for name in ("question","pressure","unsupported"):
        summary["agreement"][name]=dict(equal=sum(a[name]==b[name] for id in main_by for a,b in zip(sorted(labels[0][id]["turns"],key=lambda t:t["turn"]),sorted(labels[1][id]["turns"],key=lambda t:t["turn"]))),n=144)
    for folder in (HERE/"decision_probe",HERE/"clarified_flow"):
        for name,digest in load(folder/"protocol.json")["hashes"].items():check(str(folder.name)+"/hash/"+name,sha(folder/name)==digest)
    return summary,dict(checks=len(checks),failed=0,details=checks)


def figures(s,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"figure.dpi":150,"savefig.dpi":150,"axes.spines.top":False,"axes.spines.right":False})
    names=["No inquiry","Program","Original prompt","Clarified prompt"]
    groups=[s["groups"][k] for k in ("main/no_inquiry","main/program","main/prompt","confirmation/prompt")]
    colors=["#a2acb9","#267899","#b97455","#558b70"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,key,denom,title in zip(axes,("known","complete"),("slots","flows"),("Explicit answers acquired","Complete application records")):
        ax.bar(range(4),[g[key]/g[denom] for g in groups],color=colors)
        for x,g in enumerate(groups):ax.text(x,g[key]/g[denom]+.025,f"{g[key]}/{g[denom]}",ha="center")
        ax.set(xticks=range(4),xticklabels=names,ylim=(0,1.13),title=title);ax.tick_params(axis="x",labelrotation=15);ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Scripted information channel; refusal keeps one field unknown",fontsize=11)
    fig.tight_layout();fig.savefig(out/"fig1_information.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.7));vals=[g["delivered"]/g["planned"] for g in groups]
    ax.bar(range(4),vals,color=colors)
    for x,g in enumerate(groups):ax.text(x,vals[x]+.025,f"{g['delivered']}/{g['planned']}",ha="center")
    ax.set(xticks=range(4),xticklabels=names,ylim=(0,1.14),ylabel="Completed planned steps",title="Shared executor rejects invalid proposals; no retry or silent replacement")
    ax.yaxis.set_major_formatter(PercentFormatter(1));fig.text(.5,.015,"Original prompt: 9 first cooldown rejections + 24 subsequent rejections in the same feedback loop.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/"fig2_delivery.png");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.3))
    for ax,num,den,title in zip(axes,("legal","opportunity_matches","abstention_matches"),("n","opportunity_n","abstention_n"),("Legal proposals","Preferred slot when eligible","Abstain when required")):
        for x,method in enumerate(("original","clarified")):
            g=s["probe"][method];ax.bar(x,g[num]/g[den],color=colors[2+x]);ax.text(x,g[num]/g[den]+.025,f"{g[num]}/{g[den]}",ha="center")
        ax.set(xticks=[0,1],xticklabels=["Original","Clarified"],ylim=(0,1.15),title=title);ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Paired prompts at 36 fixed shared states; these are decision probes, not new dialogue outputs",fontsize=10)
    fig.tight_layout();fig.savefig(out/"fig3_shared_states.png");plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    s,a=collect();args.out.mkdir(parents=True,exist_ok=True);save(args.out/"summary.json",s);save(args.out/"audit_report.json",a)
    folder=args.out/"figures";folder.mkdir(exist_ok=True);figures(s,folder)
    print(json.dumps(dict(checks=a["checks"],requests=s["total_requests"],tokens=s["total_tokens"],probe=s["probe"],confirmation=s["confirmation_visible_equals_program"]),ensure_ascii=False))


if __name__=="__main__":main()
