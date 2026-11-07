"""Offline audit, anonymous coding export and descriptive pilot summaries."""
from collections import Counter, defaultdict
from dataclasses import asdict
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

HERE=Path(__file__).resolve().parent


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def collect():
    protocol=load(HERE/"protocol.json"); results=load(HERE/"results.json");plan=load(HERE/"execution_plan.json")
    raw=json.loads(gzip.decompress((HERE/"raw_calls.json.gz").read_bytes()))
    checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    check("protocol_hash",sha(HERE/"protocol.json")==results["protocol_sha256"])
    check("runner_hash",sha(HERE/"runner_snapshot.py")==protocol["runner_sha256"])
    check("raw_hash",sha(HERE/"raw_calls.json.gz")==results["raw_sha256"])
    for name,value in protocol["hashes"].items():check("fixed/"+name,sha(HERE/name)==value)
    spec=importlib.util.spec_from_file_location("_policy_pilot_frozen",HERE/"runner_snapshot.py")
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    modules=runner.implementation()
    records={r["id"]:r for r in results["records"]};assignments={r["id"]:r for r in plan}
    check("all_planned_records",len(records)==len(results["records"])==len(plan)==70 and set(records)==set(assignments))
    check("request_budget",len(raw)==results["calls_count"] and len(raw)<=76<=protocol["hard_limit"])
    cases={c["id"]:c for c in load(HERE/"cases.json")+load(HERE/"automatic_cases.json")}
    raw_by=defaultdict(list)
    for call in raw:raw_by[call["trial_id"]].append(call)
    shared=defaultdict(set);controlled=[];automatic=[]
    for id,row in records.items():
        check(id+"/assignment",all(row[k]==v for k,v in assignments[id].items()))
        calls=sorted(raw_by[id],key=lambda c:c["call_index"])
        check(id+"/call_count",len(calls)==row["calls_count"])
        for call in calls:
            check(id+f"/call{call['call_index']}/after_fixed",call["started_at"]>=protocol["fixed_at"])
            if call["stage"]=="render":check(id+"/raw_render_chunks","".join(call["chunks"])==call.get("output",""))
        if row["status"]=="ok":
            check(id+"/delivery","".join(row["delivered_chunks"])==row["say"] and row["delivery_match"])
            check(id+"/actual_reply",[c for c in calls if c["stage"]=="render"][0]["output"]==row["say"])
        if row["kind"]=="controlled":
            controlled.append(row)
            shared[(row["case_id"],row["profile"],row["replicate"])].add(row["input_hash"])
            frame=modules["policy"].ContentFrame.from_json(json.dumps(cases[row["case_id"]]["frame"]))
            check(id+"/same_content",json.loads(json.dumps(asdict(frame)))==row["content"])
            if row["method"]=="policy":
                state=modules["policy"].PolicyState.from_dict(row["state_before"])
                decision=modules["policy"].decide(frame,state,modules["policy"].PolicyConfig(**row["policy_config"]),
                    topic=cases[row["case_id"]]["topic"],key="1",at="2026-11-07T21:00:00+08:00")
                check(id+"/recomputed_decision",json.loads(json.dumps(asdict(decision)))==row["decision"])
        else:automatic.append(row)
    check("all_shared_inputs",len(shared)==32 and all(len(v)==1 for v in shared.values()))
    for path in HERE.rglob("*"):
        if not path.is_file() or path.suffix in (".png",".pyc") or "__pycache__" in path.parts or path.name in ('artifact_manifest.json','audit_report.json'):continue
        content=gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes()
        check("privacy/"+path.relative_to(HERE).as_posix(),re.search(r"sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY",content.decode("utf-8")) is None)
    summary=dict(planned_outputs=70,complete=sum(r["status"]=="ok" for r in records.values()),calls=len(raw),
        models=dict(Counter((c.get("response") or {}).get("model","unknown") for c in raw)),
        automatic=[{k:r.get(k) for k in ("id","status","error_type","calls_count","decision","state_before","state_after","say")} for r in automatic],
        groups={},ratings={},audit=dict(checks=len(checks),failed=0,details=checks))
    for method in ("traditional","policy"):
        rows=[r for r in controlled if r["method"]==method];calls=[c for r in rows for c in raw_by[r["id"]]]
        usages=[c.get("response",{}).get("usage") for c in calls]
        summary["groups"][method]=dict(outputs=len(rows),complete=sum(r["status"]=="ok" for r in rows),calls=len(calls),
            usage_coverage=sum(isinstance(u,dict) for u in usages),
            total_tokens=sum(u["total_tokens"] for u in usages if isinstance(u,dict) and isinstance(u.get("total_tokens"),int)),
            mean_seconds=statistics.mean(r["elapsed_s"] for r in rows),mean_say_characters=statistics.mean(len(r.get("say","")) for r in rows))
    return summary,records,cases


def export(records,cases):
    inputs=[];key={}
    for i,job in enumerate(load(HERE/"execution_plan.json"),1):
        row=records[job["id"]]
        if row["status"]!="ok":continue
        id=f"S{i:03d}";case=cases[row["case_id"]];key[id]=row["id"]
        inputs.append(dict(id=id,user=case["user"],sources=case["sources"],
            confirmed_boundary="当前work主题已有长期不主动建议边界" if case["seed"]=="deny" else "无已确认建议边界",
            open_commitments=["改提纲","改标题"] if case["seed"]=="commitments" else [],say=row["say"]))
    for name,value in (("review_inputs.json",inputs),("review_key.json",key)):
        path=HERE/name
        if path.exists() and load(path)!=value:raise ValueError("Frozen coding input changed")
        save(path,value)


def ratings(summary,records):
    key=load(HERE/"review_key.json") if (HERE/"review_key.json").exists() else {}
    sets=[]
    for index in (1,2):
        path=HERE/f"ratings_{index}.json"
        if not path.exists():continue
        labels=load(path);assert len(labels)==len(key) and {r['id'] for r in labels}==set(key)
        mapping={key[row["id"]]:row for row in labels};sets.append(mapping)
        for row in labels:
            assert row["advice"] in ("yes","no","unclear") and row["focus"] in ("practical","feeling","detail","mixed","unclear")
            assert row["target"] in ("correct","incorrect","no_answer","ambiguous","not_applicable")
            assert type(row["unsupported"]) is bool and type(row["naturalness"]) is int and 1<=row["naturalness"]<=5
        groups={}
        for method in ("traditional","policy"):
            for intervention in (0,1):
                free=[r for r in records.values() if r['kind']=='controlled' and r['method']==method and r['case_id'] in ('work','social') and r['policy_config']['intervention']==intervention]
                values=[mapping[r['id']] for r in free if r['id'] in mapping]
                groups[f"{method}_i{intervention}"]=dict(assigned=len(free),labelled=len(values),advice=dict(Counter(v['advice'] for v in values)))
            for focus in ('practical','feeling'):
                rows=[r for r in records.values() if r['kind']=='controlled' and r['method']==method and r['case_id']!='answer' and r['policy_config'][focus]==1]
                values=[mapping[r['id']] for r in rows if r['id'] in mapping]
                groups[f"{method}_{focus}"]=dict(assigned=len(rows),labelled=len(values),matched=sum(v['focus']==focus for v in values),focus=dict(Counter(v['focus'] for v in values)))
            denied=[r for r in records.values() if r['kind']=='controlled' and r['method']==method and r['case_id']=='denied']
            values=[mapping[r['id']] for r in denied if r['id'] in mapping]
            groups[method+'_denied']=dict(assigned=len(denied),labelled=len(values),advice=dict(Counter(v['advice'] for v in values)))
            facts=[r for r in records.values() if r['kind']=='controlled' and r['method']==method and r['case_id']=='answer']
            values=[mapping[r['id']] for r in facts if r['id'] in mapping]
            groups[method+'_facts']=dict(assigned=len(facts),labelled=len(values),target=dict(Counter(v['target'] for v in values)))
            values=[mapping[r['id']] for r in records.values() if r['kind']=='controlled' and r['method']==method and r['id'] in mapping]
            groups[method+'_expression']=dict(labelled=len(values),unsupported=sum(v['unsupported'] for v in values),naturalness=statistics.mean(v['naturalness'] for v in values))
        summary['ratings'][str(index)]=groups
    if len(sets)==2:
        summary['rating_agreement']={field:dict(equal=sum(sets[0][id][field]==sets[1][id][field] for id in sets[0]),total=len(sets[0])) for field in ('advice','focus','target','unsupported','naturalness')}


def extensions(summary,records):
    folder=HERE/'state_validation'
    if (folder/'results.json').exists():
        data=load(folder/'results.json');protocol=load(folder/'protocol.json');cases=load(folder/'cases.json')
        assert data['protocol_sha256']==sha(folder/'protocol.json')
        assert data['raw_sha256']==sha(folder/'raw_calls.json.gz')
        assert protocol['runner_sha256']==sha(folder/'run.py')
        for name,h in protocol['hashes'].items():assert sha(folder/name)==h
        for name,h in load(folder/'implementation_snapshot/manifest.json')['sources'].items():assert sha(folder/'implementation_snapshot'/name)==h
        raw=json.loads(gzip.decompress((folder/'raw_calls.json.gz').read_bytes()))
        assert len(raw)==data['calls_count']<=12
        check=[];offset=0
        for case,row in zip(cases,data['workflows'],strict=True):
            assert case['id']==row['id']
            for stage in row['stages']:
                calls=raw[offset:offset+stage['calls_count']];offset+=stage['calls_count']
                if stage['status']=='ok':assert calls[-1]['stage']=='render' and calls[-1]['output']==stage['say']
            probe=row['stages'][-1]
            check.append(dict(id=row['id'],complete=all(s['status']=='ok' for s in row['stages']),
                expected_permission=case['expected_permission'],actual_permission=probe['permission'],
                state_match=probe['permission']==case['expected_permission'] and probe['state']['revision']==case['expected_revision'],
                action=probe.get('decision',{}).get('action'),probe_say=probe.get('say','')))
        assert offset==len(raw)
        summary['state_validation']=dict(calls=len(raw),tokens=sum(c['response']['usage']['total_tokens'] for c in raw),
            complete_outputs=sum(s['status']=='ok' for r in data['workflows'] for s in r['stages']),
            state_matches=sum(c['state_match'] for c in check),workflows=check,
            scope='Four fixed state workflows after scope derivation revision; free realization still applies; not a behavior superiority test.')
    folder=HERE/'bounded_replay'
    if (folder/'results.json').exists():
        data=load(folder/'results.json');assert data['source_results_sha256']==sha(HERE/'results.json')
        assert data['runner_sha256']==sha(folder/'runner_snapshot.py')
        for name,h in load(folder/'implementation_snapshot/manifest.json')['sources'].items():assert sha(folder/'implementation_snapshot'/name)==h
        compiled=[r for r in data['records'] if r['realization']=='bounded_atoms'];reused=[r for r in data['records'] if r['realization']=='free_model']
        assert len(data['records'])==32 and len(compiled)==len(reused)==16 and all(r['unchanged_reply'] for r in reused)
        key=load(folder/'review_key.json');labels={}
        for i in (1,2):
            path=folder/f'ratings_{i}.json'
            if not path.exists():continue
            values=load(path);assert len(values)==16 and {r['id'] for r in values}==set(key)
            original=load(HERE/f'ratings_{i}.json');original_key=load(HERE/'review_key.json')
            previous={original_key[r['id']]:r for r in original}
            all_new={key[r['id']]:r for r in values}
            denied=[r for r in compiled if r['case_id']=='denied']
            labels[str(i)]=dict(labelled=len(values),advice=dict(Counter(r['advice'] for r in values)),
                focus_matched=sum(all_new[r['original_id']]['focus']==('practical' if r['profile'].endswith('practical') else 'feeling') for r in compiled),
                unsupported=sum(r['unsupported'] for r in values),naturalness=statistics.mean(r['naturalness'] for r in values),
                previous_same_replies_naturalness=statistics.mean(previous[r['original_id']]['naturalness'] for r in compiled),
                denied_advice=sum(all_new[r['original_id']]['advice']=='yes' for r in denied),denied_count=len(denied))
        summary['bounded_replay']=dict(network_calls=0,compiled_changed=len(compiled),byte_checked_reused=len(reused),ratings=labels,
            scope='Post-hoc offline compilation plus byte-identical reuse of16 old free replies. New labels for16 changed replies; not another online replication. Only the restricted speech acts are bounded.')


def plot(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    folder=out/'figures';folder.mkdir(parents=True,exist_ok=True)
    colors={'traditional':'#b87337','policy':'#217e86'}
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,field,title in zip(axes,('total_tokens','mean_seconds'),('64条受控输出的实际tokens','受控输出的完整平均处理秒数')):
        vals=[summary['groups'][m][field] for m in colors]
        ax.bar(range(2),vals,color=list(colors.values()));ax.set_xticks(range(2),['强传统提示','程序策略']);ax.set_title(title)
        for i,v in enumerate(vals):ax.text(i,v*1.015,f'{v:.2f}' if isinstance(v,float) else str(v),ha='center')
        ax.set_ylim(0,max(vals)*1.15)
    fig.text(.02,.02,'同内容与同初态；两种完整提示包比较。自动提取工作流单列，时间不代表首字延迟。',fontsize=9);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(folder/'fig1_cost.png',dpi=180);plt.close(fig)
    if not summary['ratings']:return
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,(index,r) in zip(axes,summary['ratings'].items()):
        for method,offset in (('traditional',-.18),('policy',.18)):
            vals=[]
            for i in (0,1):
                g=r[f'{method}_i{i}'];vals.append(g['advice'].get('yes',0)/g['labelled'])
            ax.bar([offset,1+offset],vals,width=.34,color=colors[method],label='强传统提示' if method=='traditional' else '程序策略')
            for i,v in enumerate(vals):ax.text(i+offset,v+.025,f'{v:.1%}',ha='center')
        ax.set_xticks([0,1],['intervention=0','intervention=1']);ax.set_ylim(0,1.15);ax.set_title('编码任务'+index+'：实际主动建议');ax.legend(loc='upper left');ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.text(.02,.02,'仅两项无边界开放场景，每方法/介入值8条；实际SAY编码，不用程序动作自报代替。',fontsize=9);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(folder/'fig2_realized_advice.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,(index,r) in zip(axes,summary['ratings'].items()):
        for method,offset in (('traditional',-.18),('policy',.18)):
            vals=[r[f'{method}_{f}']['matched']/r[f'{method}_{f}']['labelled'] for f in ('practical','feeling')]
            ax.bar([offset,1+offset],vals,width=.34,color=colors[method],label='强传统提示' if method=='traditional' else '程序策略')
            for i,v in enumerate(vals):ax.text(i+offset,v+.025,f'{v:.1%}',ha='center')
        ax.set_xticks([0,1],['事务关注','感受关注']);ax.set_ylim(0,1.15);ax.set_title('编码任务'+index+'：首要关注符合配置');ax.legend(loc='upper left');ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.text(.02,.02,'三项非事实问答场景，每方法/关注配置12条；主次与mixed分别保留，不是人类人格体验评分。',fontsize=9);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(folder/'fig3_realized_focus.png',dpi=180);plt.close(fig)
    replay=summary.get('bounded_replay',{}).get('ratings',{})
    if replay:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,(index,r) in zip(axes,replay.items()):
            # Compare the same16 inputs; the original free labels are retained.
            key=load(HERE/'bounded_replay/review_key.json');old_key=load(HERE/'review_key.json')
            old={old_key[v['id']]:v for v in load(HERE/f'ratings_{index}.json')}
            before=sum(old[id]['advice']=='yes' for id in key.values())
            values=[before/16,r['advice'].get('yes',0)/16]
            ax.bar([0,1],values,color=['#b87337','#217e86']);ax.set_xticks([0,1],['原自由表达','受限语句组合']);ax.set_ylim(0,1.15)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            for i,v in enumerate(values):ax.text(i,v+.025,f'{v:.0%}',ha='center')
            ax.set_title('编码任务'+index+'：同16条非建议动作的实际建议')
        fig.text(.02,.02,'受限部分为事后离线编译，0次新增API；其他16条仅复用逐字相同请求的旧回复，不是在线重复。',fontsize=9);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(folder/'fig4_bounded_execution.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE);p.add_argument('--export',action='store_true');p.add_argument('--no-plots',action='store_true');args=p.parse_args()
    summary,records,cases=collect()
    if args.export:export(records,cases)
    ratings(summary,records);extensions(summary,records);save(args.out/'summary.json',summary)
    if not args.no_plots:plot(summary,args.out)
    print(json.dumps({'complete':summary['complete'],'calls':summary['calls'],'checks':summary['audit']['checks'],'rating_sets':len(summary['ratings']),'network':False}))


if __name__=='__main__':main()
