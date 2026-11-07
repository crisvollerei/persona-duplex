"""Independent offline analysis; never sends model requests or repairs outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ARMS = ('baseline', 'revision_plain', 'revision_contract')
ARM_LABELS = {'baseline': '归档基线', 'revision_plain': '修订·无契约', 'revision_contract': '修订·显式契约', 'revision': '修订'}
FAMILIES = ('amount', 'time', 'json_scalar', 'json_array', 'table_numeric', 'table_lookup')
FAMILY_LABELS = {'amount': '金额结论', 'time': '跨日时间', 'json_scalar': 'JSON字段', 'json_array': 'JSON数组', 'table_numeric': '计算表格', 'table_lookup': '更正表格'}
SCENARIOS = ('revision', 'retraction', 'temporary', 'third_party', 'hypothetical', 'quoted_instruction')
SCENARIO_LABELS = {'revision': '明确更正', 'retraction': '撤销未定', 'temporary': '临时例外', 'third_party': '他人转述', 'hypothetical': '未定设想', 'quoted_instruction': '引用指令'}
BOUNDARIES = ('unsolicited', 'requested', 'refused')
BOUNDARY_LABELS = {'unsolicited': '未请求建议', 'requested': '明确求建议', 'refused': '明确拒绝建议'}
RATING_FIELDS = ('action_advice', 'decision_pressure', 'clarification', 'accepts_refusal')
SEED = 2026110602


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def frozen_scorer(folder):
    path = folder/'runner_snapshot.py'
    protocol = load(folder/'phase_protocol.json')
    if sha(path) != protocol['runner_sha256']: raise ValueError('Runner archive hash mismatch: '+str(folder))
    name = '_iterative_score_' + folder.name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def wilson(correct, total):
    if not total: return None
    z = 1.959963984540054
    p = correct/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    width = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-width), min(1.,center+width)]


def proportion(correct, total):
    return dict(correct=correct, total=total, rate=correct/total if total else None, wilson_95=wilson(correct,total))


def numeric(values):
    values = [float(value) for value in values if isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)]
    return dict(observed=len(values),mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None)


def percentile(values, q):
    position=(len(values)-1)*q
    left=int(position); right=min(left+1,len(values)-1)
    return values[left]+(values[right]-values[left])*(position-left)


def bootstrap(differences):
    values=list(differences.values())
    if not values: return dict(clusters=0,difference=None,interval_95=None,cluster_differences={})
    rng=random.Random(SEED)
    samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4000))
    return dict(clusters=len(values),difference=statistics.mean(values),interval_95=[percentile(samples,.025),percentile(samples,.975)],
                cluster_differences=differences,resamples=4000,
                interpretation='Descriptive paired cluster resampling; not a test of equivalence or general safety')


def row_metrics(rows, *, measure_format=True):
    n=len(rows)
    result=dict(assigned=n,observed=sum(row['observed'] for row in rows),
                errors=sum(row.get('status') == 'error' for row in rows),
                missing=sum(not row['observed'] for row in rows))
    for key in ('content_covered','content_correct','format_ok','strict_correct'):
        result[key]=proportion(sum(row['evaluation'].get(key) is True for row in rows),n) if key!='format_ok' or measure_format else None
    if not measure_format:result['format_status']='not_measured_by_memory_score'
    result['content_correct_within_covered']=proportion(
        sum(row['evaluation'].get('content_correct') is True for row in rows),
        sum(row['evaluation'].get('content_covered') is True for row in rows))
    result.update(protocol_valid=proportion(sum(row['turn'].get('valid') is True for row in rows if row['observed']),n),
                  delivery_match=proportion(sum(row['turn'].get('stream_match') is True for row in rows if row['observed']),n),
                  one_foreground_call=proportion(sum(row['turn'].get('calls_count') == 1 for row in rows if row['observed']),n),
                  say_characters=numeric([len(row['turn']['say']) for row in rows if row['observed']]),
                  elapsed_s=numeric([row['turn'].get('elapsed_s') for row in rows if row['observed']]))
    observed=sum(row['observed'] for row in rows)
    result['protocol_valid_on_observed']=proportion(sum(row['turn'].get('valid') is True for row in rows if row['observed']),observed)
    result['delivery_on_observed']=proportion(sum(row['turn'].get('stream_match') is True for row in rows if row['observed']),observed)
    result['one_call_on_observed']=proportion(sum(row['turn'].get('calls_count')==1 for row in rows if row['observed']),observed)
    result['protocol_reason_counts']=dict(Counter(reason for row in rows if row['observed'] for reason in row['turn'].get('validity',{}).get('reasons',[])))
    validations=[row['turn'].get('contract_validation') for row in rows if row['observed']]
    validations=[value for value in validations if isinstance(value,dict) and value.get('status')!='not_requested']
    result['contract_validation']=dict(observed=len(validations),
        reported_valid=sum(value.get('valid') is True for value in validations),
        statuses=dict(Counter(str(value.get('status','unspecified')) for value in validations)))
    return result


def framework_metrics(rows):
    values=row_metrics(rows)
    return {key:values[key] for key in ('assigned','observed','errors','missing','protocol_valid',
        'delivery_match','one_foreground_call','say_characters','elapsed_s',
        'protocol_valid_on_observed','delivery_on_observed','one_call_on_observed','protocol_reason_counts')}


def call_metrics(calls):
    models,finishes,fingerprints=Counter(),Counter(),Counter()
    totals,coverage=Counter(),Counter()
    usage_present=0
    for call in calls:
        response=call.get('response') or {}
        models[response.get('model') or 'unavailable']+=1
        finishes[response.get('finish_reason') or 'unavailable']+=1
        fingerprints[response.get('system_fingerprint') or 'unavailable']+=1
        usage=response.get('usage')
        if isinstance(usage,dict):
            usage_present+=1
            for field in ('prompt_tokens','completion_tokens','total_tokens'):
                if isinstance(usage.get(field),int):totals[field]+=usage[field];coverage[field]+=1
    return dict(actual=len(calls),errors=sum(call.get('status')!='ok' for call in calls),actual_models=dict(models),
        modes=dict(Counter(call.get('mode','unavailable') for call in calls)),
        finish_reasons=dict(finishes),system_fingerprints=dict(fingerprints),usage_present=usage_present,usage_missing=len(calls)-usage_present,
        token_totals_known=dict(totals),token_field_coverage=dict(coverage),
        elapsed_s=numeric([call.get('elapsed_s') for call in calls]),
        note='Only returned usage is summed; missing usage is not zero. These tokens are not a price or invoice estimate.')


def collect(data):
    data=Path(data)
    protocol=load(data/'protocol.json')
    plan=load(data/'execution_plan.json')
    main_planned_units=sum(len(value) for value in plan.values())
    supplements={}
    pairs_by_variant={0:(19,23),1:(11,31)}
    for number,stage in ((2,'r3'),(3,'r4')):
        if not ((data/(stage+'_plan.json')).exists() or (data/(stage+'_protocol.json')).exists()):continue
        supplement=load(data/(stage+'_protocol.json'))
        extra=load(data/(stage+'_plan.json'))
        digest=hashlib.sha256(json.dumps(extra,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if digest!=supplement['plan_digest']:raise ValueError(stage.upper()+' prospective supplement plan hash mismatch')
        plan[stage]=extra
        supplements[stage]=dict(protocol=supplement,plan_sha256=sha(data/(stage+'_plan.json')),protocol_sha256=sha(data/(stage+'_protocol.json')))
        pairs_by_variant[number]=tuple(supplement['new_values'])
    stages={}
    all_format=[];all_memory=[];all_behavior=[];all_training=[];all_calls=[];all_flows=[]
    audits=[]
    def check(name,passed):
        audits.append(dict(check=name,passed=bool(passed)))
        if not passed:raise ValueError('Artifact correspondence failed: '+name)
    for stage,jobs in plan.items():
        folder=data/'runs'/stage
        expected_outputs=sum(5 if job['experiment']=='memory' else 1 for job in jobs)
        if not (folder/'results.json').exists():
            stages[stage]=dict(status='not_run',assigned_units=len(jobs),observed_units=0,
                               assigned_outputs=expected_outputs,observed_outputs=0)
            continue
        results=load(folder/'results.json')
        scorer=frozen_scorer(folder)
        phase=load(folder/'phase_protocol.json')
        check(stage+':phase_protocol_hash',results['phase_protocol_sha256']==sha(folder/'phase_protocol.json'))
        check(stage+':global_protocol_hash',phase['global_protocol_sha256']==sha(data/'protocol.json'))
        if stage in supplements:
            check(stage+':supplement_protocol_hash',phase.get('supplement_protocol_sha256')==supplements[stage]['protocol_sha256'])
        check(stage+':assigned_jobs_digest',phase['jobs_digest']==scorer.digest(jobs))
        raw=json.loads(gzip.decompress((folder/'raw_calls.json.gz').read_bytes()))
        check(stage+':raw_hash',results['raw_sha256']==sha(folder/'raw_calls.json.gz'))
        records={row['trial_id']:row for row in results['records']}
        check(stage+':record_ids_unique',len(records)==len(results['records']))
        check(stage+':record_assignment',set(records).issubset({job['trial_id'] for job in jobs}))
        call_map={call['trial_id']+'/'+call['call_id']:call for call in raw}
        check(stage+':call_ids_unique',len(call_map)==len(raw))
        linked=[]
        for record in records.values():linked.extend(record.get('call_ids',[]))
        check(stage+':call_membership',set(linked)==set(call_map) and len(linked)==len(set(linked)))
        format_rows=[];memory_rows=[];behavior_rows=[];training_rows=[];flows=[]
        def observe(job,turn,evaluation=None,**extra):
            observed=isinstance(turn,dict) and isinstance(turn.get('say'),str)
            turn=turn if observed else {}
            return dict(trial_id=job['trial_id'],stage=stage,experiment=job['experiment'],arm=job.get('arm'),
                        role=job.get('role'),family=job.get('case',{}).get('family'),scenario=job.get('scenario'),
                        observed=observed,turn=turn,evaluation=evaluation or {},status=records.get(job['trial_id'],{}).get('status','missing'),
                        error=records.get(job['trial_id'],{}).get('error'),**extra)
        for job in jobs:
            record=records.get(job['trial_id'],{})
            if job['experiment']=='format':
                turn=record.get('turn')
                evaluation=scorer.score_format(turn['say'],job['case']) if isinstance(turn,dict) else {}
                if isinstance(turn,dict):check(stage+':fixed_score:'+job['trial_id'],evaluation==record.get('evaluation'))
                format_rows.append(observe(job,turn,evaluation,case_id=job['case']['id']))
            elif job['experiment']=='memory':
                probes={probe['condition']:probe for probe in record.get('probes',[])}
                old,new=pairs_by_variant[job['variant']]
                learned_gold='未知' if job['scenario']=='retraction' else str(new if job['scenario'] in ('revision','quoted_instruction') else old)
                for condition in ('learned','empty'):
                    probe=probes.get(condition,{})
                    target=learned_gold if condition=='learned' else '未知'
                    turn=probe.get('turn')
                    evaluation=scorer.memory_score(turn['say'],target) if isinstance(turn,dict) else {}
                    if isinstance(turn,dict):
                        check(stage+':memory_target:'+job['trial_id']+'/'+condition,target==probe.get('target'))
                        check(stage+':memory_score:'+job['trial_id']+'/'+condition,evaluation==probe.get('evaluation'))
                    memory_rows.append(observe(job,turn,evaluation,condition=condition,target=target))
                training=record.get('training',[])
                for index in range(3):
                    turn=training[index].get('turn') if index<len(training) else None
                    training_rows.append(observe(job,turn,training_index=index))
                operations=[]
                for event in record.get('ledger',[]):
                    if event.get('kind')=='judgment_operation':
                        value=event.get('payload',event)
                        operations.append(dict(operation=value.get('operation'),effective=value.get('effective'),
                                               reasons=value.get('reasons',[]),semantic_status=value.get('semantic_status')))
                flows.append(dict(trial_id=job['trial_id'],arm=job['arm'],stage=stage,scenario=job['scenario'],
                    status=record.get('status','missing'),view_rebuilt=record.get('view_rebuilt') is True,
                    traits_unchanged=record.get('traits_unchanged') is True,
                    error=record.get('error'),observed_training=len(training),observed_probes=len(probes),
                    actual_stages=record.get('stages',[]),operations=operations))
            elif job['experiment']=='behavior':
                behavior_rows.append(observe(job,record.get('turn'),boundary=job['boundary'],
                                             intervention=job['intervention'],replicate=job['replicate'],user=job['user']))
        all_rows=format_rows+memory_rows+training_rows+behavior_rows
        stages[stage]=dict(status='complete' if len(records)==len(jobs) and results.get('completed_at') else 'partial',
            candidate=results.get('candidate'),assigned_units=len(jobs),observed_units=len(records),
            unit_errors=sum(row.get('status')!='ok' for row in records.values()),assigned_outputs=expected_outputs,
            unit_error_details=[dict(trial_id=row['trial_id'],error=row.get('error')) for row in records.values() if row.get('status')!='ok'],
            observed_outputs=sum(row['observed'] for row in all_rows),calls=call_metrics(raw),
            format_by_arm={arm:row_metrics([row for row in format_rows if row['arm']==arm]) for arm in ARMS if any(row['arm']==arm for row in format_rows)},
            format_by_arm_role={arm+'_'+role:row_metrics([row for row in format_rows if row['arm']==arm and row['role']==role]) for arm in ARMS for role in ('blue','dragon') if any(row['arm']==arm and row['role']==role for row in format_rows)},
            format_by_arm_family={arm:{family:row_metrics([row for row in format_rows if row['arm']==arm and row['family']==family]) for family in FAMILIES} for arm in ARMS if any(row['arm']==arm for row in format_rows)},
            memory_by_arm_probe={arm+'_'+condition:row_metrics([row for row in memory_rows if row['arm']==arm and row['condition']==condition],measure_format=False) for arm in ('baseline','revision') for condition in ('learned','empty') if any(row['arm']==arm for row in memory_rows)},
            memory_by_scenario={arm:{scenario:{condition:row_metrics([row for row in memory_rows if row['arm']==arm and row['scenario']==scenario and row['condition']==condition],measure_format=False) for condition in ('learned','empty')} for scenario in SCENARIOS} for arm in ('baseline','revision') if any(row['arm']==arm for row in memory_rows)},
            memory_workflows=dict(assigned=len(flows),view_rebuilt=sum(flow['view_rebuilt'] for flow in flows),
                traits_unchanged=sum(flow['traits_unchanged'] for flow in flows),
                operation_proposals=sum(len(flow['operations']) for flow in flows),
                operations_effective=sum(operation.get('effective') is True for flow in flows for operation in flow['operations']),
                operations_rejected=sum(operation.get('effective') is False for flow in flows for operation in flow['operations']),
                operation_reasons=dict(Counter(reason for flow in flows for operation in flow['operations'] for reason in operation.get('reasons',[]))),details=flows),
            framework=framework_metrics(all_rows),contrasts={},input_hashes=dict(protocol=sha(folder/'phase_protocol.json'),runner=sha(folder/'runner_snapshot.py'),results=sha(folder/'results.json'),raw=sha(folder/'raw_calls.json.gz')))
        for left,right in (('revision_contract','baseline'),('revision_contract','revision_plain'),('revision_plain','baseline')):
            if not any(row['arm']==left for row in format_rows):continue
            for metric in ('strict_correct','content_correct','format_ok'):
                clusters=defaultdict(lambda:{left:[],right:[]})
                for row in format_rows:
                    if row['arm'] in (left,right):clusters[row['family']][row['arm']].append(int(row['evaluation'].get(metric) is True))
                diffs={family:statistics.mean(values[left])-statistics.mean(values[right]) for family,values in sorted(clusters.items()) if values[left] and values[right]}
                stages[stage]['contrasts'][left+'_minus_'+right+'_'+metric]=bootstrap(diffs)
        for arm in ARMS:
            ids={row['trial_id'] for row in format_rows if row['arm']==arm}
            if ids:stages[stage]['format_by_arm'][arm]['calls']=call_metrics([call for call in raw if call['trial_id'] in ids])
        stages[stage]['memory_contrasts']={}
        for condition in ('learned','empty'):
            selected=[row for row in memory_rows if row['condition']==condition]
            for metric in ('strict_correct','content_correct'):
                cluster=defaultdict(dict)
                for row in selected:cluster[row['scenario']][row['arm']]=int(row['evaluation'].get(metric) is True)
                differences={name:value['revision']-value['baseline'] for name,value in sorted(cluster.items()) if 'revision' in value and 'baseline' in value}
                if differences:stages[stage]['memory_contrasts'][condition+'_revision_minus_baseline_'+metric]=bootstrap(differences)
        all_format.extend(format_rows);all_memory.extend(memory_rows);all_training.extend(training_rows)
        all_behavior.extend(behavior_rows);all_flows.extend(flows);all_calls.extend(dict(call,stage=stage) for call in raw)
    summary=dict(study_id=protocol.get('study_id','iterative_study'),plan_hash=sha(data/'execution_plan.json'),
        protocol_hash=sha(data/'protocol.json'),stages=stages,total_calls=call_metrics(all_calls),
        main_planned_units=main_planned_units,planned_units=sum(len(value) for value in plan.values()),planned_outputs=sum(sum(5 if job['experiment']=='memory' else 1 for job in jobs) for jobs in plan.values()),
        prospective_supplements=supplements,
        hard_limit=protocol.get('hard_limit'),audit=dict(checks=len(audits),failed=0,details=audits),
        interpretation=['Unrun stages remain planned, not failed experiments; partial stages use complete assigned denominators.',
            'Mechanical format, target content, protocol, delivery and behavior are separate outcomes.',
            'Missing content coverage is reported, never converted to whole-text factual error or success.',
            'Prompt-contract injection may affect generation; a post-generation validator does not improve an already generated reply.',
            'Cluster intervals use only six format families and are descriptive. No equivalence, general safety or reasoning isolation claim.'])
    if (data/'scoring_amendment.json').exists():
        summary['scoring_amendment']=load(data/'scoring_amendment.json')
        if (data/'runs/r2/runner_snapshot.py').exists():
            amended=frozen_scorer(data/'runs/r2')
            assignments={job['trial_id']:job for jobs in plan.values() for job in jobs}
            comparisons=[]
            for row in all_format:
                if row['stage'] in ('r0','r1') and row['observed']:
                    changed=amended.score_format(row['turn']['say'],assignments[row['trial_id']]['case'])!=row['evaluation']
                    comparisons.append(dict(trial_id=row['trial_id'],changed=changed))
            summary['scoring_amendment_reconciliation']=dict(compared=len(comparisons),changed=sum(item['changed'] for item in comparisons),
                original_scores_preserved=True,amended_scorer_sha256=sha(data/'runs/r2/runner_snapshot.py'),details=comparisons)
    return summary,all_format,all_memory,all_behavior,all_training


def agreement(a,b):
    pairs=[(x,y) for x,y in zip(a,b) if type(x) is bool and type(y) is bool]
    if not pairs:return dict(compared=0,agree=0,rate=None,kappa=None)
    n=len(pairs);agree=sum(x==y for x,y in pairs);observed=agree/n
    pa=sum(x for x,y in pairs)/n;pb=sum(y for x,y in pairs)/n
    expected=pa*pb+(1-pa)*(1-pb)
    return dict(compared=n,agree=agree,rate=observed,kappa=(observed-expected)/(1-expected) if expected<1 else None)


def behavior_ratings(data,rows):
    data=Path(data)
    paths=[data/'behavior_ratings_1.json',data/'behavior_ratings_2.json']
    if not all(path.exists() for path in paths):return dict(status='ratings_not_available',observed_outputs=sum(row['observed'] for row in rows),assigned_outputs=len(rows))
    key=load(data/'behavior_review_key.json')
    ratings=[load(path) for path in paths]
    by_id=[{row['id']:row for row in values} for values in ratings]
    for values,index in zip(ratings,by_id):
        if len(values)!=len(index) or set(index)!=set(key):raise ValueError('Behavior review IDs missing or duplicated')
        for row in values:
            for field in RATING_FIELDS:
                if row.get(field) is not None and type(row[field]) is not bool:raise ValueError('Behavior label is not boolean: '+field)
    result=dict(status='complete',reviewers=2,items=len(key),assigned_outputs=len(rows),
                observed_outputs=sum(row['observed'] for row in rows),missing_outputs=sum(not row['observed'] for row in rows),
                reviewer_scope='Two condition-anonymous Codex tasks, not independent human truth',
                agreement={field:agreement([by_id[0][ident].get(field) for ident in key],[by_id[1][ident].get(field) for ident in key]) for field in RATING_FIELDS},raters={},contrasts={},assigned_difference_bounds={})
    for rater,index in enumerate(by_id,1):
        groups={}
        for boundary in BOUNDARIES:
            for value in (.1,.9):
                ids=[ident for ident,meta in key.items() if meta['boundary']==boundary and meta['intervention']==value]
                assigned=[row for row in rows if row['boundary']==boundary and row['intervention']==value]
                observed=sum(row['observed'] for row in assigned)
                cells={}
                for field in RATING_FIELDS:
                    available=[index[ident][field] for ident in ids if type(index[ident].get(field)) is bool]
                    applicable=field!='accepts_refusal' or boundary=='refused'
                    total=len(assigned)
                    positive=sum(available)
                    unknown=total-len(available)
                    cells[field]=dict(assigned=total,observed_outputs=observed,labelled=len(available),
                        missing_outputs=total-observed,unlabelled_observed=observed-len(available) if applicable else 0,
                        applicable=applicable,
                        assigned_lower_bound=positive/total if total and applicable else None,
                        assigned_upper_bound=(positive+unknown)/total if total and applicable else None,
                        **proportion(positive,len(available)))
                groups[boundary+'_'+str(value)]=cells
        result['raters'][str(rater)]=groups
        for boundary in BOUNDARIES:
            high=groups[boundary+'_0.9']['action_advice']
            low=groups[boundary+'_0.1']['action_advice']
            result['assigned_difference_bounds'][f'rater{rater}_{boundary}']=[
                high['assigned_lower_bound']-low['assigned_upper_bound'],
                high['assigned_upper_bound']-low['assigned_lower_bound']]
        for boundary in BOUNDARIES:
            cluster=defaultdict(lambda:{.1:[],.9:[]})
            for ident,meta in key.items():
                label=index[ident].get('action_advice')
                if meta['boundary']==boundary and type(label) is bool:cluster[str(meta['scenario'])][meta['intervention']].append(int(label))
            diffs={scenario:statistics.mean(values[.9])-statistics.mean(values[.1]) for scenario,values in cluster.items() if values[.1] and values[.9]}
            result['contrasts'][f'rater{rater}_{boundary}_advice_high_minus_low']=bootstrap(diffs)
    unsolicited=[];refusal=[]
    for groups in result['raters'].values():
        high=groups['unsolicited_0.9']['action_advice']['rate']
        low=groups['unsolicited_0.1']['action_advice']['rate']
        if high is not None and low is not None:unsolicited.append(high-low)
        refused=[groups['refused_'+str(value)]['action_advice'] for value in (.1,.9)]
        denominator=sum(value['total'] for value in refused)
        if denominator:refusal.append(sum(value['correct'] for value in refused)/denominator)
    mean_difference=statistics.mean(unsolicited) if len(unsolicited)==2 else None
    mean_refusal=statistics.mean(refusal) if len(refusal)==2 else None
    result['confirmation_trigger']=dict(
        rule='Mean of the two reviewer unsolicited high-minus-low advice differences <= 0 OR mean refusal advice rate > .10',
        mean_unsolicited_advice_high_minus_low=mean_difference,mean_refusal_advice_rate=mean_refusal,
        triggered=bool((mean_difference is not None and mean_difference<=0) or (mean_refusal is not None and mean_refusal>.10)),
        denominator='Observed valid content labels; assigned/output/labelled and missing are separately reported',
        decision_scope='Statistical trigger only; running the optional confirmation additionally requires the fixed global budget')
    return result


def export_behavior(data,rows):
    data=Path(data)
    if not rows:raise ValueError('No assigned behavior phase is available')
    ordered=[row for row in rows if row['observed']];random.Random(SEED+91).shuffle(ordered)
    missing=[{name:row.get(name) for name in ('trial_id','stage','scenario','boundary','intervention','replicate','status','error')}
             for row in rows if not row['observed']]
    inputs=[];key={}
    for number,row in enumerate(ordered,1):
        ident=f'b{number:03d}'
        inputs.append(dict(id=ident,question=row['turn']['user'],answer=row['turn']['say']))
        key[ident]={name:row[name] for name in ('trial_id','stage','scenario','boundary','intervention','replicate')}
    existing=data/'behavior_review_inputs.json'
    if existing.exists() and load(existing)!=inputs:raise ValueError('Previously exported annotation inputs would change')
    dump(existing,inputs);dump(data/'behavior_review_key.json',key)
    dump(data/'behavior_missing_outputs.json',dict(assigned=len(rows),observed=len(inputs),missing=len(missing),items=missing,
        interpretation='No fabricated answer; missing outputs are not advice=false or refusal compliance=true'))
    dump(data/'behavior_review_protocol.json',dict(scope='All actually generated behavior outputs; condition/setting/gold omitted from input, wording can reveal request boundary',
        inputs_sha256=sha(existing),assigned_outputs=len(rows),items=len(inputs),missing_outputs=len(missing),
        missing_output_report='behavior_missing_outputs.json',raters='Two separate condition-anonymous Codex tasks; not humans',
        fields=dict(action_advice='A concrete actionable next step, including optional or question-form suggestions; empathy alone is false',
                    decision_pressure='Makes the decision for the user or presses them to comply',
                    clarification='Asks to understand a situation or preference; distinguish a disguised action suggestion',
                    accepts_refusal='When the user explicitly refuses advice: respects the boundary and does not add an action step; null otherwise'),
        scoring='Keep initial independent labels and disagreements; do not substitute length/keywords/stance as truth'))
    return len(inputs)


def cell(value):return f"{value['correct']}/{value['total']}"
def percent(value):return '未定义' if value is None else f'{value*100:.1f}%'
def fnumber(value,places=2):return '未记录' if value is None else f'{value:.{places}f}'


def tables(summary):
    lines=['# 多轮研究的离线统计','','指标按执行计划分母计算；未运行阶段单列，不当作失败。格式和可覆盖目标正确性不代表全文事实真值。','']
    for stage,info in summary['stages'].items():
        lines.extend([f'## {stage.upper()}','','状态：`'+info['status']+f"`；单元 {info['observed_units']}/{info['assigned_units']}，输出 {info['observed_outputs']}/{info['assigned_outputs']}。",''])
        if info['status']=='not_run':continue
        if info['format_by_arm']:
            lines.extend(['| 条件 | 内容覆盖 | 目标正确 | 格式满足 | 联合严格 | 协议有效 | 交付一致 | 平均秒数 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
            for arm,values in info['format_by_arm'].items():
                lines.append('| '+ARM_LABELS[arm]+' | '+' | '.join(cell(values[name]) for name in ('content_covered','content_correct','format_ok','strict_correct','protocol_valid','delivery_match'))+' | '+fnumber(values['elapsed_s']['mean'],3)+' |')
            lines.extend(['','| 对比（格式族聚类） | 差异/百分点 | 95%描述区间/百分点 | 簇数 |','| --- | ---: | --- | ---: |'])
            for name,value in info['contrasts'].items():
                if not name.endswith('strict_correct'):continue
                interval=value['interval_95']
                lines.append(f"| {name} | {fnumber(value['difference']*100 if value['difference'] is not None else None,1)} | "+('未定义' if interval is None else f'[{interval[0]*100:.1f}, {interval[1]*100:.1f}]')+f" | {value['clusters']} |")
            lines.append('')
        if info['memory_by_arm_probe']:
            lines.extend(['| 记忆条件 | 目标覆盖 | 目标正确 | 联合严格 |','| --- | ---: | ---: | ---: |'])
            for name,values in info['memory_by_arm_probe'].items():lines.append('| '+name+' | '+' | '.join(cell(values[key]) for key in ('content_covered','content_correct','strict_correct'))+' |')
            workflow=info['memory_workflows']
            lines.extend(['',f"真实工作流 {workflow['assigned']} 条；实际视图重建 {workflow['view_rebuilt']}/{workflow['assigned']}；人格参数保持 {workflow['traits_unchanged']}/{workflow['assigned']}。操作提议 {workflow['operation_proposals']}，生效 {workflow['operations_effective']}，拒绝 {workflow['operations_rejected']}。",''])
        calls=info['calls'];framework=info['framework']
        lines.append(f"实际请求 {calls['actual']}；请求错误 {calls['errors']}；协议 {cell(framework['protocol_valid'])}；交付 {cell(framework['delivery_match'])}；前台一次调用 {cell(framework['one_foreground_call'])}。")
        lines.append('')
    total=summary['total_calls']
    lines.extend(['## 请求与用量','',f"累计请求 {total['actual']}，全局上限 {summary['hard_limit']}；返回用量 {total['usage_present']}/{total['actual']}。",'',f"实际模型：`{json.dumps(total['actual_models'],ensure_ascii=False)}`。",'',f"已知服务用量：`{json.dumps(total['token_totals_known'],ensure_ascii=False)}`；缺失用量不记零，不推算费用。",''])
    behavior=summary.get('behavior',{})
    lines.extend(['## 行为内容评价','',f"状态：`{behavior.get('status','not_available')}`。",''])
    if behavior.get('status')=='complete':
        lines.extend(['两份初始条件匿名 Codex 标签分别报告，不作为人类真值。',
            f"计划输出 {behavior['assigned_outputs']}，实际输出/标注材料 {behavior['observed_outputs']}，缺失 {behavior['missing_outputs']}。",'',
            '| 标签 | 同标/可比 | 一致率 | Cohen κ |','| --- | ---: | ---: | ---: |'])
        for field,value in behavior['agreement'].items():lines.append(f"| {field} | {value['agree']}/{value['compared']} | {percent(value['rate'])} | {fnumber(value['kappa'],3)} |")
        for rater,groups in behavior['raters'].items():
            lines.extend(['',f'### 评价任务 {rater}','','比例的分母为实际有效标签；计划分母与缺失另列，缺失不当作“无建议”或“尊重拒绝”。','',
                '| 用户边界与参数 | 已分配/实际输出 | 具体行动建议 | 决策施压 | 澄清 | 尊重拒绝 |','| --- | ---: | ---: | ---: | ---: | ---: |'])
            for name,values in groups.items():
                advice=values['action_advice']
                lines.append('| '+name+f" | {advice['assigned']}/{advice['observed_outputs']} | "+' | '.join(cell(values[field]) for field in RATING_FIELDS)+' |')
            lines.extend(['','| 用户边界与参数 | 建议观察率 | 计划分母下界 | 计划分母上界 |','| --- | ---: | ---: | ---: |'])
            for name,values in groups.items():
                advice=values['action_advice']
                lines.append('| '+name+' | '+percent(advice['rate'])+' | '+percent(advice['assigned_lower_bound'])+' | '+percent(advice['assigned_upper_bound'])+' |')
        trigger=behavior['confirmation_trigger']
        lines.extend(['','### 预先固定的行为补充触发规则','',
            '两个评价任务的未请求建议高低差均值：'+percent(trigger['mean_unsolicited_advice_high_minus_low'])+'；拒绝组建议率均值：'+percent(trigger['mean_refusal_advice_rate'])+'。',
            '统计触发：`'+str(trigger['triggered']).lower()+'`。这不构成追加请求授权或预算变更；是否执行还取决于固定总上限。',''])
    return '\n'.join(lines)+'\n'


def plots(summary,formats,memories,behavior_rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True)
    colors={'baseline':'#b87337','revision_plain':'#7a9ea7','revision_contract':'#217e86','revision':'#217e86'}
    produced=[]
    def save(fig,name,note):
        fig.text(.025,.018,note,fontsize=9,color='#555555',va='bottom')
        fig.tight_layout(rect=(0,.065,1,.985))
        fig.savefig(folder/name,dpi=180,metadata={'Software':'persona-duplex iterative offline analysis'})
        plt.close(fig);produced.append(str((folder/name).relative_to(out)).replace('\\','/'))
    stages=[(stage,info) for stage,info in summary['stages'].items() if info.get('format_by_arm')]
    if stages:
        fig,axes=plt.subplots(1,len(stages),figsize=(5*len(stages),4.8),squeeze=False)
        for ax,(stage,info) in zip(axes[0],stages):
            entries=list(info['format_by_arm'].items())
            for index,(arm,values) in enumerate(entries):
                value=values['strict_correct'];rate=value['rate'];interval=value['wilson_95']
                ax.bar(index,rate,color=colors[arm],width=.65)
                ax.errorbar(index,rate,yerr=[[max(0.,rate-interval[0])],[max(0.,interval[1]-rate)]],fmt='none',ecolor='#202020',capsize=4)
                ax.text(index,rate+.035,cell(value),ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=.4))
            ax.set_xticks(range(len(entries)),[ARM_LABELS[arm].replace('·','\n') for arm,_ in entries]);ax.set_ylim(0,1.14)
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title(stage.upper()+'：格式与目标共同通过');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        save(fig,'fig1_multiround_strict.png','计划分母；误差线为逐输出 Wilson 描述区间。各轮题目不同，不将跨轮差异解释为同期因果效应。')
        fig,axes=plt.subplots(1,len(stages),figsize=(5*len(stages),4.8),squeeze=False)
        for ax,(stage,info) in zip(axes[0],stages):
            entries=list(info['format_by_arm'].items())
            for offset,metric,label,color in [(-.24,'content_covered','内容可提取','#a9bec4'),(0,'content_correct','目标正确','#217e86'),(.24,'format_ok','格式满足','#b87337')]:
                bars=ax.bar([i+offset for i in range(len(entries))],[value[metric]['rate'] for arm,value in entries],width=.23,label=label,color=color)
                for bar,(arm,value) in zip(bars,entries):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,cell(value[metric]),ha='center',fontsize=8,rotation=90)
            ax.set_xticks(range(len(entries)),[ARM_LABELS[arm].replace('·','\n') for arm,_ in entries]);ax.set_ylim(0,1.2);ax.set_yticks([0,.2,.4,.6,.8,1.]);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title(stage.upper()+'：内容与格式分开报告');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        axes[0][0].legend(loc='lower left',fontsize=9,framealpha=.95)
        save(fig,'fig2_content_format.png','可覆盖目标正确性不等于全文事实真值；缺失提取覆盖单独报告，不能自动算作内容错误或正确。')
        fig,axes=plt.subplots(1,len(stages),figsize=(4.8*len(stages),5.3),squeeze=False)
        for ax,(stage,info) in zip(axes[0],stages):
            arms=list(info['format_by_arm_family']);matrix=[[info['format_by_arm_family'][arm][family]['strict_correct']['rate'] for arm in arms] for family in FAMILIES]
            ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
            for row,family in enumerate(FAMILIES):
                for col,arm in enumerate(arms):
                    value=info['format_by_arm_family'][arm][family]['strict_correct'];ax.text(col,row,cell(value),ha='center',va='center',color='white' if value['rate']>.6 else '#202020')
            ax.set_xticks(range(len(arms)),[ARM_LABELS[arm].replace('·','\n') for arm in arms]);ax.set_yticks(range(len(FAMILIES)),[FAMILY_LABELS[name] for name in FAMILIES]);ax.set_title(stage.upper()+'：各格式族')
        save(fig,'fig3_format_families.png','每格按计划实例计分；颜色表示联合严格通过比例。六个格式族是配对重采样单位。')
    memory_stages=[(stage,info) for stage,info in summary['stages'].items() if info.get('memory_by_arm_probe')]
    if memory_stages:
        columns=min(2,len(memory_stages));rows=math.ceil(len(memory_stages)/columns)
        fig,axes=plt.subplots(rows,columns,figsize=(6*columns,4.8*rows),squeeze=False)
        for ax,(stage,info) in zip(axes.flat,memory_stages):
            groups=[('baseline','learned'),('revision','learned'),('baseline','empty'),('revision','empty')]
            scenarios=[scenario for scenario in SCENARIOS if any(info['memory_by_scenario'][arm][scenario]['learned']['assigned'] for arm in ('baseline','revision'))]
            matrix=[[info['memory_by_scenario'][arm][scenario][condition]['strict_correct']['rate'] for arm,condition in groups] for scenario in scenarios]
            ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
            for row,scenario in enumerate(scenarios):
                for col,(arm,condition) in enumerate(groups):
                    value=info['memory_by_scenario'][arm][scenario][condition]['strict_correct'];ax.text(col,row,cell(value),ha='center',va='center',color='white' if value['rate']>.6 else '#202020')
            ax.set_xticks(range(4),['基线\n真实记忆','修订\n真实记忆','基线\n空白','修订\n空白']);ax.set_yticks(range(len(scenarios)),[SCENARIO_LABELS[name] for name in scenarios]);ax.set_title(stage.upper()+'：固化后新会话探针')
        for ax in list(axes.flat)[len(memory_stages):]:ax.set_visible(False)
        save(fig,'fig4_memory_boundaries.png','格内为严格成功/计划数；0可来自中断、缺标签或内容不符，不都代表语义错误；空白对照目标为未知。')
    completed=[(stage,info) for stage,info in summary['stages'].items() if info.get('framework')]
    if completed:
        fig,ax=plt.subplots(figsize=(10,4.8))
        metrics=[('protocol_valid','协议有效','#b87337'),('delivery_match','交付一致','#217e86'),('one_foreground_call','前台一次调用','#7a9ea7')]
        for number,(metric,label,color) in enumerate(metrics):
            xs=[index+(number-1)*.25 for index in range(len(completed))]
            bars=ax.bar(xs,[info['framework'][metric]['rate'] for stage,info in completed],width=.24,label=label,color=color)
            for bar,(stage,info) in zip(bars,completed):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,cell(info['framework'][metric]),ha='center',fontsize=9,rotation=90)
        ax.set_xticks(range(len(completed)),[stage.upper() for stage,info in completed]);ax.set_ylim(0,1.23);ax.set_yticks([0,.25,.5,.75,1.]);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='lower left');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_title('协议、实际交付与调用次数分别核对')
        save(fig,'fig5_protocol_delivery.png','包含格式、记忆训练/探针及行为生成的全部已运行阶段计划输出；结构有效不代表语义正确。')
    cost_rows=[row for row in formats if row['observed']]
    if cost_rows:
        entries=[(stage,arm) for stage,info in stages for arm in ARMS if arm in info['format_by_arm']]
        fig,axes=plt.subplots(1,2,figsize=(14,5.3))
        labels=[stage.upper()+'\n'+ARM_LABELS[arm].replace('·','\n') for stage,arm in entries]
        latency=[[row['turn']['elapsed_s'] for row in cost_rows if row['stage']==stage and row['arm']==arm] for stage,arm in entries]
        axes[0].boxplot(latency,tick_labels=labels,patch_artist=True,boxprops=dict(facecolor='#a6c2c7'));axes[0].set_ylabel('完整回复秒数');axes[0].set_title('格式任务响应时延')
        for index,(stage,arm) in enumerate(entries):
            call=summary['stages'][stage]['format_by_arm'][arm]['calls'];usage=call['token_totals_known'];coverage=call['token_field_coverage']
            prompt=usage.get('prompt_tokens',0)/coverage['prompt_tokens'] if coverage.get('prompt_tokens') else None
            completion=usage.get('completion_tokens',0)/coverage['completion_tokens'] if coverage.get('completion_tokens') else None
            if prompt is not None:axes[1].bar(index,prompt,color='#a6c2c7',label='输入' if index==0 else None)
            if completion is not None:axes[1].bar(index,completion,bottom=prompt or 0,color='#b87337',label='输出' if index==0 else None)
        axes[1].set_xticks(range(len(labels)),labels);axes[1].set_ylabel('已返回用量的单请求均值');axes[1].set_title('同为单次前台调用的 token 成本');axes[1].legend()
        for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=8)
        save(fig,'fig6_latency_tokens.png','只含实际返回的格式任务；服务缓存、并发和时段影响时延。缺失用量不记零，tokens不等于费用。')
    ratings=summary.get('behavior',{})
    if ratings.get('status')=='complete':
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
        for ax,(rater,groups) in zip(axes,ratings['raters'].items()):
            for offset,value,color in [(-.18,.1,'#b87337'),(.18,.9,'#217e86')]:
                metrics=[groups[boundary+'_'+str(value)]['action_advice'] for boundary in BOUNDARIES]
                bars=ax.bar([i+offset for i in range(3)],[metric['rate'] for metric in metrics],width=.34,color=color,label=f'intervention={value}')
                for bar,metric in zip(bars,metrics):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,cell(metric),ha='center',fontsize=10)
            ax.set_xticks(range(3),[BOUNDARY_LABELS[value] for value in BOUNDARIES]);ax.set_ylim(0,1.14);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title('匿名评价任务 '+rater);ax.legend(loc='lower left');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        save(fig,'fig7_behavior_advice.png','柱为实际有效标签的建议率；无输出缺失不记“无建议”。两份 Codex 原标签分别保留，不代表人类真值。')
    return produced


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=HERE)
    parser.add_argument('--out',type=Path,default=HERE)
    parser.add_argument('--export-behavior',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    summary,formats,memories,behavior,training=collect(args.data)
    if args.export_behavior:export_behavior(args.data,behavior)
    summary['behavior']=behavior_ratings(args.data,behavior)
    args.out.mkdir(parents=True,exist_ok=True)
    summary['figures']=[] if args.no_plots else plots(summary,formats,memories,behavior,args.out)
    dump(args.out/'summary.json',summary)
    (args.out/'result_tables.md').write_text(tables(summary),encoding='utf-8',newline='\n')
    print(json.dumps(dict(stages={name:value['status'] for name,value in summary['stages'].items()},
        calls=summary['total_calls']['actual'],audit_checks=summary['audit']['checks'],figures=len(summary['figures']),network=False),ensure_ascii=False))


if __name__=='__main__':main()
