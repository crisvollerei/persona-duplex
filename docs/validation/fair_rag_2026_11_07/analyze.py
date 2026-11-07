"""Offline shared-retrieval comparison; no model requests or answer rewriting."""
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

HERE=Path(__file__).resolve().parent
METHODS=('traditional','framework')
PROFILES=('neutral','blue','dragon')
METHOD_LABELS={'traditional':'传统强提示＋共享资料','framework':'默认框架＋共享资料'}
PROFILE_LABELS={'neutral':'中性参照','blue':'蓝鱼','dragon':'白龙'}
METRICS=('content_covered','target_success','conclusion_label_present','format_ok','strict_correct')
RATING_SIDES=('a','b')
SEED=2026110701


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def wilson(correct,total):
    if total==0:return None
    z=1.959963984540054;p=correct/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    width=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-width),min(1.,center+width)]
def proportion(correct,total):
    return dict(correct=correct,total=total,rate=correct/total if total else None,wilson_95=wilson(correct,total))
def number(values):
    values=[value for value in values if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)]
    return dict(observed=len(values),mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,minimum=min(values) if values else None,maximum=max(values) if values else None)
def percentile(values,q):
    index=(len(values)-1)*q;lower=int(index);upper=min(lower+1,len(values)-1)
    return values[lower]+(values[upper]-values[lower])*(index-lower)
def bootstrap(clusters,seed=SEED):
    values=list(clusters.values())
    if not values:return dict(estimate=None,interval_95=None,clusters=0,cluster_values={})
    rng=random.Random(seed)
    samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4000))
    return dict(estimate=statistics.mean(values),interval_95=[percentile(samples,.025),percentile(samples,.975)],
                clusters=len(values),cluster_values=clusters,resamples=4000,
                scope='Fixed-family descriptive paired bootstrap, not evidence of equivalence or general reasoning isolation')


def scorer(data,protocol):
    source=next((path for path in (data/'runner_snapshot.py',data/'benchmark.py') if path.exists() and sha(path)==protocol['runner_sha256']),None)
    if source is None:raise ValueError('No runner matching the fixed protocol hash')
    spec=importlib.util.spec_from_file_location('_fair_rag_frozen_scoring',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def row_summary(rows):
    facts=[row for row in rows if row['kind']!='open']
    observed=[row for row in rows if row['observed']]
    framework=[row for row in rows if row['method']=='framework']
    result=dict(assigned=len(rows),observed=len(observed),missing=len(rows)-len(observed),
                errors=sum(row['status']=='error' for row in rows),scored_assigned=len(facts))
    for metric in METRICS:
        result[metric]=proportion(sum(row['evaluation'].get(metric) is True for row in facts),len(facts))
    covered=sum(row['evaluation'].get('content_covered') is True for row in facts)
    correct=sum(row['evaluation'].get('target_correct') is True for row in facts)
    result['target_correct_within_covered']=proportion(correct,covered)
    result['known_incorrect']=sum(row['evaluation'].get('target_correct') is False for row in facts)
    result['unscored_target']=len(facts)-covered
    result['assigned_target_success_bounds']=[correct/len(facts),(correct+len(facts)-covered)/len(facts)] if facts else None
    required=[row for row in facts if row['case']['required_conditions']]
    result['required_condition_word_coverage']=proportion(sum(row['evaluation'].get('required_condition_words_present') is True for row in required),len(required))
    result['delivery_match']=proportion(sum(row['record'].get('delivery_match') is True for row in rows),len(rows))
    result['delivery_on_observed']=proportion(sum(row['record'].get('delivery_match') is True for row in observed),len(observed))
    result['one_call_on_observed']=proportion(sum(row['record'].get('calls_count')==1 for row in observed),len(observed))
    result['protocol_valid']=proportion(sum(row['record'].get('protocol_valid') is True for row in framework),len(framework)) if framework else None
    fw_observed=[row for row in framework if row['observed']]
    result['protocol_on_observed']=proportion(sum(row['record'].get('protocol_valid') is True for row in fw_observed),len(fw_observed)) if framework else None
    result['protocol_reason_counts']=dict(Counter(reason for row in fw_observed for reason in row['record'].get('validity',{}).get('reasons',[])))
    result['say_characters']=number([len(row['record']['say']) for row in observed])
    result['elapsed_s']=number([row['record'].get('elapsed_s') for row in observed])
    return result


def calls_summary(calls):
    models=Counter();fingerprints=Counter();finishes=Counter();known=Counter();coverage=Counter();usages=0
    request_seconds=[]
    for call in calls:
        response=call.get('response') or {}
        models[response.get('model') or 'unavailable']+=1
        fingerprints[response.get('system_fingerprint') or 'unavailable']+=1
        finishes[response.get('finish_reason') or 'unavailable']+=1
        usage=response.get('usage')
        if isinstance(usage,dict):
            usages+=1
            for field in ('prompt_tokens','completion_tokens','total_tokens'):
                if isinstance(usage.get(field),int) and not isinstance(usage[field],bool):known[field]+=usage[field];coverage[field]+=1
        request_seconds.append(call.get('elapsed_s'))
    return dict(actual=len(calls),errors=sum(call.get('status')=='error' for call in calls),
        incomplete=sum(call.get('status') not in ('ok','error') for call in calls),
        actual_models=dict(models),system_fingerprints=dict(fingerprints),finish_reasons=dict(finishes),
        usage_present=usages,usage_missing=len(calls)-usages,known_token_totals=dict(known),token_field_coverage=dict(coverage),
        token_means_known={field:known[field]/coverage[field] if coverage[field] else None for field in ('prompt_tokens','completion_tokens','total_tokens')},
        elapsed_s=number(request_seconds),note='Only returned usage is summed. Missing usage is not zero; no billing estimate.')


def collect(data):
    protocol=load(data/'protocol.json');plan=load(data/'execution_plan.json');cases={row['id']:row for row in load(data/'cases.json')}
    retrieval=load(data/'retrieval.json');configs=load(data/'configurations.json')
    fixed=scorer(data,protocol)
    audits=[]
    def check(label,condition):
        audits.append(dict(check=label,passed=bool(condition)))
        if not condition:raise ValueError('Material reconciliation failed: '+label)
    for name,expected in protocol['hashes'].items():check('fixed:'+name,sha(data/name)==expected)
    results=load(data/'results.json') if (data/'results.json').exists() else {'records':[]}
    raw=json.loads(gzip.decompress((data/'raw_calls.json.gz').read_bytes())) if (data/'raw_calls.json.gz').exists() else []
    assigned={job['trial_id']:job for job in plan};records={row['trial_id']:row for row in results['records']}
    check('unique_assignments',len(assigned)==len(plan))
    check('unique_observed_units',len(records)==len(results['records']))
    check('no_unknown_observed_units',set(records)<=set(assigned))
    if results.get('protocol_sha256'):check('result_protocol_hash',results['protocol_sha256']==sha(data/'protocol.json'))
    if results.get('raw_sha256'):check('raw_hash',results['raw_sha256']==sha(data/'raw_calls.json.gz'))
    if 'calls_count' in results:check('recorded_call_count',results['calls_count']==len(raw))
    check('hard_request_budget',len(raw)<=protocol['hard_limit'])
    call_map={call['trial_id']+'/'+call['call_id']:call for call in raw}
    check('unique_call_ids',len(call_map)==len(raw))
    rows=[];discrepancies=[]
    for ident,job in assigned.items():
        record=records.get(ident,{})
        case=cases[job['case_id']]
        observed=record.get('status')=='ok' and isinstance(record.get('say'),str)
        evaluation=fixed.score(record['say'],case) if observed and case['kind']!='open' else None
        if observed and evaluation!=record.get('evaluation'):discrepancies.append(ident)
        if record:
            check(ident+'/config_hash',record['config_hash']==fixed.digest(configs[job['profile']]))
            check(ident+'/pack_reference',record['context_sha256']==retrieval[case['id']]['context_sha256'])
            linked=[call_map[key] for key in record.get('call_ids',[])]
            check(ident+'/linked_calls',all(call['trial_id']==ident for call in linked) and len(linked)==record.get('calls_count'))
            if observed:check(ident+'/one_foreground_call',len(linked)==1)
            for call in linked:
                check(ident+'/'+call['call_id']+'/same_pack',call['context_sha256']==retrieval[case['id']]['context_sha256'])
                messages=call['request']['messages']
                check(ident+'/'+call['call_id']+'/system_pack',messages[0]['role']=='system' and messages[0]['content'].endswith(retrieval[case['id']]['context']))
                check(ident+'/'+call['call_id']+'/common_rules',fixed.COMMON in messages[0]['content'])
                check(ident+'/'+call['call_id']+'/same_question',messages[-1]['role']=='user' and messages[-1]['content']==case['user'])
                if call.get('status')=='ok':check(ident+'/'+call['call_id']+'/chunks',''.join(call['chunks'])==call['output'])
        rows.append({**job,'status':record.get('status','missing'),'observed':observed,'record':record,
                     'case':case,'evaluation':evaluation or {}})
    check('saved_scores_match_frozen_runner',not discrepancies)
    for ident,context in retrieval.items():
        check(ident+'/pack_hash',hashlib.sha256(context['context'].encode()).hexdigest()==context['context_sha256'])
        check(ident+'/budget',len(context['context'])<=2600)
        check(ident+'/required_sources',set(cases[ident]['required_source_ids'])<=set(context['source_ids']))
    facts=[row for row in rows if row['kind']!='open'];opened=[row for row in rows if row['kind']=='open']
    scores={(row['method'],row['profile'],row['case_id'],row['replicate']):int(row['evaluation'].get('target_success') is True) for row in facts}
    families=sorted({row['family'] for row in facts});contrasts={}
    if records:
        for profile in ('blue','dragon'):
            for method in METHODS:
                clusters={family:statistics.mean(scores[(method,'neutral',row['case_id'],row['replicate'])]-scores[(method,profile,row['case_id'],row['replicate'])]
                    for row in facts if row['method']==method and row['profile']==profile and row['family']==family) for family in families}
                contrasts[f'persona_target_score_loss_{method}_{profile}']=bootstrap(clusters)
            interactions={family:contrasts[f'persona_target_score_loss_framework_{profile}']['cluster_values'][family]-contrasts[f'persona_target_score_loss_traditional_{profile}']['cluster_values'][family] for family in families}
            contrasts[f'persona_loss_framework_minus_traditional_{profile}']=bootstrap(interactions)
        for profile in PROFILES:
            clusters={family:statistics.mean(scores[('framework',profile,row['case_id'],row['replicate'])]-scores[('traditional',profile,row['case_id'],row['replicate'])]
                for row in facts if row['method']=='framework' and row['profile']==profile and row['family']==family) for family in families}
            contrasts[f'target_framework_minus_traditional_{profile}']=bootstrap(clusters)
    group_calls={}
    for method in METHODS:
        for profile in PROFILES:
            selected=[call for call in raw if assigned[call['trial_id']]['method']==method and assigned[call['trial_id']]['profile']==profile]
            group_calls[method+'_'+profile]=calls_summary(selected)
    fact_groups={method+'_'+profile:row_summary([row for row in facts if row['method']==method and row['profile']==profile]) for method in METHODS for profile in PROFILES}
    if contrasts:
        def bounds(method,profile):return fact_groups[method+'_'+profile]['assigned_target_success_bounds']
        for profile in ('blue','dragon'):
            losses={}
            for method in METHODS:
                neutral,persona=bounds(method,'neutral'),bounds(method,profile)
                losses[method]=[neutral[0]-persona[1],neutral[1]-persona[0]]
                contrasts[f'persona_target_score_loss_{method}_{profile}']['unscored_identification_bounds']=losses[method]
            contrasts[f'persona_loss_framework_minus_traditional_{profile}']['unscored_identification_bounds']=[losses['framework'][0]-losses['traditional'][1],losses['framework'][1]-losses['traditional'][0]]
        for profile in PROFILES:
            frame,plain=bounds('framework',profile),bounds('traditional',profile)
            contrasts[f'target_framework_minus_traditional_{profile}']['unscored_identification_bounds']=[frame[0]-plain[1],frame[1]-plain[0]]
    return dict(study_id=protocol['study_id'],status='complete' if len(records)==len(plan) else 'partial' if records else 'not_run',
        input_hashes=dict(protocol=sha(data/'protocol.json'),plan=sha(data/'execution_plan.json'),results=sha(data/'results.json') if records else None,
                          raw=sha(data/'raw_calls.json.gz') if raw else None),
        design=dict(assigned_units=len(plan),recorded_units=len(records),observed_outputs=sum(row['observed'] for row in rows),
                    missing_outputs=sum(not row['observed'] for row in rows),unit_errors=sum(row['status']=='error' for row in rows),
                    facts_assigned=len(facts),open_assigned=len(opened)),
        facts=fact_groups,
        facts_by_family={method+'_'+profile:{family:row_summary([row for row in facts if row['method']==method and row['profile']==profile and row['family']==family]) for family in families} for method in METHODS for profile in PROFILES},
        open={method+'_'+profile:row_summary([row for row in opened if row['method']==method and row['profile']==profile]) for method in METHODS for profile in ('blue','dragon')},
        framework=row_summary([row for row in rows if row['method']=='framework']),
        calls=calls_summary(raw),calls_by_group=group_calls,contrasts=contrasts,
        retrieval=dict(corpus_documents=len(load(data/'corpus.json')),contexts=len(retrieval),
            required_sources_present=sum(context['required_sources_present'] for context in retrieval.values()),
            context_characters=number([context['character_count'] for context in retrieval.values()]),
            context_hashes_shared_and_checked=True,
            interpretation='Each entity has exactly six documents and top-k is six. This is controlled shared evidence, not a retrieval-quality or compression-advantage experiment.'),
        audit=dict(checks=len(audits),failed=0,details=audits),
        limitations=['Target success is an assigned-denominator extraction match lower bound. Uncovered targets remain unscored, not known reasoning errors; identification bounds and covered-subset accuracy are separate.',
          'Difference from each method neutral reference describes configuration-associated target-score loss, not an observed internal reasoning process.',
          'Sixteen closed-world fact instances, eight families, two repetitions; bootstrap intervals are descriptive with few clusters.',
          'Whole prompt packages differ. One call and equal output caps do not imply equal tokens or computation.',
          'Same service/model metadata, no server seed or independent-model replication.',
          'Two Codex coding tasks are exploratory auxiliary evaluations, not independent human ratings or a persona noninferiority test.']),rows


CREATIVE_PROFILES={
 'neutral':'中性表达参照：没有特定角色身份，使用自然中文。角色一致性不评分，填null。',
 'blue':'蓝色大肥鱼表达参考：聪明、贪吃、有点爱面子，亲切俏皮，偶尔轻轻顶一句；被亲昵调侃时轻轻抗议，实际求助时认真帮忙。自然中文偏短句。鲸鱼、蓝色、饭碗是虚构意象，不要求每句提及，不以口癖数量评分。',
 'dragon':'白色龙娘表达参考：优雅沉稳、平静礼貌、略带清冷与反差，耐心而有条理，情绪克制，给人自己决定的空间。龙角、白色、龙翼是虚构意象，不要求每句提及，不以口癖数量评分。'}


def export_pairs(data,rows):
    grouped=defaultdict(dict)
    for row in rows:grouped[(row['case_id'],row['profile'],row['replicate'])][row['method']]=row
    assignment=load(data/'pair_assignment.json')
    if (data/'review_protocol.json').exists() and load(data/'review_protocol.json')['assignment_sha256']!=sha(data/'pair_assignment.json'):
        raise ValueError('Pre-generation pair assignment hash changed')
    if len(assignment)!=len(grouped) or {(a['case_id'],a['profile'],a['replicate']) for a in assignment}!=set(grouped):
        raise ValueError('Pair assignment differs from planned cells')
    inputs=[];mapping={};missing=[]
    contexts={}
    retrieval=load(data/'retrieval.json')
    for item in assignment:
        key=(item['case_id'],item['profile'],item['replicate'])
        pair=grouped[key];ident=item['id']
        if not all(pair.get(method,{}).get('observed') for method in METHODS):
            missing.append(dict(id=ident,case_id=key[0],profile=key[1],replicate=key[2],
                statuses={method:pair.get(method,{}).get('status','missing') for method in METHODS}));continue
        a,b=item['a_method'],item['b_method']
        case=pair[a]['case'];context_id=case['entity']
        if item['context_id']!=context_id:raise ValueError('Assigned anonymous context differs')
        retrieved=retrieval[case['id']]
        documents=retrieved.get('visible_documents',[{k:doc[k] for k in ('id','subject','date','text')} for doc in retrieved['documents']])
        contexts[context_id]=dict(question=case['user'],documents=documents,kind=case['kind'],necessary_conditions=case['required_conditions'])
        inputs.append(dict(id=ident,context_id=context_id,profile=key[1],a=pair[a]['record']['say'],b=pair[b]['record']['say']))
        mapping[ident]=dict(case_id=key[0],family=case['family'],kind=case['kind'],profile=key[1],replicate=key[2],
            a=dict(method=a,trial_id=pair[a]['trial_id']),b=dict(method=b,trial_id=pair[b]['trial_id']))
    payload=dict(cases=contexts,profiles=CREATIVE_PROFILES)
    for path,value in [(data/'review_contexts.json',payload),(data/'pair_inputs.json',inputs),(data/'pair_key.json',mapping)]:
        if path.exists() and load(path)!=value:raise ValueError('Refusing to overwrite different anonymous review inputs: '+path.name)
        dump(path,value)
    dump(data/'pair_missing.json',dict(assigned_pairs=len(assignment),exported_pairs=len(inputs),missing_pairs=missing,
        scope='Only complete actual A/B pairs exported; missing outputs are not invented or treated as absence of a problem'))
    return dict(assigned_pairs=len(assignment),exported_pairs=len(inputs),a_method_counts=dict(Counter(value['a']['method'] for value in mapping.values())))


def ratings(data,rows):
    if not (data/'pair_key.json').exists():return dict(status='inputs_not_exported')
    key=load(data/'pair_key.json');by_trial={row['trial_id']:row for row in rows}
    raters={};raw_by_rater={}
    for index in (1,2):
        path=data/f'pair_ratings_{index}.json'
        if not path.exists():continue
        items=load(path);mapping={item['id']:item for item in items}
        if len(mapping)!=len(items) or set(mapping)!=set(key):raise ValueError('Rater IDs do not cover exported complete pairs')
        flat=[];preferences=Counter();preference_groups=defaultdict(Counter);semantic=[]
        for ident,item in mapping.items():
            if item.get('preference') not in ('A','B','tie','unclear'):raise ValueError('Invalid pair preference')
            meta=key[ident]
            preferred=item['preference']
            named_preference=meta[preferred.lower()]['method'] if preferred in ('A','B') else preferred
            preferences[named_preference]+=1
            preference_groups[meta['profile']][named_preference]+=1
            for side in RATING_SIDES:
                value=item[side];profile=meta['profile']
                if profile=='neutral':
                    if value.get('role_fidelity') is not None:raise ValueError('Neutral role fidelity must be null')
                elif type(value.get('role_fidelity')) is not int or not 1<=value['role_fidelity']<=5:raise ValueError('Role fidelity must be 1..5')
                if type(value.get('naturalness')) is not int or not 1<=value['naturalness']<=5:raise ValueError('Naturalness must be 1..5')
                if type(value.get('unsupported_assertion')) is not bool:raise ValueError('Unsupported assertion flag must be boolean')
                for field in ('necessary_conditions','boundary_respected'):
                    allowed=('yes','no','unclear','not_applicable') if field=='boundary_respected' else ('yes','no','not_applicable')
                    if value.get(field) not in allowed:raise ValueError('Invalid categorical rating: '+field)
                if not isinstance(value.get('target_answer'),str):raise ValueError('Target answer extraction must be a string')
                if value.get('target_verdict') not in ('correct','incorrect','no_answer','ambiguous','not_applicable'):raise ValueError('Invalid requested-conclusion verdict')
                source=by_trial[meta[side]['trial_id']]
                entry=dict(**meta[side],id=ident,side=side,profile=profile,case_id=meta['case_id'],family=meta['family'],kind=meta['kind'],replicate=meta['replicate'],**value)
                flat.append(entry)
                if meta['kind']!='open':
                    answer=value['target_answer'].strip().rstrip('。.!！;；').replace(' ','')
                    gold=source['case']['gold']
                    covered=answer not in ('unclear','no_answer','ambiguous','not_applicable','')
                    if source['case']['kind']=='number':
                        from decimal import Decimal,InvalidOperation
                        if answer.endswith('元'):answer=answer[:-1]
                        try:correct=covered and Decimal(answer)==Decimal(gold)
                        except InvalidOperation:correct=False
                    else:correct=covered and answer==gold
                    semantic.append(dict(trial_id=entry['trial_id'],method=entry['method'],profile=profile,family=meta['family'],
                        case_id=meta['case_id'],replicate=meta['replicate'],
                        extracted=covered,extracted_exact_match=bool(correct),verdict=value['target_verdict']))
        groups={}
        for method in METHODS:
            for profile in PROFILES:
                selected=[row for row in flat if row['method']==method and row['profile']==profile]
                groups[method+'_'+profile]=dict(labelled=len(selected),role_fidelity=number([row['role_fidelity'] for row in selected]),
                    naturalness=number([row['naturalness'] for row in selected]),unsupported_assertion=proportion(sum(row['unsupported_assertion'] for row in selected),len(selected)),
                    necessary_conditions=dict(Counter(row['necessary_conditions'] for row in selected)),boundary_respected=dict(Counter(row['boundary_respected'] for row in selected)))
        sem_groups={}
        for method in METHODS:
            for profile in PROFILES:
                selected=[row for row in semantic if row['method']==method and row['profile']==profile]
                assigned=sum(row['kind']!='open' and row['method']==method and row['profile']==profile for row in rows)
                verdicts=Counter(row['verdict'] for row in selected)
                sem_groups[method+'_'+profile]=dict(assigned=assigned,labelled=len(selected),verdict_counts=dict(verdicts),
                    extracted=proportion(sum(row['extracted'] for row in selected),assigned),
                    target_supported=proportion(verdicts['correct'],assigned),
                    known_incorrect=proportion(verdicts['incorrect'],assigned),
                    correct_within_decidable=proportion(verdicts['correct'],verdicts['correct']+verdicts['incorrect']),
                    extraction_exact_match=proportion(sum(row['extracted_exact_match'] for row in selected),assigned),
                    note='Auxiliary conclusion extraction from the visible answer, not full semantic truth; original automatic labels unchanged')
        paired={}
        for profile in PROFILES:
            for field in ('role_fidelity','naturalness'):
                if profile=='neutral' and field=='role_fidelity':continue
                by_family=defaultdict(list)
                for ident,item in mapping.items():
                    meta=key[ident]
                    if meta['profile']!=profile:continue
                    values={meta[side]['method']:item[side][field] for side in RATING_SIDES}
                    by_family[meta['family']].append(values['framework']-values['traditional'])
                paired[profile+'_'+field+'_framework_minus_traditional']=bootstrap({family:statistics.mean(values) for family,values in sorted(by_family.items())})
        semantic_by_trial={value['trial_id']:value for value in semantic}
        facts=[row for row in rows if row['kind']!='open'];families=sorted({row['family'] for row in facts})
        semantic_scores={(row['method'],row['profile'],row['case_id'],row['replicate']):int(semantic_by_trial.get(row['trial_id'],{}).get('verdict')=='correct') for row in facts}
        semantic_contrasts={}
        for profile in ('blue','dragon'):
            for method in METHODS:
                clusters={family:statistics.mean(semantic_scores[(method,'neutral',row['case_id'],row['replicate'])]-semantic_scores[(method,profile,row['case_id'],row['replicate'])]
                    for row in facts if row['method']==method and row['profile']==profile and row['family']==family) for family in families}
                semantic_contrasts[f'persona_labelled_target_loss_{method}_{profile}']=bootstrap(clusters)
            semantic_contrasts[f'persona_loss_framework_minus_traditional_{profile}']=bootstrap({family:
                semantic_contrasts[f'persona_labelled_target_loss_framework_{profile}']['cluster_values'][family]-semantic_contrasts[f'persona_labelled_target_loss_traditional_{profile}']['cluster_values'][family] for family in families})
        for profile in PROFILES:
            semantic_contrasts[f'target_framework_minus_traditional_{profile}']=bootstrap({family:statistics.mean(
                semantic_scores[('framework',profile,row['case_id'],row['replicate'])]-semantic_scores[('traditional',profile,row['case_id'],row['replicate'])]
                for row in facts if row['method']=='framework' and row['profile']==profile and row['family']==family) for family in families})
        raters[str(index)]=dict(pairs=len(items),groups=groups,preferences=dict(preferences),preferences_by_profile={k:dict(v) for k,v in preference_groups.items()},
                               target_extraction=sem_groups,target_verdict_contrasts=semantic_contrasts,paired_rating_differences=paired)
        raw_by_rater[str(index)]=mapping
    agreement={};applicable_agreement={}
    if len(raw_by_rater)==2:
        comparisons={field:[] for field in ('role_fidelity','naturalness','target_verdict','unsupported_assertion','necessary_conditions','boundary_respected','preference')}
        applicable={field:[] for field in ('target_verdict','necessary_conditions','boundary_respected')}
        for ident in key:
            a,b=raw_by_rater['1'][ident],raw_by_rater['2'][ident]
            comparisons['preference'].append(a['preference']==b['preference'])
            for side in RATING_SIDES:
                for field in comparisons:
                    if field=='preference' or a[side][field] is None and b[side][field] is None:continue
                    comparisons[field].append(a[side][field]==b[side][field])
                    if field in applicable and not (a[side][field]=='not_applicable' and b[side][field]=='not_applicable'):
                        applicable[field].append(a[side][field]==b[side][field])
        agreement={field:proportion(sum(values),len(values)) for field,values in comparisons.items()}
        applicable_agreement={field:proportion(sum(values),len(values)) for field,values in applicable.items()}
    return dict(status='complete' if len(raters)==2 else 'partial' if raters else 'awaiting_ratings',
        exported_pairs=len(key),raters=raters,exact_label_agreement=agreement,exact_agreement_on_applicable=applicable_agreement,
        scope='Two condition-anonymous Codex tasks; neutral fidelity unscored. Exploratory ratings, not human truth or persona noninferiority.')


def cell(value):return f"{value['correct']}/{value['total']}"
def percent(value):return '未定义' if value is None else f'{value*100:.1f}%'
def interval(value):return '未定义' if value is None else '['+', '.join(f'{x*100:.1f}' for x in value)+']'
def tables(summary):
    lines=['# 共享检索资料的公平对照：离线统计','',
        '主自动指标只评价固定可提取目标；缺失、覆盖与格式分开。条件匿名辅助编码不替代原始评分或人类评价。','',
        '| 方法 / 配置 | 计划/返回 | 内容覆盖 | 已分配目标命中下界 | 标签存在 | 末行格式 | 联合严格 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,group in summary['facts'].items():
        lines.append('| '+name+f" | {group['assigned']}/{group['observed']} | "+' | '.join(cell(group[field]) for field in METRICS)+' |')
    lines.extend(['','## 目标覆盖内正确及未覆盖界限','',
        '| 方法 / 配置 | 覆盖内正确 | 已覆盖明确不匹配数 | 未评分目标数 | 已分配目标成功可能区间 |','| --- | ---: | ---: | ---: | --- |'])
    for name,value in summary['facts'].items():lines.append('| '+name+' | '+cell(value['target_correct_within_covered'])+f" | {value['known_incorrect']} | {value['unscored_target']} | "+interval(value['assigned_target_success_bounds'])+' |')
    lines.extend(['','## 配置相对中性的可提取命中分数损失与方法差异','',
        '损失为本方法中性命中分数减人格命中分数；交互为框架损失减传统损失。正交互表示框架的下降更大，不称为神经推理污染。未覆盖是未评分，命中下界差异不能直接解释为推理错误差异。','',
        '| 对比 | 百分点估计 | 95% 八族聚类bootstrap描述区间 | 未评分识别界限 | 簇数 |','| --- | ---: | --- | --- | ---: |'])
    for name,value in summary['contrasts'].items():lines.append('| '+name+' | '+f"{value['estimate']*100:.1f}"+' | '+interval(value['interval_95'])+' | '+interval(value.get('unscored_identification_bounds'))+f" | {value['clusters']} |")
    lines.extend(['','## 请求、成本与实际输出','',f"状态：`{summary['status']}`；实际请求{summary['calls']['actual']}；完整输出{summary['design']['observed_outputs']}/{summary['design']['assigned_units']}；单元异常{summary['design']['unit_errors']}。",'',
        '| 方法 / 配置 | 请求 | 已知用量条数 | 平均输入tokens | 平均输出tokens | 完整输出均秒 |','| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name,value in summary['calls_by_group'].items():
        mean=lambda x:'未得' if x is None else f'{x:.2f}'
        facts=summary['facts'][name]
        lines.append('| '+name+f" | {value['actual']} | {value['usage_present']} | "+mean(value['token_means_known']['prompt_tokens'])+' | '+mean(value['token_means_known']['completion_tokens'])+' | '+mean(facts['elapsed_s']['mean'])+' |')
    fw=summary['framework']
    lines.extend(['',f"框架协议：{cell(fw['protocol_valid'])}（计划），{cell(fw['protocol_on_observed'])}（返回）；交付一致：{cell(fw['delivery_on_observed'])}（返回）。传统提示无相同三段协议，不计入协议分母。",'',
        '资料包是实体过滤后的全部六份资料，词法排序只影响顺序；不将相同证据输入解释为检索算法质量或压缩优势。','',
        f"材料对应检查：{summary['audit']['checks']}，失败{summary['audit']['failed']}。",'',
        '## 条件匿名配对评价','',f"状态：`{summary.get('paired_ratings',{}).get('status','not_exported')}`。"])
    ratings_summary=summary.get('paired_ratings',{})
    for rater,value in ratings_summary.get('raters',{}).items():
        lines.extend(['',f'### 评价任务 {rater}','',
            '| 方法 / 配置 | 已标回复 | 角色一致性均分 | 自然度均分 | 无依据断言 | 边界标签 yes/no/unclear/不适用 |',
            '| --- | ---: | ---: | ---: | ---: | --- |'])
        for name,group in value['groups'].items():
            show=lambda v:'不评分' if v is None else f'{v:.2f}'
            boundary=group['boundary_respected']
            lines.append('| '+name+f" | {group['labelled']} | "+show(group['role_fidelity']['mean'])+' | '+show(group['naturalness']['mean'])+' | '+cell(group['unsupported_assertion'])+' | '+str([boundary.get(k,0) for k in ('yes','no','unclear','not_applicable')])+' |')
        lines.extend(['', '全部配对偏好计数：`'+json.dumps(value['preferences'],ensure_ascii=False)+'`。含中性配对，允许平局与不确定；均分不能证明人格非劣。','',
            '| 配置 | 传统偏好 | 框架偏好 | 平局 | 无法判断 |','| --- | ---: | ---: | ---: | ---: |'])
        for profile,counts in value['preferences_by_profile'].items():lines.append('| '+profile+' | '+' | '.join(str(counts.get(k,0)) for k in ('traditional','framework','tie','unclear'))+' |')
        lines.extend(['','| 方法 / 配置 | 已分配/已标注 | correct | incorrect | no_answer | ambiguous | 可判断中正确 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for name,group in value['target_extraction'].items():
            counts=group['verdict_counts']
            lines.append('| '+name+f" | {group['assigned']}/{group['labelled']} | "+' | '.join(str(counts.get(k,0)) for k in ('correct','incorrect','no_answer','ambiguous'))+' | '+cell(group['correct_within_decidable'])+' |')
    if ratings_summary.get('exact_label_agreement'):
        lines.extend(['','### 两套标签的精确一致','',
            '| 字段 | 全部标签精确相同 | 排除双方均不适用后的相同 |','| --- | ---: | ---: |'])
        for field,value in ratings_summary['exact_label_agreement'].items():
            applicable=ratings_summary.get('exact_agreement_on_applicable',{}).get(field)
            lines.append('| '+field+' | '+cell(value)+' | '+(cell(applicable) if applicable else '—')+' |')
        lines.extend(['','角色分排除中性null；分类适用分母取至少一套认为适用的回复。大量不适用标签不作为实际语义一致的证据。'])
    return '\n'.join(lines)+'\n'


def plots(summary,out):
    if summary['design']['observed_outputs']==0:return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True);produced=[]
    colors={'traditional':'#b87337','framework':'#217e86'}
    def save(fig,name,note):
        fig.text(.025,.018,note,fontsize=8.5,color='#555555');fig.tight_layout(rect=(0,.065,1,.985))
        fig.savefig(folder/name,dpi=180,metadata={'Software':'persona-duplex fair shared retrieval offline analysis'});plt.close(fig);produced.append('figures/'+name)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,metric,title in zip(axes,('target_success','format_ok'),('可提取目标命中下界','末行结论格式')):
        for method,offset in [('traditional',-.18),('framework',.18)]:
            cells=[summary['facts'][method+'_'+profile][metric] for profile in PROFILES]
            ax.bar([i+offset for i in range(3)],[c['rate'] for c in cells],width=.34,color=colors[method],label=METHOD_LABELS[method])
            for i,c in enumerate(cells):ax.text(i+offset,c['rate']+.025,cell(c),ha='center',fontsize=9)
        ax.set_xticks(range(3),[PROFILE_LABELS[p] for p in PROFILES]);ax.set_ylim(0,1.17);ax.set_yticks([0,.25,.5,.75,1.]);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left',fontsize=8)
    save(fig,'fig1_target_and_format.png','每格32个计划输出；目标覆盖与格式分开，不把缺少标签当全文事实错误；相同资料包不证明推理隔离。')
    fig,ax=plt.subplots(figsize=(9,4.8))
    names=[f'persona_loss_framework_minus_traditional_{profile}' for profile in ('blue','dragon')]
    estimates=[summary['contrasts'][name] for name in names]
    for i,value in enumerate(estimates):
        center=value['estimate'];interval_=value['interval_95'];ax.errorbar(center,i,xerr=[[center-interval_[0]],[interval_[1]-center]],fmt='o',capsize=5,color='#217e86')
        ax.annotate(f'{center*100:.1f}pp',xy=(center,i),xytext=(5,9),textcoords='offset points')
    ax.axvline(0,color='#777777',linestyle='--');ax.set_yticks(range(2),['蓝鱼','白龙']);ax.set_xlabel('框架的配置相对中性目标损失 − 传统提示的对应损失');ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_title('方法与表达配置的描述性交互');ax.grid(axis='x',alpha=.15)
    save(fig,'fig2_neutral_reference_interaction.png','八任务族配对bootstrap描述区间；负值表示框架的配置下降较小，不是神经污染测量或等效性检验。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for method,offset in [('traditional',-.18),('framework',.18)]:
        costs=[summary['calls_by_group'][method+'_'+profile] for profile in PROFILES]
        inputs=[v['token_means_known']['prompt_tokens'] if v['token_means_known']['prompt_tokens'] is not None else float('nan') for v in costs]
        outputs=[v['token_means_known']['completion_tokens'] if v['token_means_known']['completion_tokens'] is not None else float('nan') for v in costs]
        x=[i+offset for i in range(3)];axes[0].bar(x,inputs,width=.34,color=colors[method],alpha=.55,label=METHOD_LABELS[method]+' 输入');axes[0].bar(x,outputs,bottom=inputs,width=.34,color=colors[method],label=METHOD_LABELS[method]+' 输出')
        values=[summary['facts'][method+'_'+p]['elapsed_s']['mean'] if summary['facts'][method+'_'+p]['elapsed_s']['mean'] is not None else float('nan') for p in PROFILES]
        axes[1].bar(x,values,width=.34,color=colors[method],label=METHOD_LABELS[method])
    for ax in axes:ax.set_xticks(range(3),[PROFILE_LABELS[p] for p in PROFILES]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_title('实际返回用量的单请求均值');axes[0].set_ylabel('tokens');axes[0].legend(fontsize=8)
    axes[1].set_title('实际完整事实回复的端到端均时');axes[1].set_ylabel('秒');axes[1].legend(fontsize=8)
    save(fig,'fig3_tokens_and_latency.png','一调用与同输出上限不等于等成本；仅累加已返回用量，缺失不估零；图中均值的覆盖数见统计表。')
    raters=summary.get('paired_ratings',{}).get('raters',{})
    if raters:
        fig,axes=plt.subplots(1,len(raters),figsize=(6*len(raters),4.8),squeeze=False)
        for ax,(index,rater) in zip(axes[0],raters.items()):
            for method,offset in [('traditional',-.18),('framework',.18)]:
                groups=[rater['groups'][method+'_'+profile] for profile in ('blue','dragon')]
                values=[v['role_fidelity']['mean'] for v in groups]
                ax.bar([i+offset for i in range(2)],values,width=.34,color=colors[method],label=METHOD_LABELS[method])
                for i,value in enumerate(values):ax.text(i+offset,value+.06,f'{value:.2f}',ha='center')
            ax.set_xticks(range(2),['蓝鱼','白龙']);ax.set_ylim(0,5.5);ax.set_yticks([0,1,2,3,4,5]);ax.set_title('匿名评价任务 '+index+'：角色一致性');ax.legend(fontsize=8,loc='lower left')
        save(fig,'fig4_exploratory_role_ratings.png','角色分与事实正确性分开；每回复1—5分，两套Codex标签原样保留，不能证明人类体验或人格非劣。')
        fig,axes=plt.subplots(2,len(raters),figsize=(6*len(raters),9),squeeze=False)
        for column,(index,rater) in enumerate(raters.items()):
            ax=axes[0,column]
            names=['traditional','framework','tie','unclear'];counts=[rater['preferences'].get(name,0) for name in names]
            ax.bar(range(4),counts,color=['#b87337','#217e86','#9ba6ad','#d3d5d6'])
            for i,count in enumerate(counts):ax.text(i,count+.4,str(count),ha='center')
            ax.set_xticks(range(4),['传统强提示','默认框架','平局','无法判断']);ax.set_title('匿名评价任务 '+index+'：配对偏好');ax.set_ylabel('配对数')
            ax=axes[1,column]
            for method,offset in [('traditional',-.18),('framework',.18)]:
                groups=[rater['groups'][method+'_'+profile] for profile in PROFILES]
                unsupported=sum(value['unsupported_assertion']['correct'] for value in groups)
                conditions={label:sum(value['necessary_conditions'].get(label,0) for value in groups) for label in ('yes','no')}
                boundary={label:sum(value['boundary_respected'].get(label,0) for value in groups) for label in ('yes','no','unclear')}
                totals=[sum(value['labelled'] for value in groups),sum(conditions.values()),sum(boundary.values())]
                counts=[unsupported,conditions['no'],boundary['no']]
                rates=[count/total if total else float('nan') for count,total in zip(counts,totals)]
                x=[i+offset for i in range(3)];ax.bar(x,rates,width=.34,color=colors[method],label=METHOD_LABELS[method])
                for i,(count,total,rate) in enumerate(zip(counts,totals,rates)):
                    if total:ax.text(i+offset,rate+.018,f'{count}/{total}',ha='center',fontsize=9)
            ax.set_xticks(range(3),['无依据断言','关键条件缺失','明确边界未尊重']);ax.set_ylim(0,1.08);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title('匿名评价任务 '+index+'：语义边界标签');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        save(fig,'fig5_exploratory_pair_preferences.png','上：112对整体偏好（含中性）；下：断言按全部回复、条件/边界按各编码的适用数，未知另表保留。\n两套Codex编码为探索性；无输出不能算边界成功，非独立人类真值或人格非劣检验。')
    return produced


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=HERE);parser.add_argument('--out',type=Path,default=HERE)
    parser.add_argument('--export-pairs',action='store_true');parser.add_argument('--no-plots',action='store_true');args=parser.parse_args()
    summary,rows=collect(args.data)
    if args.export_pairs:summary['pair_export']=export_pairs(args.data,rows)
    summary['paired_ratings']=ratings(args.data,rows)
    summary['figures']=[] if args.no_plots else plots(summary,args.out)
    dump(args.out/'summary.json',summary);(args.out/'result_tables.md').write_text(tables(summary),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=summary['status'],assigned=summary['design']['assigned_units'],observed=summary['design']['observed_outputs'],calls=summary['calls']['actual'],checks=summary['audit']['checks'],figures=len(summary['figures']),network=False)))


if __name__=='__main__':main()
