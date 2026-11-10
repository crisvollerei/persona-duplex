"""Offline descriptive analysis. Never generates, repairs, or merges ratings."""
from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics

import benchmark as b
import gate
import mechanism as m


FIELDS = tuple(m.LABEL_VALUES)
METHODS = ('program', 'prompt')
PROFILES = ('shorekeeper', 'lan')
PROTECTED = ('draft_id', 'draft_title', 'draft_status', 'draft_revision', 'advice_permission')
DISPLAY = {'program': '程序选择', 'prompt': '强提示', 'shorekeeper': '守岸人', 'lan': '岚'}


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def counter(values):
    return dict(Counter(str(v) for v in values))


def delta(a, z):
    return a-z if a is not None and z is not None else None


def ratings(path):
    obj = b.load(path)
    return obj['ratings'] if isinstance(obj, dict) else obj


def calibration():
    key = b.load(b.HERE / 'calibration_key.json')
    inputs = b.load(b.HERE / 'calibration_inputs.json')
    assessments, judgments = {}, {}
    for coder in ('1', '2'):
        path = b.HERE / 'calibration_reviews' / ('ratings_' + coder + '.json')
        if not path.exists():
            continue
        name = '评审' + coder
        rows = ratings(path)
        assessments[name] = gate.assess(rows, key)
        judgments[name] = {r['id']: r for r in rows}
    results = b.HERE / 'calibration_run/results.json'
    if results.exists():
        for record in b.load(results)['records']:
            name = '端点重复' + str(record['repetition'])
            if record['status'] == 'ok':
                assessments[name] = gate.assess(record['ratings'], key)
                judgments[name] = {r['id']: r for r in record['ratings']}
            else:
                assessments[name] = dict(status=record['status'])
    clear = [id for id, row in key.items() if not row.get('probe', False)]
    probes = [id for id in key if id not in clear]
    probe_counts = {name: {field: counter(rs[id][field] for id in probes) for field in FIELDS}
                    for name, rs in judgments.items()}
    mutual = {}
    for a, z, name in (('评审1', '评审2', 'automated_reviewers'),
                        ('端点重复1', '端点重复2', 'endpoint_repeats')):
        if a in judgments and z in judgments:
            mutual[name] = {field: dict(agree=sum(judgments[a][id][field] == judgments[z][id][field] for id in key),
                                       denominator=len(key)) for field in FIELDS}
    return dict(planned_items=len(inputs), clear_n=len(clear), probe_n=len(probes),
                assessments=assessments, probe_counts=probe_counts, mutual_agreement=mutual,
                interpretation='Author-expectation checks; probes do not have a forced semantic truth.')


def aggregate_execution(rows):
    protected_changes = []
    for r in rows:
        changed = [key for key in PROTECTED
                   if r['state_after_user'].get(key) != r['state_after_action'].get(key)]
        if changed:
            protected_changes.append(dict(id=r['id'], keys=changed))
    return dict(planned=len(rows), ok=sum(r['status'] == 'ok' for r in rows),
                allowed=sum(bool(r.get('allowed')) for r in rows),
                request_scope_match=sum(bool(r.get('request_scope_match')) for r in rows),
                rule_match=sum(bool(r.get('rule_match')) for r in rows),
                executed=sum(r.get('executed_id') is not None for r in rows),
                status=counter(r['status'] for r in rows),
                execution_reason=counter(r['execution_reason'] for r in rows),
                action_ids=counter(r.get('model_reported_id') for r in rows),
                action_state_updates=sum(r['state_after_user'] != r['state_after_action'] for r in rows),
                confirmed_user_updates=sum(r['state_before'] != r['state_after_user'] for r in rows),
                protected_model_changes=protected_changes)


def aggregate_rating(rows):
    return dict(n=len(rows),
                labels={field: counter(r[field] for r in rows) for field in FIELDS},
                persona_fit=dict(mean=mean([r['persona_fit'] for r in rows]),
                                 observed=sum(r['persona_fit'] is not None for r in rows),
                                 distribution=counter(r['persona_fit'] for r in rows)),
                context_fit=dict(mean=mean([r['context_fit'] for r in rows]),
                                 observed=sum(r['context_fit'] is not None for r in rows),
                                 distribution=counter(r['context_fit'] for r in rows)),
                focus=counter(r['focus'] for r in rows))


def candidate(row, id):
    return next((a for a in row['payload']['candidates'] if a['id'] == id), None)


def selection_diagnostics(rows):
    groups = defaultdict(list)
    details = []
    for row in rows:
        expected = candidate(row, row.get('program_selected_id'))
        reported = candidate(row, row.get('model_reported_id'))
        entry = dict(id=row['id'], method=row['method'], profile=row['profile'], turn=row['turn'],
            id_match=bool(expected and reported and expected['id'] == reported['id']),
            known_pair=expected is not None and reported is not None,
            same_kind=bool(expected and reported and expected['kind'] == reported['kind']),
            expected_id=expected['id'] if expected else None, reported_id=reported['id'] if reported else None,
            expected_kind=expected['kind'] if expected else None, reported_kind=reported['kind'] if reported else None,
            expected_focus=expected['focus'] if expected else None, reported_focus=reported['focus'] if reported else None)
        groups[(row['method'], str(row['turn']))].append(entry)
        if not entry['id_match']:
            details.append(entry)
    def tally(rs):
        mismatched = [r for r in rs if not r['id_match']]
        return dict(n=len(rs), known_pairs=sum(r['known_pair'] for r in rs),
                    exact_id_match=sum(r['id_match'] for r in rs),
                    same_author_kind=sum(r['same_kind'] for r in rs), id_mismatch=len(mismatched),
                    same_kind_among_mismatch=sum(r['same_kind'] for r in mismatched),
                    different_kind_among_mismatch=sum(r['known_pair'] and not r['same_kind'] for r in mismatched),
                    unknown_pair_among_mismatch=sum(not r['known_pair'] for r in mismatched))
    return dict(by_method_turn={'/'.join(k): tally(v) for k,v in groups.items()},
                by_method={method:tally([r for values in groups.values() for r in values if r['method']==method]) for method in METHODS},
                mismatch_details=details,
                interpretation='IDs and kinds are author metadata. Same-kind alternatives can be semantically similar; ID mismatch is not automatically a behavioral or quality error.')


def focus_diagnostics(joined):
    details, groups = [], defaultdict(list)
    for source, rating in joined:
        selected = candidate(source, source.get('model_reported_id'))
        authored = selected['focus'] if selected else None
        observed = rating['focus']
        single = observed in ('practical', 'feeling', 'detail') and authored is not None
        entry = dict(id=source['id'], method=source['method'], profile=source['profile'], turn=source['turn'],
                     review_id=rating['id'], author_focus=authored, rated_focus=observed,
                     comparable_single_focus=single, exact_match=single and authored == observed,
                     reason=rating['reason'])
        details.append(entry)
        groups[(source['method'], source['profile'], str(source['turn']))].append(entry)
    def tally(rs):
        singles = [r for r in rs if r['comparable_single_focus']]
        return dict(n=len(rs), author_focus=counter(r['author_focus'] for r in rs),
                    independent_focus=counter(r['rated_focus'] for r in rs),
                    mixed=sum(r['rated_focus']=='mixed' for r in rs),
                    unclear=sum(r['rated_focus']=='unclear' for r in rs),
                    comparable_single_focus=len(singles), exact_single_focus_match=sum(r['exact_match'] for r in singles),
                    single_focus_disagreement=sum(not r['exact_match'] for r in singles))
    return dict(overall=tally(details),
        by_method_profile_turn={'/'.join(k):tally(v) for k,v in groups.items()},
        by_method_profile={method+'/'+profile:tally([r for r in details if r['method']==method and r['profile']==profile])
                           for method in METHODS for profile in PROFILES},
        by_turn={str(turn):tally([r for r in details if r['turn']==turn]) for turn in sorted({r['turn'] for r in details})},
        disagreement_details=[r for r in details if r['comparable_single_focus'] and not r['exact_match']],
        interpretation='Agreement between author-assigned selected-candidate focus and independent reading of actual words, not label accuracy. Mixed/unclear are separate, not forced into a single-focus error.')


def analysis_main():
    path = b.HERE / 'run/results.json'
    if not path.exists():
        return None, []
    data = b.load(path)
    rows = data['records']
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['profile'], str(row['turn']))].append(row)
    executions = {'/'.join(key): aggregate_execution(rs) for key, rs in groups.items()}
    by_method = {method: aggregate_execution([r for r in rows if r['method'] == method]) for method in METHODS}
    final = {}
    for method in METHODS:
        for profile in PROFILES:
            episodes = [e for e in data['episodes'] if e['method'] == method and e['profile'] == profile]
            final[method + '/' + profile] = dict(planned_episodes=len(episodes),
                complete=sum(e['complete'] for e in episodes),
                final_revision=counter(e['final_state'].get('draft_revision') for e in episodes),
                final_permission=counter(e['final_state'].get('advice_permission') for e in episodes),
                acknowledged_revision=counter(e['final_state'].get('acknowledged_revision') for e in episodes),
                final_states=[dict(id=e['id'], state=e['final_state']) for e in episodes])
    result = dict(planned_outputs=len(rows), planned_episodes=len(data['episodes']),
        scenario_template_instances=len({r['case'] for r in rows}),
        semantic_eligible=data.get('semantic_eligible'), execution_by_method=by_method,
        execution_groups=executions, final_state_groups=final,
        selection_diagnostics=selection_diagnostics(rows), reviews={}, mutual_agreement=None,
        interpretation='Six surface instances of one authored four-turn template, not six independent tasks or 192 independent situations; state may diverge by route.')
    key_path = b.HERE / 'review_key.json'
    if not key_path.exists():
        return result, rows
    mapping = b.load(key_path)
    ids = sorted(set(mapping.values()))
    reviews = {}
    for coder in ('1', '2'):
        path = b.HERE / 'reviews' / ('ratings_' + coder + '.json')
        if not path.exists():
            result['reviews'][coder] = dict(status='not_available')
            continue
        rs = m.validate_labels({'ratings': ratings(path)}, ids, main=True)
        by_id = {r['id']: r for r in rs}
        reviews[coder] = by_id
        joined = [(r, by_id[mapping[r['id']]]) for r in rows if r['id'] in mapping]
        rating_groups, overall_groups, episode_ratings, case_ratings = (defaultdict(list) for _ in range(4))
        for source, rating in joined:
            rating_groups[(source['method'], source['profile'], str(source['turn']))].append(rating)
            overall_groups[(source['method'], source['profile'])].append(rating)
            episode_id = source['id'].rsplit('-', 1)[0]
            episode_ratings[episode_id].append(rating)
            case_ratings[(source['case'], source['method'], source['profile'])].append(rating)
        result['reviews'][coder] = dict(status='complete', unique_review_units=len(rs), mapped_outputs=len(joined),
            overall=aggregate_rating([r for _, r in joined]),
            focus_diagnostics=focus_diagnostics(joined),
            groups={'/'.join(k): aggregate_rating(v) for k,v in rating_groups.items()},
            by_method_profile={'/'.join(k): aggregate_rating(v) for k,v in overall_groups.items()},
            by_case_method_profile={'/'.join(k): aggregate_rating(v) for k,v in case_ratings.items()},
            episode_scores={k: dict(observed_turns=len(v), persona_mean=mean([r['persona_fit'] for r in v]),
                                    context_mean=mean([r['context_fit'] for r in v])) for k,v in episode_ratings.items()})
        paired = []
        for case in sorted({r['case'] for r in rows}):
            for profile in PROFILES:
                a = case_ratings[(case, 'program', profile)]
                z = case_ratings[(case, 'prompt', profile)]
                paired.append(dict(case=case, profile=profile,
                    program_n=len(a), prompt_n=len(z),
                    persona_delta=delta(mean([r['persona_fit'] for r in a]),mean([r['persona_fit'] for r in z])),
                    context_delta=delta(mean([r['context_fit'] for r in a]),mean([r['context_fit'] for r in z]))))
        result['reviews'][coder]['paired_case_differences'] = paired
    if len(reviews) == 2:
        result['mutual_agreement'] = {field: dict(agree=sum(reviews['1'][id][field] == reviews['2'][id][field] for id in ids),
                                                denominator=len(ids)) for field in (*FIELDS, 'persona_fit', 'context_fit', 'focus')}
        result['mapped_outputs'] = len(mapping)
        result['unique_review_units'] = len(ids)
    return result, rows


def costs(rows):
    lookup = {r['id']: r for r in rows}
    calls = []
    for stage in ('calibration_run', 'run'):
        for path in sorted((b.HERE / stage / 'calls').glob('*.json')):
            record = b.load(path)
            source = lookup.get(record['id'], {})
            calls.append(dict(id=record['id'], stage=stage, method=source.get('method'), profile=source.get('profile'),
                episode=record['id'].rsplit('-', 1)[0] if stage == 'run' else None,
                status=record['status'], elapsed_s=record.get('elapsed_s'),
                response_model=record.get('response', {}).get('model', 'unknown'),
                usage=record.get('response', {}).get('usage', {})))
    def total(rs):
        return dict(actual_requests=len(rs), statuses=counter(r['status'] for r in rs),
            response_models=counter(r['response_model'] for r in rs),
            usage_observed=sum('total_tokens' in r['usage'] for r in rs),
            prompt_tokens=sum(r['usage'].get('prompt_tokens',0) for r in rs),
            completion_tokens=sum(r['usage'].get('completion_tokens',0) for r in rs),
            total_tokens=sum(r['usage'].get('total_tokens',0) for r in rs),
            request_time_sum_s=sum(r['elapsed_s'] or 0 for r in rs),
            request_time_mean_s=mean([r['elapsed_s'] for r in rs]))
    result = dict(overall=total(calls), stages={s:total([r for r in calls if r['stage']==s]) for s in ('calibration_run','run')},
        method={}, method_profile={}, episodes={},
        interpretation='Only recorded endpoint usage; external automated review cost excluded. Times sum calls, not user-wait wall time.')
    result['actual_requests'] = result['overall']['actual_requests']
    result['total_tokens'] = result['overall']['total_tokens']
    for method in METHODS:
        result['method'][method] = total([r for r in calls if r['method']==method])
        for profile in PROFILES:
            selected = [r for r in calls if r['method']==method and r['profile']==profile]
            value = total(selected)
            episodes = {r['episode'] for r in selected}
            value['episodes_with_calls'] = len(episodes)
            value['tokens_per_episode_with_calls'] = value['total_tokens']/len(episodes) if episodes else None
            value['request_time_sum_per_episode_s'] = value['request_time_sum_s']/len(episodes) if episodes else None
            result['method_profile'][method + '/' + profile] = value
    by_episode = defaultdict(list)
    for row in calls:
        if row['episode'] is not None:
            by_episode[row['episode']].append(row)
    result['episodes'] = {key:total(rs) for key,rs in by_episode.items()}
    return result


def counterfactual(rows):
    profiles = {p['id']: p for p in b.load(b.HERE/'profiles.json')}
    records = []
    for row in rows:
        data = deepcopy(row['payload'])
        baseline = m.choose(data)
        low, high, swapped = (deepcopy(data) for _ in range(3))
        low['persona']['intervention']=0
        high['persona']['intervention']=1
        other = next(p for id,p in profiles.items() if id != row['profile'])
        swapped['persona'] = deepcopy(other)
        preserved = ('facts','state','allowed_actions','request_scope','context','user','requested','candidates','history','draw')
        assert all(data[k] == cf[k] for k in preserved for cf in (low,high,swapped))
        choices = dict(intervention_0=m.choose(low), intervention_1=m.choose(high),
                       persona_swap=m.choose(swapped), ignore_request_scope=m.choose(data,contextual=False))
        records.append(dict(id=row['id'],case=row['case'],profile=row['profile'],method=row['method'],turn=row['turn'],
                            baseline=baseline, choices=choices,
                            intervention_changes=choices['intervention_0'] != choices['intervention_1'],
                            persona_swap_changes=choices['persona_swap'] != baseline,
                            request_scope_ablation_changes=choices['ignore_request_scope'] != baseline,
                            preserved_factual_inputs=True,
                            payload_sha256=hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True).encode()).hexdigest()))
    by_turn = {}
    for turn in sorted({r['turn'] for r in records}):
        rs = [r for r in records if r['turn']==turn]
        by_turn[str(turn)] = dict(replayed_records=len(rs),
            intervention_changes=sum(r['intervention_changes'] for r in rs),
            persona_swap_changes=sum(r['persona_swap_changes'] for r in rs),
            request_scope_ablation_changes=sum(r['request_scope_ablation_changes'] for r in rs))
    return dict(records=records, by_turn=by_turn,
                interpretation='Offline same-payload deterministic replay; duplicated route states are not new independent observations or generated replies.')


def figures(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'svg.hashsalt':'situated-policy-v2'})
    folder = out/'figures';folder.mkdir(parents=True,exist_ok=True)
    colors = ['#286f92','#60a7ae','#d98b45','#b85b54']
    def save(fig, name):
        for suffix in ('png','svg'):
            fig.savefig(folder/(name+'.'+suffix),dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
    assessments=summary['calibration']['assessments'];x=np.arange(6)
    for i,(name,row) in enumerate(assessments.items()):
        if 'clear_field_correct' not in row:
            continue
        ax.bar(x+(i-1.5)*.19,[row['clear_field_correct'][f]/row['clear_n'] for f in FIELDS],.18,
               label=name,color=colors[i%len(colors)])
    ax.set_xticks(x,['请求存在','请求履行','额外议程','边界类型','边界遵守','事实逻辑'])
    ax.set_ylim(0,1.15);ax.set_ylabel('符合作者预期的比例（明确子集 n=20）')
    ax.set_title('统一说明下的明确材料校准；歧义探针4项单列')
    ax.legend(ncol=4,loc='upper center',fontsize=9)
    save(fig,'fig1_calibration')
    main=summary['main']
    if main is None:
        return
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    metrics=('ok','allowed','request_scope_match','rule_match','executed');x=np.arange(len(metrics))
    for i,method in enumerate(METHODS):
        group=main['execution_by_method'][method]
        values=[group[f]/group['planned'] for f in metrics]
        bars=ax.bar(x+(i-.5)*.32,values,.3,label=DISPLAY[method],color=colors[i*2])
        for bar,f in zip(bars,metrics):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,
                    f"{group[f]}/{group['planned']}",ha='center',fontsize=8)
    ax.set_xticks(x,['完整解析','允许行动','满足请求范围','符合共享选择规则','接受状态执行'])
    ax.set_ylim(0,1.16);ax.set_ylabel('计划生成单元中的比例')
    ax.set_title('模型报告的行动与共同执行器；不代替实际文字评价');ax.legend(loc='lower right')
    save(fig,'fig2_choice_execution')
    complete={c:r for c,r in main['reviews'].items() if r.get('status')=='complete'}
    if not complete:
        return
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    for pi,profile in enumerate(PROFILES):
        ax=axes[pi]
        for coder,review in complete.items():
            for mi,method in enumerate(METHODS):
                vals=[];labels=[]
                for turn in ('T1','T2','T3','T4'):
                    group=review['groups'].get(method+'/'+profile+'/'+turn)
                    labels.append(turn)
                    vals.append(group['labels']['extension'].get('yes',0)/group['n'] if group else np.nan)
                ax.plot(labels,vals,marker='o' if coder=='1' else 's',linestyle='-' if coder=='1' else '--',
                        color=colors[mi*2],label='评审'+coder+'/'+DISPLAY[method])
        ax.set_ylim(-.04,1.12);ax.set_title(DISPLAY[profile]);ax.set_ylabel('实际回复新增议程的比例')
        ax.set_xlabel('T1开放；T2明确求助；T3禁建议问答；T4更正确认')
        ax.legend(fontsize=8,loc='upper right')
    save(fig,'fig3_actual_extension')
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),layout='constrained')
    for j,metric in enumerate(('persona_fit','context_fit')):
        labels,values=[],[]
        for coder,review in complete.items():
            for profile in PROFILES:
                for method in METHODS:
                    labels.append('评审'+coder+'/'+DISPLAY[profile]+'/'+DISPLAY[method])
                    value=review['by_method_profile'][method+'/'+profile][metric]['mean']
                    values.append(value if value is not None else np.nan)
        axes[j].barh(labels,values,color=[colors[0],colors[2]]*4)
        axes[j].set_xlim(0,5);axes[j].set_title('人格符合度' if j==0 else '情境适切性')
        axes[j].set_xlabel('等级均值；各维独立，非综合质量分')
    labels,values=[],[]
    for profile in PROFILES:
        for method in METHODS:
            labels.append(DISPLAY[profile]+'/'+DISPLAY[method])
            values.append(summary['costs']['method_profile'][method+'/'+profile]['tokens_per_episode_with_calls'])
    axes[2].barh(labels,values,color=[colors[0],colors[2]]*2)
    axes[2].set_title('生成端点成本');axes[2].set_xlabel('四轮轨迹平均 token（不含外部评审）')
    save(fig,'fig4_quality_cost')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=b.HERE);args=parser.parse_args()
    b.verify()
    main_result,rows=analysis_main()
    replay=counterfactual(rows)
    context_protocol=b.HERE/'review_context_protocol.json'
    summary=dict(protocol_sha256=b.sha(b.HERE/'protocol.json'),
                 review_context_protocol_sha256=b.sha(context_protocol) if context_protocol.exists() else None,
                 calibration=calibration(),gate=b.load(b.HERE/'calibration_gate.json'),
                 main=main_result,costs=costs(rows),counterfactual_by_turn=replay['by_turn'])
    b.save(args.out/'summary.json',summary)
    b.save(args.out/'counterfactual_replay.json',replay)
    import matplotlib,numpy
    b.save(args.out/'analysis_environment.json',dict(python=platform.python_version(),
        matplotlib=matplotlib.__version__,numpy=numpy.__version__))
    figures(summary,args.out)
    print(json.dumps(dict(actual_requests=summary['costs']['overall']['actual_requests'],
        main_outputs=main_result['planned_outputs'] if main_result else 0,
        review_status={k:v['status'] for k,v in main_result['reviews'].items()} if main_result else {}),ensure_ascii=False))


if __name__=='__main__':
    main()
