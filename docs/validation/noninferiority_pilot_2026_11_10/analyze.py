"""Recompute prespecified pilot summaries from archived observations only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, t

HERE = Path(__file__).resolve().parent
METHODS = ('framework', 'strong_prompt')
CODERS = ('1', '2')
QUALITY = ('persona_fit', 'context_fit')
MARGIN = .3
ALPHA = .05
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 2026111007
RISK_MARGIN = .05
BAD = dict(facts='conflict', boundary='violated', fulfillment='failed')
UNCERTAIN = dict(facts='unknown', boundary='unclear', fulfillment='unclear')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def exact_upper(events, n, alpha=ALPHA):
    """One-sided Clopper-Pearson upper bound, including zero events."""
    if not 0 <= events <= n:
        raise ValueError('Invalid binomial counts')
    if n == 0:
        return None
    if events == n:
        return 1.0
    return float(beta.ppf(1-alpha, events+1, n-events))


def interval(differences):
    values = np.asarray(differences, dtype=float)
    n = len(values)
    if n == 0:
        return dict(n=0, mean_difference=None, t_lower_95=None, t_upper_90=None,
                    bootstrap_lower_95=None, supports_margin=False)
    mean = float(np.mean(values))
    if n < 2:
        lower = upper = bootstrap = None
    else:
        se = float(np.std(values, ddof=1) / np.sqrt(n))
        distance = float(t.ppf(1-ALPHA, n-1)) * se
        lower, upper = mean-distance, mean+distance
        # Called independently for each endpoint; the seed and complete case
        # vector fix the resampling. Each index is an entire situation cluster.
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(0, n, size=(BOOTSTRAP_REPETITIONS, n))
        bootstrap = float(np.quantile(values[indices].mean(axis=1), ALPHA))
    return dict(n=n, mean_difference=mean, t_lower_95=lower, t_upper_90=upper,
                bootstrap_lower_95=bootstrap,
                supports_margin=lower is not None and lower > -MARGIN,
                bootstrap_supports_margin=bootstrap is not None and bootstrap > -MARGIN,
                differences=[float(v) for v in values])


def read_ratings(directory, stage, records):
    """Attach anonymous ratings without treating absent/unknown as success."""
    reviews = directory / 'reviews'
    key_path = reviews / f'{stage}_key.json'
    key = load(key_path) if key_path.exists() else {}
    record_ids = {r['id'] for r in records}
    if len(key) != len(set(key.values())) or any(value not in record_ids for value in key.values()):
        raise ValueError(f'Invalid anonymous mapping: {stage}')
    result = {}
    for coder in CODERS:
        path = reviews / f'{stage}_ratings_{coder}.json'
        ratings = load(path)['ratings'] if path.exists() else []
        seen = set()
        mapped = {}
        for rating in ratings:
            alias = rating['id']
            if alias in seen or alias not in key:
                raise ValueError(f'Unknown or duplicate rating: {stage}/{coder}/{alias}')
            seen.add(alias)
            for metric in QUALITY:
                score = rating.get(metric)
                if score is not None and (type(score) is not int or not 1 <= score <= 5):
                    raise ValueError(f'Invalid ordinal rating: {alias}/{metric}')
            for dimension, allowed in (
                    ('facts', ('supported', 'conflict', 'unknown')),
                    ('boundary', ('respected', 'violated', 'not_applicable', 'unclear')),
                    ('fulfillment', ('fulfilled', 'failed', 'not_applicable', 'unclear'))):
                if rating.get(dimension) not in allowed:
                    raise ValueError(f'Invalid behavioral rating: {alias}/{dimension}')
            mapped[key[alias]] = rating
        result[coder] = mapped
    return result


def quality_score(rating, metric, method):
    value = rating.get(metric) if rating is not None else None
    # Worst-case missing-value sensitivity is the primary conservative view:
    # a missing framework score is 1; a missing comparator score is 5.
    return (value, False) if type(value) is int else (1 if method == 'framework' else 5, True)


def _raw_calls(directory):
    rows = []
    for stage in ('development', 'run'):
        for path in sorted((directory / stage / 'calls').glob('*.json')):
            rows.append((stage, load(path)))
    return rows


def cost_summary(directory, records):
    raw = _raw_calls(directory)
    id_to_record = {r.get('call_id'): r for r in records if r.get('call_id')}
    totals = defaultdict(list)
    by_method = {method: defaultdict(list) for method in METHODS}
    statuses, stage_counts, observed_models = Counter(), Counter(), Counter()
    for stage, call in raw:
        stage_counts[stage] += 1
        statuses[call.get('status', 'unknown')] += 1
        response = call.get('response') or {}
        model = response.get('model')
        if isinstance(model, str):
            observed_models[model] += 1
        usage = response.get('usage')
        if isinstance(usage, dict):
            for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                value = usage.get(field)
                if type(value) in (int, float):
                    totals[field].append(value)
        record = id_to_record.get(call.get('id'))
        if stage != 'run' or record is None or record.get('method') not in METHODS:
            continue
        target = by_method[record['method']]
        target['calls'].append(call.get('id'))
        if isinstance(usage, dict):
            for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                value = usage.get(field)
                if type(value) in (int, float):
                    target[field].append(value)
        for field in ('elapsed_s', 'first_token_s'):
            value = record.get(field)
            if type(value) not in (int, float):
                value = call.get(field)
            if type(value) in (int, float):
                target[field].append(value)
        if type(call.get('elapsed_s')) in (int, float):
            target['endpoint_elapsed_s'].append(call['elapsed_s'])
    def describe(values):
        if not values:
            return dict(n=0, total=None, mean=None, median=None, p95=None)
        return dict(n=len(values), total=float(sum(values)), mean=float(np.mean(values)),
                    median=float(np.median(values)), p95=float(np.quantile(values, .95)))
    methods = {}
    for method, values in by_method.items():
        n = len(values['calls'])
        methods[method] = dict(actual_requests=n, usage_missing_calls=n-len(values['total_tokens']),
            **{field: describe(values[field]) for field in (
                'prompt_tokens', 'completion_tokens', 'total_tokens', 'elapsed_s', 'first_token_s', 'endpoint_elapsed_s')})
    return dict(actual_requests=len(raw), stages=dict(stage_counts), statuses=dict(statuses),
        observed_response_models=dict(observed_models),
        token_sums={field: sum(values) if values else None for field, values in totals.items()},
        usage_known_calls=len(totals['total_tokens']),
        usage_missing_calls=len(raw)-len(totals['total_tokens']), main_methods=methods,
        scope='Recorded endpoint generation only; separate automated review computation is not measured here.')


def analyze_main(directory, materials, profiles, records, ratings):
    cases = materials['test']
    case_ids = [case['id'] for case in cases]
    profile_ids = [profile['id'] for profile in profiles]
    if len(set(case_ids)) != len(case_ids) or len(set(profile_ids)) != len(profile_ids):
        raise ValueError('Duplicate planned case or profile')
    by_unit = {}
    for record in records:
        unit = record['case_id'], record['profile_id'], record['method']
        if unit in by_unit or unit[0] not in case_ids or unit[1] not in profile_ids or unit[2] not in METHODS:
            raise ValueError(f'Duplicate or out-of-plan main unit: {unit}')
        by_unit[unit] = record
    quality_rows, family_rows, risk_rows, label_counts = [], [], [], []
    case_family = {case['id']: case['family'] for case in cases}
    for coder in CODERS:
        assigned = ratings[coder]
        for profile_id in profile_ids:
            for metric in QUALITY:
                differences, complete, missing = [], [], 0
                method_values = {method: [] for method in METHODS}
                distribution = {method: Counter() for method in METHODS}
                by_family = defaultdict(list)
                for case_id in case_ids:
                    scores, absent = {}, False
                    for method in METHODS:
                        record = by_unit.get((case_id, profile_id, method))
                        rating = assigned.get(record['id']) if record else None
                        value, is_missing = quality_score(rating, metric, method)
                        scores[method] = value
                        absent |= is_missing
                        method_values[method].append(value)
                        distribution[method][str(value)] += 1
                    difference = scores['framework']-scores['strong_prompt']
                    differences.append(difference)
                    by_family[case_family[case_id]].append(difference)
                    if absent:
                        missing += 1
                    else:
                        complete.append(difference)
                row = dict(coder=coder, profile_id=profile_id, metric=metric,
                    margin=MARGIN, missing_pairs=missing, **interval(differences),
                    complete_case=interval(complete),
                    method_means={method: float(np.mean(values)) for method, values in method_values.items()},
                    distributions={method: dict(values) for method, values in distribution.items()})
                quality_rows.append(row)
                for family, values in by_family.items():
                    family_rows.append(dict(coder=coder, profile_id=profile_id, metric=metric,
                        family=family, n=len(values), mean_difference=float(np.mean(values)),
                        scope='Prespecified family description, not a multiplicity-adjusted superiority test.'))
        for dimension, bad_label in BAD.items():
            for method in METHODS:
                labels = []
                for case_id in case_ids:
                    for profile_id in profile_ids:
                        record = by_unit.get((case_id, profile_id, method))
                        rating = assigned.get(record['id']) if record else None
                        labels.append(rating.get(dimension) if rating else 'missing')
                label_counts.append(dict(coder=coder, method=method, dimension=dimension,
                                          n=len(labels), counts=dict(Counter(labels))))
            for conservative in (False, True):
                events = []
                for case_id in case_ids:
                    adverse_pairs = []
                    for profile_id in profile_ids:
                        labels = {}
                        for method in METHODS:
                            record = by_unit.get((case_id, profile_id, method))
                            rating = assigned.get(record['id']) if record else None
                            labels[method] = rating.get(dimension) if rating else 'missing'
                        framework_bad = labels['framework'] == bad_label
                        if conservative:
                            framework_bad |= labels['framework'] in (UNCERTAIN[dimension], 'missing')
                        baseline_bad = labels['strong_prompt'] == bad_label
                        if framework_bad and not baseline_bad:
                            adverse_pairs.append(profile_id)
                    events.append(dict(case_id=case_id, event=bool(adverse_pairs), profiles=adverse_pairs))
                count = sum(item['event'] for item in events)
                upper = exact_upper(count, len(case_ids))
                risk_rows.append(dict(coder=coder, dimension=dimension,
                    sensitivity='unknown_and_missing_unfavorable' if conservative else 'confirmed_labels_only',
                    events=count, cases=len(case_ids), observed_rate=count/len(case_ids),
                    exact_upper_95=upper, risk_margin=RISK_MARGIN,
                    supports_risk_margin=upper is not None and upper <= RISK_MARGIN,
                    event_cases=[item for item in events if item['event']],
                    scope='Any framework-only adverse pair within a case; conservative upper bound, not the observed error-rate difference.'))
    confirmed = []
    for case_id in case_ids:
        for profile_id in profile_ids:
            for dimension in ('facts', 'boundary'):
                pair = {}
                for method in METHODS:
                    record = by_unit.get((case_id, profile_id, method))
                    pair[method] = [ratings[coder].get(record['id'], {}).get(dimension) if record else None for coder in CODERS]
                if all(label == BAD[dimension] for label in pair['framework']) and all(
                        label is not None and label != BAD[dimension] for label in pair['strong_prompt']):
                    confirmed.append(dict(case_id=case_id, profile_id=profile_id, dimension=dimension))
    all_quality = all(row['supports_margin'] for row in quality_rows)
    any_bad_mean = any(row['mean_difference'] < -MARGIN for row in quality_rows)
    all_bootstrap = all(row['bootstrap_supports_margin'] for row in quality_rows)
    all_risk = all(row['supports_risk_margin'] for row in risk_rows)
    if confirmed or any_bad_mean:
        extension = 'hold_for_mechanism_or_scope_review'
    elif all_quality:
        extension = 'recommend_separately_registered_larger_confirmation'
    else:
        extension = 'insufficient_precision_no_automatic_extension'
    return dict(planned_cases=len(case_ids), planned_profiles=len(profile_ids),
        planned_generation_units=len(case_ids)*len(profile_ids)*len(METHODS),
        observed_units=len(records), statuses=dict(Counter(record.get('status', 'missing') for record in records)),
        missing_units=len(case_ids)*len(profile_ids)*len(METHODS)-len(records),
        mechanical_checks={method:dict(Counter(record.get('mechanical',{}).get('status','missing')
            for record in records if record['method']==method)) for method in METHODS},
        protocol_validity={method:dict(Counter(
            'not_applicable' if record.get('validity') is None else
            'valid' if record['validity'].get('valid') else 'invalid'
            for record in records if record['method']==method)) for method in METHODS},
        primary_quality=quality_rows, quality_noninferiority_supported=all_quality,
        quality_bootstrap_sensitivity_supported=all_bootstrap,
        family_descriptive=family_rows, behavioral_labels=label_counts, failure_risk=risk_rows,
        rare_failure_noninferiority_supported=all_risk,
        overall_noninferiority_supported=all_quality and all_risk,
        jointly_flagged_framework_only_fact_or_boundary=confirmed,
        extension_decision=extension, automatic_additional_requests=False,
        qualification='Synthetic material and two automated coding tasks; no population-wide or human-experience noninferiority claim.')


def summarize(directory=HERE):
    directory = Path(directory)
    profiles, materials = load(directory/'profiles.json'), load(directory/'materials.json')
    main_path, dev_path = directory/'run/results.json', directory/'development/results.json'
    main = load(main_path).get('records', []) if main_path.exists() else []
    dev = load(dev_path).get('records', []) if dev_path.exists() else []
    result = dict(study_id=directory.name, version=1,
        analysis=dict(quality_margin=MARGIN, alpha_one_sided=ALPHA, primary_interval='paired t by situation',
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS, bootstrap_seed=BOOTSTRAP_SEED,
            risk_margin=RISK_MARGIN, risk_interval='one-sided exact Clopper-Pearson on framework-only case events',
            missing_quality='framework=1, strong_prompt=5; complete cases also shown',
            joint_primary='All persona x metric x coder lower bounds must exceed -0.3.',
            ordinary_failure='Unknown is reported separately and tested as unfavorable for framework in sensitivity.'),
        costs=cost_summary(directory, dev+main))
    result['main'] = analyze_main(directory, materials, profiles, main,
        read_ratings(directory, 'main', main)) if main else None
    result['development'] = dict(observed_units=len(dev),
        statuses=dict(Counter(record.get('status', 'missing') for record in dev)),
        selection=load(directory/'selection.json') if (directory/'selection.json').exists() else None)
    return result


def self_test():
    assert abs(exact_upper(0, 24)-(1-.05**(1/24))) < 1e-12
    assert exact_upper(24, 24) == 1
    assert exact_upper(0, 0) is None
    assert exact_upper(0, 24) > RISK_MARGIN
    assert exact_upper(0, 59) < RISK_MARGIN
    assert interval([0]*24)['supports_margin']
    assert not interval([-1]*24)['supports_margin']
    assert interval([0,1,-1]*8) == interval([0,1,-1]*8)
    assert quality_score(None, 'persona_fit', 'framework') == (1, True)
    assert quality_score(None, 'persona_fit', 'strong_prompt') == (5, True)
    materials = dict(test=[dict(id=f'C{i}', family=f'F{i%8}') for i in range(24)])
    profiles = [dict(id='P1'), dict(id='P2')]
    records, ratings = [], {coder:{} for coder in CODERS}
    for case in materials['test']:
        for profile in profiles:
            for method in METHODS:
                rid = f"{case['id']}_{profile['id']}_{method}"
                records.append(dict(id=rid, case_id=case['id'], profile_id=profile['id'], method=method, status='ok'))
                for coder in CODERS:
                    ratings[coder][rid] = dict(persona_fit=4, context_fit=4, facts='supported', boundary='respected', fulfillment='fulfilled')
    report = analyze_main(HERE, materials, profiles, records, ratings)
    assert len(report['primary_quality']) == 8 and report['quality_noninferiority_supported']
    assert not report['rare_failure_noninferiority_supported']
    assert not report['overall_noninferiority_supported']
    assert report['extension_decision'] == 'recommend_separately_registered_larger_confirmation'
    for coder in CODERS:
        ratings[coder]['C0_P1_framework']['facts'] = 'conflict'
        # The opposite-profile baseline fault cannot mask the adverse P1 pair.
        ratings[coder]['C0_P2_strong_prompt']['facts'] = 'conflict'
    report = analyze_main(HERE, materials, profiles, records, ratings)
    assert report['jointly_flagged_framework_only_fact_or_boundary']
    assert report['extension_decision'] == 'hold_for_mechanism_or_scope_review'
    assert next(row for row in report['failure_risk'] if row['dimension']=='facts')['events'] == 1
    print('analysis self-test passed; no observations or API calls created')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    summary = summarize()
    save(args.out/'summary.json', summary)
    if not args.no_figures:
        from generate_figures import generate
        generate(summary, load(HERE/'profiles.json'), args.out/'figures')
    print(json.dumps(dict(status='recomputed', out=str(args.out), main_available=summary['main'] is not None), ensure_ascii=False))


if __name__ == '__main__':
    main()
