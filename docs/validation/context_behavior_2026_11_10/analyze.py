"""Recompute calibration, paired descriptions, costs, and scientific figures."""
from collections import Counter, defaultdict
import argparse
import statistics
from pathlib import Path
import benchmark as b
import gate
import mechanism as m


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def rating_rows(path):
    rows = b.load(path)
    return rows['ratings'] if isinstance(rows, dict) else rows


def calibration_summary():
    key = b.load(b.HERE / 'calibration_key.json')
    scores, ratings = {}, {}
    for c in ('1', '2'):
        name = '评审' + c
        path = b.HERE / 'calibration_reviews' / ('ratings_' + c + '.json')
        ratings[name] = {r['id']: r for r in rating_rows(path)}
        scores[name] = gate.assess(list(ratings[name].values()), key)
    for row in b.load(b.HERE / 'calibration_run/results.json')['records']:
        name = '端点重复' + str(row['repetition'])
        if row['status'] == 'ok':
            ratings[name] = {r['id']: r for r in row['ratings']}
            scores[name] = gate.assess(row['ratings'], key)
        else:
            scores[name] = dict(status=row['status'])
    agreement = {}
    for field in gate.FIELDS:
        agreement[field] = dict(correct=sum(ratings['评审1'][id][field] == ratings['评审2'][id][field] for id in key),
                                denominator=len(key))
    backend_agreement = {}
    if '端点重复1' in ratings and '端点重复2' in ratings:
        backend_agreement = {f: dict(correct=sum(ratings['端点重复1'][id][f] == ratings['端点重复2'][id][f] for id in key),
                                     denominator=len(key)) for f in gate.FIELDS}
    absence = [id for id in key if key[id]['expected']['fulfills_request'] is None]
    absence_counts = {name: dict(correct_null=sum(rs[id]['fulfills_request'] is None for id in absence),
                                denominator=len(absence)) for name, rs in ratings.items()}
    return dict(assessments=scores, codex_exact_agreement=agreement,
                coding_configurations={'Codex': 'full rubric.md and review_instructions.md',
                                       'Backend': 'abbreviated mechanism.LABEL_SYSTEM; model/instruction/form confounded'},
                backend_repeat_agreement=backend_agreement, no_request_null=absence_counts,
                planned_items=24, authored_pairs=12,
                interpretation='Agreement with author-created expectations, not natural-distribution accuracy.')


def online_cost():
    records = []
    for stage in ('calibration_run', 'run'):
        for path in sorted((b.HERE / stage / 'calls').glob('*.json')):
            row = b.load(path)
            records.append(dict(id=row['id'], stage=stage, status=row['status'],
                                elapsed_s=row['elapsed_s'], response_model=row.get('response', {}).get('model'),
                                usage=row.get('response', {}).get('usage', {})))
    return dict(actual_requests=len(records), statuses=dict(Counter(r['status'] for r in records)),
                response_models=dict(Counter(r['response_model'] for r in records)),
                total_tokens=sum(r['usage'].get('total_tokens', 0) for r in records), records=records)


def main_summary():
    path = b.HERE / 'run/results.json'
    if not path.exists():
        return None
    data = b.load(path)['records']
    mapping = b.load(b.HERE / 'review_key.json')
    coders = {}
    for coder in ('1', '2'):
        path = b.HERE / 'reviews' / ('ratings_' + coder + '.json')
        if not path.exists():
            continue
        rs = rating_rows(path)
        assert {r['id'] for r in rs} == set(mapping.values()) and len(rs) == len(set(mapping.values()))
        by_id = {r['id']: r for r in rs}
        for r in rs:
            assert set(r) == {'id', 'fulfills_request', 'adds_agenda', 'fact_status', 'boundary_status',
                              'persona_fit', 'context_fit', 'focus', 'reason'}
            assert all(r[f] is None or type(r[f]) is bool for f in gate.FIELDS[:2])
            assert r['fact_status'] in ('supported', 'contradicted', 'unknown')
            assert r['boundary_status'] in ('respected', 'violated', 'unknown')
            assert all(r[f] is None or type(r[f]) is int and 1 <= r[f] <= 5 for f in ('persona_fit', 'context_fit'))
            assert r['focus'] in ('practical', 'feeling', 'detail', 'mixed', 'unknown')
        groups = defaultdict(list)
        for row in data:
            if row['id'] in mapping:
                groups[(row['method'], row['profile'], row['variant'])].append(by_id[mapping[row['id']]])
        coders[coder] = {'/'.join(k): dict(n=len(v), persona_mean=mean([r['persona_fit'] for r in v]),
            context_mean=mean([r['context_fit'] for r in v]),
            distributions={f: dict(Counter(str(r[f]) for r in v)) for f in (*gate.FIELDS, 'persona_fit', 'context_fit', 'focus')})
            for k, v in groups.items()}
    execution = defaultdict(lambda: dict(planned=0, ok=0, rule_match=0, allowed=0))
    mismatches = []
    for row in data:
        group = execution[row['method']]
        group['planned'] += 1
        group['ok'] += row['status'] == 'ok'
        group['rule_match'] += bool(row.get('rule_match'))
        group['allowed'] += bool(row.get('allowed'))
        if row['status'] == 'ok' and not row.get('rule_match'):
            mismatches.append(row['id'])
        expected = m.choose(row['payload'])
        assert expected == row['selected_by_program']
        if row['method'] == 'program':
            assert m.messages(row['payload'], 'program')[1]['content'] == b.load(b.HERE / 'run/calls' / (row['id'] + '.json'))['request']['messages'][1]['content']
    contrasts = []
    for case in b.load(b.HERE / 'cases.json'):
        for profile in b.load(b.HERE / 'profiles.json'):
            for rep in range(1, 4):
                choices = {v['id']: m.choose(m.payload(case, v, profile, rep)) for v in case['variants']}
                contrasts.append(dict(case=case['id'], profile=profile['id'], rep=rep, choices=choices,
                                      changed=len(set(choices.values())) > 1))
    return dict(execution=dict(execution), rule_mismatch_ids=mismatches, coders=coders,
                context_replay=contrasts, unique_review_units=len(set(mapping.values())),
                mapped_outputs=len(mapping),
                static_ablation_changed=sum(r['selected_by_program'] != r['static_ablation'] for r in data),
                static_ablation_denominator=len(data),
                interpretation='Six synthetic scenario clusters; repetitions/variants are not independent scenarios.')


def figures(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'svg.hashsalt': 'context-behavior-v1', 'font.size': 10})
    folder = out / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    colors = ['#266b8f', '#5aa5ad', '#d68340', '#b55650']
    values = summary['calibration']['assessments']
    fig, ax = plt.subplots(figsize=(9.4, 4.8), layout='constrained')
    x = np.arange(4)
    for i, (name, row) in enumerate(values.items()):
        if 'field_correct' in row:
            ax.bar(x + (i-1.5)*.19, [row['field_correct'][f]/24 for f in gate.FIELDS], .18, label=name, color=colors[i])
    ax.set_xticks(x, ['请求履行', '额外议程', '事实逻辑', '明确边界'])
    ax.set_ylim(0, 1.13)
    ax.set_ylabel('与作者冻结预期一致的比例（各维 n=24）')
    ax.set_title('同文换情境：分维度校准；不是自然分布准确率')
    ax.legend(ncol=2, loc='lower left')
    for suffix in ('png', 'svg'):
        fig.savefig(folder / ('fig1_calibration.' + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), layout='constrained')
    names = list(values)
    for index, (field, denominator, title) in enumerate((
            ('correct_contrast_pairs', 'contrast_pairs', '行为改变配对（n=8）'),
            ('correct_control_pairs', 'control_pairs', '负对照配对（n=4）'))):
        numbers = [values[n].get(field, 0) for n in names]
        axes[index].bar(names, numbers, color=colors)
        axes[index].set_ylim(0, 9 if index == 0 else 5)
        axes[index].set_title(title)
        axes[index].set_ylabel('完整符合预期的配对数')
        axes[index].tick_params(axis='x', rotation=20)
        for i, v in enumerate(numbers):
            axes[index].text(i, v + .1, str(v), ha='center')
    for suffix in ('png', 'svg'):
        fig.savefig(folder / ('fig2_context_pairs.' + suffix), dpi=180)
    plt.close(fig)
    main = summary['main']
    if main and main['coders']:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
        for j, metric in enumerate(('persona_mean', 'context_mean')):
            labels, y = [], []
            for coder, groups in main['coders'].items():
                for profile in ('shorekeeper', 'lan'):
                    for method in ('program', 'prompt'):
                        selected = [r for k,r in groups.items() if k.startswith(method + '/' + profile + '/')]
                        labels.append(coder + '/' + profile + '/' + method)
                        y.append(sum(r[metric]*r['n'] for r in selected)/sum(r['n'] for r in selected))
            axes[j].barh(labels, y, color=['#266b8f', '#d68340']*4)
            axes[j].set_xlim(0, 5)
            axes[j].set_title('人格符合度' if j == 0 else '情境适切性')
            axes[j].set_xlabel('等级均值（不是综合质量分）')
        for suffix in ('png', 'svg'):
            fig.savefig(folder / ('fig3_expression_ratings.' + suffix), dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=b.HERE); a = p.parse_args()
    b.verify()
    summary = dict(calibration=calibration_summary(), gate=b.load(b.HERE / 'calibration_gate.json'),
                   costs=online_cost(), main=main_summary())
    b.save(a.out / 'summary.json', summary)
    figures(summary, a.out)
    print({'decision': summary['gate']['decision'], 'actual_requests': summary['costs']['actual_requests'],
           'main_executed': summary['main'] is not None})


if __name__ == '__main__':
    main()
