"""Render four descriptive figures from recomputed archived pilot statistics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from analyze import HERE, MARGIN, RISK_MARGIN, load, summarize

BLUE, ORANGE, INK = '#3578a6', '#ca7c42', '#26384b'
QUALITY_NAMES = dict(persona_fit='人格符合', context_fit='情境适切')
DIMENSION_NAMES = dict(facts='事实', boundary='边界', fulfillment='履行请求')
FAMILY_NAMES = dict(explicit_task_format='明确任务与格式', listening='倾听与照应',
    memory_revision='记忆更正', old_memory_recall='相关旧记忆', persona_task_tension='人格与任务协调',
    scope_time_exceptions='范围与临时例外', uncertainty_conditions='未知与条件', user_boundaries='用户边界')


def configure():
    available = {font.name for font in font_manager.fontManager.ttflist}
    choices = ['Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams.update({'font.family': [name for name in choices if name in available],
        'axes.unicode_minus': False, 'figure.facecolor': '#fbfcfe', 'axes.facecolor': '#fbfcfe',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'font.size': 10,
        'svg.fonttype': 'none', 'savefig.bbox': 'tight'})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def quality_figure(main, profiles, path):
    names = {profile['id']: profile['name'] for profile in profiles}
    rows = main['primary_quality']
    fig, ax = plt.subplots(figsize=(11, 6.5))
    values = []
    for i, row in enumerate(rows):
        color = BLUE if row['metric']=='persona_fit' else ORANGE
        mean, low, high = row['mean_difference'], row['t_lower_95'], row['t_upper_90']
        if low is not None:
            ax.plot([low, high], [i,i], color=color, lw=2)
            ax.plot([low, high], [i,i], '|', color=color, ms=9)
            values.extend([low,high])
        ax.plot(mean,i,'o',color=color,ms=7)
        if row['bootstrap_lower_95'] is not None:
            ax.plot(row['bootstrap_lower_95'],i,'<',mfc='white',mec=color,ms=7)
        values.append(mean)
    left = min(-.5, min(values, default=0)-.12)
    right = max(.3,max(values,default=0)+.12)
    ax.axvspan(left,-MARGIN,color='#f5e4e0',zorder=0)
    ax.axvline(-MARGIN,color='#ac4b43',ls='--',label='预定不劣界 −0.3')
    ax.axvline(0,color='#9aa5b1',lw=1)
    ax.set_xlim(left,right)
    ax.set_yticks(range(len(rows)), [f"评审{r['coder']} · {names.get(r['profile_id'],r['profile_id'])} · {QUALITY_NAMES[r['metric']]}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('框架减强提示的等级差；每项按24个情境配对')
    ax.set_title('表达质量：分人格、分评审的不劣性估计',loc='left',fontweight='bold',pad=17)
    ax.text(0,-.17,'实心点：均值；线：90%双侧t区间（左端为单侧95%下界）；空心三角：bootstrap单侧95%下界。',
            transform=ax.transAxes,fontsize=9)
    ax.legend(loc='best',frameon=False)
    ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.27,bottom=.18)
    save(fig,path)


def family_figure(main, profiles, path):
    rows = main['family_descriptive']
    families = list(dict.fromkeys(row['family'] for row in rows))
    names = {profile['id']:profile['name'] for profile in profiles}
    indexes = [(family,profile['id']) for family in families for profile in profiles]
    columns = [(coder,metric) for coder in ('1','2') for metric in QUALITY_NAMES]
    matrix = np.zeros((len(indexes),len(columns)))
    for i,(family,profile_id) in enumerate(indexes):
        for j,(coder,metric) in enumerate(columns):
            selected = [row for row in rows if row['family']==family and row['profile_id']==profile_id and row['coder']==coder and row['metric']==metric]
            matrix[i,j] = np.mean([row['mean_difference'] for row in selected]) if selected else np.nan
    bound=max(.5,float(np.nanmax(np.abs(matrix))))
    fig,ax=plt.subplots(figsize=(10,9))
    im=ax.imshow(matrix,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    for i in range(len(indexes)):
        for j in range(len(columns)):
            ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>bound*.6 else INK)
    ax.set_yticks(range(len(indexes)),[f'{FAMILY_NAMES.get(family,family)} · {names[profile_id]}' for family,profile_id in indexes])
    ax.set_xticks(range(len(columns)),[f'评审{coder}\n{QUALITY_NAMES[metric]}' for coder,metric in columns])
    ax.set_title('预定场景类别的描述性差异',loc='left',fontweight='bold',pad=15)
    ax.text(0,-.1,'颜色为框架减强提示的等级均值差，两人格分别展示；仅定位复验假设，不作局部优越证明。',transform=ax.transAxes,fontsize=9)
    fig.colorbar(im,ax=ax,label='等级差')
    fig.subplots_adjust(left=.35,bottom=.14)
    save(fig,path)


def cost_figure(costs, path):
    methods=costs['main_methods']
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    metrics=[('total_tokens','每次主生成的平均token','token'),('elapsed_s','每次主生成的平均总等待','秒'),('actual_requests','主试验端点调用数','次')]
    for ax,(metric,title,unit) in zip(axes,metrics):
        heights=[]
        for i,method in enumerate(('framework','strong_prompt')):
            record=methods[method][metric] if metric!='actual_requests' else dict(mean=methods[method]['actual_requests'],n=methods[method]['actual_requests'])
            value=record['mean']
            if value is None:
                ax.text(i,.05,'未记录',ha='center',transform=ax.get_xaxis_transform())
            else:
                heights.append(value)
                ax.bar(i,value,width=.58,color=(BLUE,ORANGE)[i])
                ax.annotate(f'{value:.2f}\nn={record["n"]}',(i,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks([0,1],['框架','强提示'])
        ax.set_ylabel(unit)
        ax.set_title(title,fontsize=11,pad=14)
        ax.set_ylim(0,max(heights,default=1)*1.28 or 1)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('主比较的实际调用成本',x=.07,ha='left',fontweight='bold')
    fig.text(.07,.02,'各面板单位独立；只统计端点实际用量与完整单元等待，缺失不填零；首字等待未测，端点外评审成本另列。',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    save(fig,path)


def risk_figure(main, path):
    rows=[row for row in main['failure_risk'] if row['sensitivity']=='confirmed_labels_only']
    fig,ax=plt.subplots(figsize=(10,5.7))
    for i,row in enumerate(rows):
        point,upper=row['observed_rate']*100,row['exact_upper_95']*100
        ax.plot([point,upper],[i,i],color=BLUE,lw=2)
        ax.plot(point,i,'o',color=BLUE)
        ax.plot(upper,i,'|',color=BLUE,ms=10)
        ax.annotate(f"{row['events']}/{row['cases']}；上界{upper:.1f}%",(upper,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
    ax.axvline(RISK_MARGIN*100,color='#ac4b43',ls='--',label='预定风险增量界 5个百分点')
    ax.set_yticks(range(len(rows)),[f"评审{row['coder']} · {DIMENSION_NAMES[row['dimension']]}" for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0,max(25,max(row['exact_upper_95']*100 for row in rows)+15))
    ax.set_xlabel('任一人格出现框架特有不利事件的情境比例（%）')
    ax.set_title('稀有失效：零观测仍有非零风险上界',loc='left',fontweight='bold',pad=15)
    ax.text(0,-.18,'线右端为24个情境的单侧95%二项精确上界；事件比例保守上界错误率差，不能当成已测增量。',transform=ax.transAxes,fontsize=9)
    ax.legend(frameon=False,loc='best')
    ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.2,bottom=.2)
    save(fig,path)


def generate(summary, profiles, out):
    configure()
    out=Path(out)
    main=summary.get('main')
    if main is None:
        raise ValueError('Main observations are not available; do not fabricate figures')
    quality_figure(main,profiles,out/'fig1_quality_noninferiority')
    family_figure(main,profiles,out/'fig2_scenario_differences')
    cost_figure(summary['costs'],out/'fig3_generation_costs')
    risk_figure(main,out/'fig4_failure_risk')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'figures')
    args=parser.parse_args()
    generate(summarize(),load(HERE/'profiles.json'),args.out)
    print('Recomputed four PNG/SVG figures from archived observations:',args.out)


if __name__=='__main__':
    main()
