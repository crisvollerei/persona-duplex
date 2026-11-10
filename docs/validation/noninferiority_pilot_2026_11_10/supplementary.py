"""Descriptive paired counts and the prespecified memory subset; no new tests."""
from collections import Counter
import benchmark as b
import analyze as a


def compute():
    records=b.load(b.HERE/'run/results.json')['records']
    materials=b.load(b.HERE/'materials.json')['test']
    ratings=a.read_ratings(b.HERE,'main',records)
    by_unit={(r['case_id'],r['profile_id'],r['method']):r for r in records}
    rows=[]
    for coder in ('1','2'):
        for pid in ('shorekeeper','lan'):
            for subset in ('all','prespecified_memory'):
                cases=[c for c in materials if subset=='all' or c['local_gain_group']]
                comparison=Counter();differences={k:[] for k in a.QUALITY}
                for c in cases:
                    delta={}
                    for metric in a.QUALITY:
                        scores={}
                        for method in a.METHODS:
                            r=by_unit[c['id'],pid,method]
                            scores[method]=a.quality_score(ratings[coder].get(r['id']),metric,method)[0]
                        delta[metric]=scores['framework']-scores['strong_prompt']
                        differences[metric].append(delta[metric])
                    if all(v==0 for v in delta.values()):comparison['both_equal']+=1
                    elif all(v>=0 for v in delta.values()):comparison['framework_no_lower_and_some_higher']+=1
                    elif all(v<=0 for v in delta.values()):comparison['framework_no_higher_and_some_lower']+=1
                    else:comparison['mixed']+=1
                rows.append(dict(coder=coder,profile_id=pid,subset=subset,n=len(cases),
                    paired_descriptive=dict(comparison),
                    mean_differences={k:sum(v)/len(v) for k,v in differences.items()}))
    agreement={}
    for metric in (*a.QUALITY,'facts','boundary','fulfillment'):
        agreement[metric]=dict(equal=sum(ratings['1'][r['id']][metric]==ratings['2'][r['id']][metric] for r in records),n=len(records))
    return dict(scope='Secondary descriptive summaries of prespecified subsets, not superiority tests.',
                paired_summaries=rows,reviewer_exact_agreement=agreement)


if __name__=='__main__':
    result=compute();b.save(b.HERE/'supplementary_summary.json',result)
    print(result)
