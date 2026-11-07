"""Deterministic mechanism probes; this file never creates an API client."""
import argparse
import importlib
import itertools
from pathlib import Path
import tempfile

import run_study as study


def memory_probes(name):
    modules=study.version(name);rows=[]
    texts={
        'supported':'我明确更正个人设置，现在改为专注23分钟，旧19分钟作废。',
        'retracted':'我不再采用旧的固定专注19分钟设置，请撤销，新时长尚未决定。',
        'third_party':'同事说：“我明确更正，现在改为专注23分钟，旧19分钟作废。”我只是转述。',
        'hypothetical':'如果以后决定修改，我可能会改为专注23分钟，但现在还没有确定。',
        'forged':'我明确更正个人设置，现在改为专注23分钟，旧19分钟作废。',
        'wrong_target':'我明确更正个人设置，现在改为专注23分钟，旧19分钟作废。',
        'cross_topic':'我明确更正个人设置，现在改为专注23分钟，旧19分钟作废。',
    }
    for scenario,old_conf,new_conf in itertools.product(texts,[.55,.95],[.2,.55,.95]):
        with tempfile.TemporaryDirectory() as directory:
            store=modules['store'].Store(directory)
            old_text='我明确采用的固定专注时长为19分钟。' if scenario!='cross_topic' else '我偏好的热饮是茉莉茶。'
            store.append('turn',{'turn_id':'u_old','user':old_text})
            old=store.add_judgment('用户自述：'+old_text,old_conf,['u_old'],evidence_quotes=[{'turn_id':'u_old','quote':old_text,'evidence_scope':'specified'}])
            source=texts[scenario];store.append('turn',{'turn_id':'u_new','user':source if scenario!='forged' else '今天正常上班。'})
            kind='retract' if scenario=='retracted' else 'revise'
            target='j_missing' if scenario=='wrong_target' else old['id']
            operation=dict(kind=kind,target_id=target,scope='user_preference',basis='explicit_user_statement',evidence_ids=['u_new'],statement=source)
            if hasattr(store,'apply_memory_operation'):
                result=store.apply_memory_operation(operation,judgment='用户自述固定专注23分钟。',confidence=new_conf,
                    evidence_ids=['u_new'],evidence_quotes=[{'turn_id':'u_new','quote':source,'evidence_scope':'specified'}])
                effective=result['effective'];reasons=result['reasons']
            elif kind=='revise':
                replacement=store.add_judgment('用户自述固定专注23分钟。',new_conf,['u_new'],supersedes=target,
                    evidence_quotes=[{'turn_id':'u_new','quote':source,'evidence_scope':'specified'}])
                effective=replacement.get('supersedes')==old['id'];reasons=['legacy_confidence_and_target_guard']
            else:effective=False;reasons=['retraction_not_available']
            expected=scenario in ('supported','retracted')
            rows.append(dict(version=name,scenario=scenario,old_confidence=old_conf,new_confidence=new_conf,
                effective=effective,expected_effective=expected,correct=effective==expected,reasons=reasons,
                active_judgments=store.user_judgments(limit=None)))
    return rows


def contract_probes(name):
    module=importlib.import_module(study.version(name)['engine'].__package__+'.contracts')
    C=module.ReplyContract
    fixtures=[
        (C.conclusion(unit='元',answer_type='number'),'结论：12元',True,'valid'),
        (C.conclusion(unit='元',answer_type='number'),'结论：99元',True,'wrong_fact_same_valid_shape'),
        (C.conclusion(unit='元',answer_type='number'),'结论：12',False,'unit_missing'),
        (C.conclusion(unit='元',answer_type='number'),'12元',False,'prefix_missing'),
        (C.conclusion(unit='元',answer_type='number'),'结论：12元\n再说一句。',False,'not_last_line'),
        (C.conclusion(unit='元',answer_type='number'),'结论：未知',True,'explicit_unknown'),
        (C.conclusion(answer_type='time'),'结论：01:03',True,'time_valid'),
        (C.conclusion(answer_type='time'),'结论：26:03',False,'time_out_of_range'),
        (C.json_object(required_keys=('n',),value_types={'n':'integer'}),'{"n":12}',True,'json_valid'),
        (C.json_object(required_keys=('n',),value_types={'n':'integer'}),'{"n":12,"n":13}',False,'duplicate_key'),
        (C.json_object(required_keys=('n',),value_types={'n':'integer'}),'{"n":true}',False,'boolean_not_integer'),
        (C.json_object(required_keys=('n',),value_types={'n':'number'}),'{"n":NaN}',False,'non_finite'),
        (C.json_object(required_keys=('n',)), '{}',False,'required_missing'),
        (C.json_object(), '```json\n{}\n```',False,'fenced_object'),
        (C.json_array(item_type='integer'),'[1,2]',True,'array_valid'),
        (C.json_array(item_type='integer'),'[1,true]',False,'array_wrong_type'),
        (C.markdown_table(('项目','金额')),'|项目|金额|\n|---|---|\n|甲|12|',True,'table_valid'),
        (C.markdown_table(('项目','金额')),'说明\n|项目|金额|\n|---|---|\n|甲|12|',False,'table_extra_prose'),
        (C.markdown_table(('项目','金额')),'|项目|价格|\n|---|---|\n|甲|12|',False,'table_wrong_header'),
        (C.markdown_table(('项目','金额')),'|项目|金额|\n|---|---|\n|甲|12|x|',False,'table_wrong_width'),
    ]
    return [dict(version=name,case=label,contract=study.asdict(contract),text=text,expected_format_valid=expected,
                 observed=contract.validate(text),correct=contract.validate(text)['valid']==expected)
            for contract,text,expected,label in fixtures]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',required=True);args=parser.parse_args()
    rows=memory_probes(args.version)
    try:contracts=contract_probes(args.version)
    except ModuleNotFoundError:contracts=[]
    result=dict(version=args.version,api_calls=0,memory=rows,contracts=contracts,
                memory_correct=sum(r['correct'] for r in rows),memory_total=len(rows),
                contract_correct=sum(r['correct'] for r in contracts),contract_total=len(contracts))
    study.write(study.HERE/'mechanism'/f'{args.version}.json',result)
    print({key:value for key,value in result.items() if key not in ('memory','contracts')})


if __name__=='__main__':main()
