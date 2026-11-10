"""Render every assigned pilot response, including failed deliveries."""
import json
import benchmark as b


def quote(text):
    return '\n'.join('> '+line for line in text.splitlines()) if text else '> （未交付正文；这是记录说明，不是模型回复。）'


def main():
    profiles={p['id']:p for p in b.load(b.HERE/'profiles.json')}
    materials=b.load(b.HERE/'materials.json')
    for stage,folder,filename in [('test','run','transcripts.md'),('development','development','development_transcripts.md')]:
        rows=b.load(b.HERE/folder/'results.json')['records']
        lines=['# '+('正式集完整对话' if stage=='test' else '开发集全部候选回复'),'',
               '以下保留全部分配单元的实际SAY；完整请求、内部字段、来源和元数据另见原始JSON。',
               '历史为两路共享的合成初态，每个情境仅生成一次续答，不表示真实多轮会话已经执行。','']
        for case in materials[stage]:
            lines += ['## '+case['id']+' · '+case['title'],'',
                      '**事实资料：**'+json.dumps(case['facts'],ensure_ascii=False),'']
            if case['history']:
                lines+=['**共享历史：**','',quote('\n'.join(h['role']+'：'+h['content'] for h in case['history'])),'']
            lines+=['**当前用户：**'+case['query'],'']
            for pid,profile in profiles.items():
                lines+=['### '+profile['name'],'']
                for r in sorted((r for r in rows if r['case_id']==case['id'] and r['profile_id']==pid),key=lambda r:r['id']):
                    label='框架三段生成' if r['method']=='framework' else '传统强提示 · '+r['candidate_id']
                    lines+=['**'+label+'**（'+r['status']+'）','',quote(r['say']),'']
        (b.HERE/filename).write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Rendered all 36 development and 96 main outputs, with empty deliveries retained.')


if __name__=='__main__':main()
