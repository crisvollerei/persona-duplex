"""Export all observed dialogues, never select only favorable examples."""
import benchmark as b


def main():
    b.verify()
    cases = {c['id']: c for c in b.load(b.HERE / 'cases.json')}
    profiles = {p['id']: p for p in b.load(b.HERE / 'profiles.json')}
    data = b.load(b.HERE / 'run/results.json')
    lines = ['# 完整四轮对话', '', '[论文](paper.md) · [原始结果](run/results.json)', '',
             '以下按情境、人格、方法与重复列出全部48段对话，不筛除短回复或不利样例。模拟状态不代表现实执行。', '']
    for episode in data['episodes']:
        lines += ['## ' + episode['id'], '', profiles[episode['profile']]['name'] + '；方法：' + episode['method'], '',
                  '**背景：**' + cases[episode['case']]['context'], '']
        for row in episode['turns']:
            lines += ['**' + row['turn'] + ' 用户：**' + row['user'], '',
                      '**回复：**' + (row['say'] if row['say'] is not None else '〔缺失：' + row['status'] + '〕'), '',
                      '程序选择 `' + str(row['program_selected_id']) + '`；模型报告 `' + str(row['model_reported_id'])
                      + '`；执行 `' + str(row['executed_id']) + '`。', '']
    (b.HERE / 'transcripts.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
