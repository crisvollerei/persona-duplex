# 新实例泛化与记忆更新研究

[研究目录](../../research.md) · [论文](paper.md) · [请求前固定方案](protocol.md) · [数据字典](data_dictionary.md)

2026-11-06，按请求前固定的任务、评分和预算完成150个单元、186条对话输出、210次请求。研究考察新的事实实例、受控记忆冲突和真实固化后的设置更新。全部请求成功，实际返回模型均为 `deepseek-flash`；请求标识是 `deepseek-chat`。

| 主要观察 | 结果 |
| --- | --- |
| 新题严格结论通过 | 普通提示：中性12/16、蓝鱼11/16、白龙12/16；默认text：10/16、14/16、13/16 |
| 未解决冲突、旧记忆遇当前更正、空白更正、撤销 | 各8/8；无关记忆5/8严格通过 |
| 真实更新记忆召回 / 当前更正 / 空白答未知 | 各6/6 |
| 框架协议有效 / 实际交付一致 | 136/138、138/138 |

27条严格失败输出的事后条件匿名 Codex 复核均支持其所问目标结论；失败主要是缺失规定标签。原始分数保持不变，补充标签不是人类真值或全文事实评价。具体区间、反例与结论边界见论文。

## 材料

- [逐单元结果](run/results.json)、[210次完整调用](run/raw_calls.json.gz)、[固定配置和预算](run/protocol.json)、[执行计划](run/execution_plan.json)。
- [运行器快照](run/runner_snapshot.py)、[核心源码清单](run/implementation_snapshot/manifest.json)、[人设配置](personas)。人设参考来源沿用[前期材料](../followup_2026_11_05/persona_sources.md)。
- [统计汇总](summary.json)、[自动结果表](results_tables.md)、[四幅图](figures)。
- [补充复核方案](format_failure_review_protocol.json)、[匿名输入](format_failure_review_inputs.json)、[标签](format_failure_review_ratings.json)、[映射](format_failure_review_key.json)、[汇总](format_failure_review_summary.json)。
- [材料审计](audit_report.json)和[公开文件哈希](artifact_manifest.json)。

## 离线重算、重绘与审计

从仓库根目录执行。分析不调用模型，只覆盖派生统计和图表，保留 `run/` 原始证据：

```powershell
python -m pip install -r docs/validation/generalization_2026_11_06/requirements.txt
python -B -X utf8 docs/validation/generalization_2026_11_06/test_study.py
python -B -X utf8 docs/validation/generalization_2026_11_06/analyze.py --data docs/validation/generalization_2026_11_06/run --out docs/validation/generalization_2026_11_06
python -B -X utf8 docs/validation/generalization_2026_11_06/audit.py
```

若只想检查再现性而不覆盖仓库配图，将 `--out` 指向新的 `data/` 子目录即可。中文字体优先 Microsoft YaHei、SimHei 或 Noto Sans CJK SC；不同字体与 matplotlib 环境可能改变 PNG 字节。

## 使用归档实现重新在线执行

默认只准备隔离运行树，不发请求；目标必须是仓库 `data/` 下尚不存在的目录：

```powershell
python -B -X utf8 docs/validation/generalization_2026_11_06/reproduce.py --out data/generalization-reproduction-001
```

若确实需要新请求，先在**进程环境**设置 `PERSONA_API_KEY`，再使用新的目录并加 `--run`。不将密钥写入脚本、配置、人设或研究记录：

```powershell
python -B -X utf8 docs/validation/generalization_2026_11_06/reproduce.py --out data/generalization-reproduction-002 --run
```

该命令使用归档核心和运行器、同一任务方案和210次请求上限。它生成新的固定时间和数据，不覆盖本研究，也不保证远端服务逐字相同。对新输出需要重新组织补充内容复核，不能沿用旧标签。当前工作源码的新研究可使用 `study.py --out <新目录> --prepare --run`，与归档复现明确区分。

## 适用边界

普通提示与默认框架是完整提示包对照，非单组件消融；真实固化在阶段边界显式执行，非默认定时调度或跨日测量。六次更新成功只覆盖两套新数字。人格体验、参数建议行为、判断硬隔离及长期成长仍待独立研究。所有材料均为合成任务，不包含私人账本或认证信息。
