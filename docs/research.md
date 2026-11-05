# 研究资料与复现

[文档导航](README.md) · [项目首页](../README.md)

本页按研究版本组织论文、数据与复现入口。设计目标、实际观测和机制解释分别报告；本地协议有效、指定答案匹配和模型检查通过，都不能代替全文事实正确性或人格效果的独立评价。

## 研究目录

| 材料 | 关注的问题 | 数据范围 | 入口 |
| --- | --- | --- | --- |
| 初步实证研究 | 人格配置、生成协议、关系修复、记忆和显式参数推动 | 65 条对话输出及参数状态快照 | [论文](validation/validation_study_paper.md) · [数据](validation/experiment_results.json) |
| 重复对照与边界验证 | 九条件答案交付、单因素参数、受控记忆、局部检查器及真实固化补充 | 528 个主单元；主实验与补充共 516 条对话输出、744 次请求 | [论文](validation/followup_2026_11_05/paper.md) · [完整材料](validation/followup_2026_11_05/README.md) |
| 框架修订的定向验证 | 原话证据保留、指定格式和流式交付一致性 | 36 个单元、52 条对话输出、60 次请求；另重放 400 条历史框架流 | [验证报告](validation/framework_revision_2026_11_05/README.md) · [数据](validation/framework_revision_2026_11_05/results.json) |

对话输出数与测试单元数不同：局部检查器任务有请求和结果标签，但不计用户对话输出；真实记忆重复则包含训练、固化和新会话探针。各材料中的分母以对应方案为准，不将不同指标合并成一个正确率。

## 结果怎样解读

重复对照研究使用相同的服务接口模型生成中性、蓝鱼和白龙表达条件。请求标识为 `deepseek-chat`，归档响应标识为 `deepseek-flash`；两种社区参考人设不是两个底层模型品牌的性能比较。人设资料、来源边界及研究者操作化见[社区人设参考](validation/followup_2026_11_05/persona_sources.md)。

论文同时报告严格结论格式、所问结论内容、协议结构、交付一致性、记忆细节和状态结果。事后模型辅助标注与预先固定指标分开保存；它们不是人类真值，也不能证明全文全部断言正确。

框架定向验证中，载入真实固化记忆后的精确双数字召回为 4/4，空白对照回答未知为 4/4；在线交付与保存文本一致为 52/52，本地协议有效为 50/52。该小样本覆盖具体合成条件，仍不足以证明一般安全性、长期记忆效果、人格与判断的因果独立或人格成长。

## 核查保存的数据

后续研究保存逐任务记录、实际调用、可获得的响应元数据、文本增量和运行源码哈希。请优先阅读：

- [数据字典](validation/followup_2026_11_05/data_dictionary.md)：字段含义、分母、缺失与评分边界。
- [统计汇总](validation/followup_2026_11_05/summary.json)及[结果表](validation/followup_2026_11_05/results_tables.md)：可重新计算的派生结果。
- [固定方案](validation/followup_2026_11_05/protocol.md)及[真实固化补充方案](validation/followup_2026_11_05/supplement_protocol.md)：条件与测量对象。
- [归档源码](validation/followup_2026_11_05/implementation_snapshot/README.md)：论文实际评估的实现，不要求现行代码永远不变。
- [材料审计](validation/followup_2026_11_05/audit_report.json)：检验材料之间的对应关系，检查通过不等于所有实验成功。

运行时的私人账本不应直接发布。仓库中的研究材料使用合成任务；公开逐轮记录与源码快照用于核查这些特定实验。

## 离线重算与在线重跑

以下命令在仓库根目录执行。核心框架运行时无第三方依赖，研究绘图额外依赖 matplotlib：

```powershell
python -m pip install -r docs/validation/followup_2026_11_05/requirements.txt
python -B -X utf8 docs/validation/followup_2026_11_05/analyze.py
python -B -X utf8 docs/validation/followup_2026_11_05/audit_results.py
python -B -X utf8 docs/validation/framework_revision_2026_11_05/analyze.py
```

离线分析重算已有材料，不调用模型。绘图会受字体环境影响，优先使用 Microsoft YaHei、SimHei 或 Noto Sans CJK SC。

已在进程环境设置 `PERSONA_API_KEY` 后，可把后续对照实验独立重跑到一个尚不存在的目录：

```powershell
python -B -X utf8 docs/validation/followup_2026_11_05/reproduce.py --out data/reproduction-001 --implementation study --run
```

`--implementation study` 使用归档的研究实现，`--implementation current` 使用现行框架。两个选项都构建独立运行树，保留原始发布证据。未传 `--run` 时只准备新目录，不发请求。详细操作见[研究目录说明](validation/followup_2026_11_05/README.md)。

源码、条件与原始请求支持复核本次结果；远端采样、服务版本、缓存及调度仍可能变化，在线重跑不保证逐字相同。内容标注需针对新输出重新组织，不能沿用旧标签。

初步研究的配图可通过 [generate_figures.py](validation/generate_figures.py) 重绘，需要 Pillow；配图中的固定示意与适用边界见该论文。各研究都应区分绘图复现、材料核对和重新执行在线实验。
