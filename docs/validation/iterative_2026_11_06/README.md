# 输出契约、证据操作与人格行动：多阶段研究

[论文](paper.md) · [固定主方案](protocol.md) · [统计结果](result_tables.md) · [数据字典](data_dictionary.md) · [研究目录](../../research.md)

2026-11-06，完成修订前诊断、同期开发对照、保留题复验、行为实验及两项独立补充。共520个计划单元、680条计划输出，实际791次请求、664条完整输出；硬上限800次，结束后不追加采样。归档基线与A/B/C/D五个版本均保留。

| 主要结果 | 证据边界 |
| --- | --- |
| 格式336/336严格通过 | 三轮均有天花板，不能证明新增契约提高生成成功率 |
| 离线记忆基线21/42，最终D42/42 | B同样42/42，却在保留题真实记忆仅2/6，对照6/6 |
| C与同期基线真实记忆均5/6 | 失败场景不同，没有优越性证据 |
| D两个定向更新场景及其对照均2/2 | 只支持局部兼容验证，不代表六场景或一般记忆效果 |
| 高intervention未请求建议率较高 | 两份条件匿名Codex标签分别报告，非人类真值；保留1条缺失 |
| 完整返回交付664/664，协议619/680计划输出 | 可读回复、协议元数据与可采纳状态分别评价 |

## 研究材料

- [主协议](protocol.json)及[504单元执行计划](execution_plan.json)。R3、R4各自有独立的[方案与计划](r3_protocol.json)、[最终定向方案](r4_protocol.json)，不替换原R2结果。
- [各阶段原始结果与完整调用](runs)：`results.json`、`raw_calls.json.gz`、`runner_snapshot.py`、`phase_protocol.json`。
- [五版源码与逐文件哈希](versions)、[表达配置](personas)。
- [统计汇总](summary.json)、[自动表格](result_tables.md)、[七幅图](figures)、[材料审计](audit_report.json)。
- [离线机制案例](mechanism)及[运行脚本](mechanism_checks.py)：42组记忆和20种契约形状，原发现性探针另存。
- [行为规则](behavior_codebook.md)、[前置评价方案](behavior_rating_protocol.json)、[143条匿名输入](behavior_review_inputs.json)、[映射](behavior_review_key.json)、[缺失报告](behavior_missing_outputs.json)、[评价1](behavior_ratings_1.json)、[评价2](behavior_ratings_2.json)、[补充未触发决策](behavior_decision.json)。
- [JSON评分补充](scoring_amendment.json)：前两轮192个原分数重核无变化，旧标签保持。
- [六条异常工作流的原始账本与积压](failed_workflows.json.gz)：合成实验材料，独立保存，不改写原始失败结果。
- [公开材料逐文件哈希](artifact_manifest.json)：保存本目录原始字节；运行时目录和该清单自身除外。

## 离线重算、绘图与审计

在仓库根目录执行，不调用模型：

```powershell
python -m pip install -r docs/validation/iterative_2026_11_06/requirements.txt
python -B -X utf8 docs/validation/iterative_2026_11_06/analyze.py --out data/iterative-analysis-check
python -B -X utf8 docs/validation/iterative_2026_11_06/audit.py
```

`analyze.py` 默认读取本目录，`--out` 可指定新的派生输出目录。未指定时重算本研究的统计和七幅图，不修改原始 `runs/`、评分或标签；`--no-plots` 仅用于不生成图的检查。中文字体与matplotlib环境可能改变PNG字节。

`--export-behavior` 从真实完整输出生成匿名评价输入；已存在且内容不同的输入不会被覆盖。新实验必须重新评价，不能把本次143份标签套到新输出。缺失不补造，不视为无建议或尊重拒绝。

## 归档实现重跑

先准备隔离副本，目标为仓库 `data/` 下尚不存在的目录：

```powershell
python -B -X utf8 docs/validation/iterative_2026_11_06/reproduce.py --out data/iterative-reproduction-001
```

默认不发请求。确需重跑时，在进程环境设置 `PERSONA_API_KEY`，选择另一个新目录并加 `--run`。该操作使用各阶段归档版本与运行器，不覆盖本研究；它是重新执行归档对照，不是重新开展自适应的代码发现研究。服务采样、版本和缓存变化使逐字一致无法保证。

所有阶段共享800次请求硬上限，重跑仍不得并发启动多个阶段进程。本次补充的可行性依据各阶段当时的实际已用请求数；新运行的固化调用数可能不同，触及上限时可能无法完成所有补充，不能为完成全阶段自动提高预算。来源验证失败和异常保留，不自动重试或替换。密钥不得写入脚本、配置或研究材料。

## 如何理解改进

新增契约只检查显式要求的机械格式，不评判事实真伪、不重写已交付文本、不增加默认模型调用。有来源的记忆修订/撤销保留提议、来源和生效/拒绝原因，完整原话不等于模型解释正确。默认仍为一次前台 `HEAD → SAY → TAIL`，记忆另行固化。本研究没有自动人格成长、长期真人体验或人格与判断的硬隔离证明。
