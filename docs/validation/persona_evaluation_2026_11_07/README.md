# 人格、情境与逻辑：条件化评价试验

[阶段进展](../../research.md#当前研究进展) · [论文](paper.md) · [评价规范](../../evaluation.md) · [24 条最终回复](final_replies.md)

在六个相同情境中，按行为标签选择候选，相比直接生成提高了守岸人的人格均分，却降低了活泼同伴岚的均分。要了解原因与解释边界，可对照阅读两条[最终回复](final_replies.md)：R053 是获得两位评审人格满分的记录式回应；R030 的人格和事实评价一致，主动性却分别被标为 0 和 2。

**样本与口径：**12 条直接回复与 72 条候选合计 84 条在线文本，其中 24 条最终回复进入主比较；12 项手写校准单列。两项 Codex 评审各编码 96 项。在线生成与标注共 36 次请求、49,167 token，无重试或补样。请求标识为 `deepseek-chat`，归档响应均标识 `deepseek-flash`。

## 材料

| 材料 | 内容 |
| --- | --- |
| [论文](paper.md) | 研究问题、方法、分维度结果、图表与限制 |
| [协议](protocol.md) / [冻结哈希](protocol.json) | 调用前固定的设计、实现和材料 |
| [量表](rubric.md) / [共同评审说明](review_instructions.md) | 评价锚点、风格与错误区分、逐字证据 |
| [人格](profiles.json) / [情境](cases.json) | 明确适配规格、人工选择目标与合成材料 |
| [结果](run/results.json) / [完整调用](run/raw_calls.json.gz) | 最终文本、候选、自标签与证据标签、请求响应、usage |
| [匿名评审项](reviews/inputs.json) / [来源映射](review_key.json) | 按情境、人格和文字去重的输入及来源 |
| [编码者1](reviews/ratings_1.json) / [编码者2](reviews/ratings_2.json) | 独立Codex评价与证据，保留分歧 |
| [校准项](fixtures.json) / [作者预期](fixture_key.json) | 同文换人格与明确缺陷；不混入在线样本 |
| [统计](summary.json) / [数据字典](data_dictionary.md) | 所有分母、未知、缺失及字段定义 |
| [归档重放](replay_report.json) / [材料审计](audit_report.json) | 请求和选择可复算、材料关联、历史完整性 |
| [交付清单](artifact_manifest.json) | 文件哈希，不含清单自身 |

## 离线复核

从仓库根目录执行，以下命令不发送API请求：

```powershell
python -m pip install -r docs/validation/persona_evaluation_2026_11_07/requirements.txt
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/test_mechanism.py
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/replay.py
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/analyze.py --out data/persona-evaluation-analysis
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/export_replies.py
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/audit.py
```

`analyze.py --out`支持把派生统计与图表写入另一个目录；图表为PNG与SVG。重放只使用归档响应，不产生新观察。评审编码属于原始研究数据，离线计算不重新制造或覆盖评分。

在线重跑应使用新输出目录，脚本从终端安全读取临时密钥或使用进程环境变量，不把密钥保存进材料：

```powershell
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/benchmark.py --out data/persona-evaluation-new-run --run
```

服务端未固定随机种子，新调用不预期逐字一致。本目录的分析默认引用归档样本；新样本应另建评审输入、映射和独立编码，不自动并入已有统计。`freeze.py`是原研究固定材料的入口，已存在的冻结协议不重写。
