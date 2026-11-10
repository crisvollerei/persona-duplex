# 人格策略的情境执行与语言交付

[论文](paper.md) · [完整对话](transcripts.md) · [研究目录](../../research.md)

本阶段把请求存在性、请求履行、额外议程、边界明确性、边界遵守及事实分开，先用新材料校准，再比较程序先选行动与相同规则的强提示。主研究完成48段四轮合成对话、192条回复；两方法均一次调用选择或表达、使用同一执行器，且不重试。程序目标、模型报告、实际模拟状态与最终文字分别评价。

**主要结果：**配置规则匹配为程序96/96、强提示60/96；36次差异中26次仍是同类候选。两份实际文字评价均认为程序的岚人格评分略高、守岸人略低，没有跨人格一致优势；两路均完成所测明确请求、遵守边界。一个强提示回复的“核对尚未开始”被共同记为事实依据未知，不能算正确。新研究194次端点请求、230,175 token；加前期预检为196次、236,195 token，不含端点外评审成本。

主研究的六例是同一四轮结构的不同对象、数字和标题实例，不是六种独立任务。第三轮用户明确限制，第四轮更正初始草案；候选效果只修改模拟备忘，不执行现实事务。两份独立自动化评审提供分维语义评价，不是人类体验验证。

## 材料

| 文件 | 内容 |
| --- | --- |
| [论文](paper.md) / [协议](protocol.md) | 研究问题、事先分支、结果和适用范围 |
| [统一量表](rubric.md) | 服务端标注和独立评审看到的相同可见说明 |
| [校准输入](calibration_inputs.json) / [作者预期](calibration_key.json) | 24项新材料：20明确项、4个探针对照项；预期隐藏 |
| [校准评审1](calibration_reviews/ratings_1.json) / [评审2](calibration_reviews/ratings_2.json) | 独立自动化编码，保留分歧 |
| [端点校准](calibration_run/results.json) / [完整调用](calibration_run/raw_calls.json.gz) | 同一量表的两次批量重复 |
| [解释分支](calibration_gate.json) / [冻结哈希](protocol.json) | 明确子集的语义解释条件与全部源文件摘要 |
| [情境](cases.json) / [人格](profiles.json) | 六个模板实例、96个逐轮候选与两个研究人格 |
| [主试验结果](run/results.json) / [完整请求响应](run/raw_calls.json.gz) | 48轨迹、192回复、状态及实际调用 |
| [完整对话](transcripts.md) | 全部实际回复及三类行动ID，不只选有利样例 |
| [匿名评审输入](reviews/inputs.json) / [映射](review_key.json) | 完整必要历史、人格与执行器确认，隐藏生成方法和动作ID |
| [主评审1](reviews/ratings_1.json) / [主评审2](reviews/ratings_2.json) | 两份实际文字编码及分维等级 |
| [评审上下文契约](review_context_protocol.json) | 主生成前固定，提供独立应用结果以核对模拟状态声明 |
| [统计](summary.json) / [反事实重放](counterfactual_replay.json) | 分组结果、成本、共同状态下的零新增API诊断 |
| [归档重放](replay_report.json) / [审计](audit_report.json) / [文件清单](artifact_manifest.json) | 原始调用重建、材料关联及文件哈希 |
| [数据字典](data_dictionary.md) / [分析环境](analysis_environment.json) | 字段、分母、依赖版本及范围 |

## 离线复核

从仓库根目录执行；测试、重放和审计只需Python标准库，绘图依赖另列：

```powershell
python -B -X utf8 docs/validation/situated_policy_2026_11_10/test_mechanism.py
python -B -X utf8 docs/validation/situated_policy_2026_11_10/replay.py
python -B -X utf8 docs/validation/situated_policy_2026_11_10/audit.py
python -m pip install -r docs/validation/situated_policy_2026_11_10/requirements.txt
python -B -X utf8 docs/validation/situated_policy_2026_11_10/analyze.py --out data/situated-policy-analysis
```

论文保留配置执行及实际质量与成本两幅主图；校准和逐轮计数见正文表格。`analyze.py --out`在新位置重算统计和完整派生PNG/SVG；不调用模型、不生成评分。端点外评审计算成本不包含在本地API token统计中。更换字体或绘图库版本时图像字节可能变化，原始结果和统计口径应保持。

在线重跑应另建研究目录与输出，不把新服务版本、随机采样或新评分合并进归档。`benchmark.py`必须显式`--run`并指定不存在的输出目录，临时密钥从进程环境或安全终端读取，不保存授权头。复用本量表仍需重新评分和报告实际模型标识，不能借用旧评分证明新输出。
