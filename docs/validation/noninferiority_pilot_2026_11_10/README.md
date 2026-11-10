# 常见情境下的人格表达：与优化强提示的非劣性先导研究

[论文](paper.md) · [完整对话](transcripts.md) · [研究目录](../../research.md)

本研究预设人格与情境各0.30分的可接受差距，将表达质量与事实、边界、请求履行分开。先在6个独立开发情境、两个人格上比较三种强提示候选，选定少样本提示；再以24个新情境、两个人格比较默认三段生成与直接SAY，共96条正式回复。两路获得相同事实、历史、人格语义和逐字相同的检索上下文，各一次前台生成。

**主要结果：**两份评审均有36/48配对的框架等级不低于基线，但8项共同主要质量下界均未过预设界值，岚的均差也低于容许差距。框架有一次字面协议误用导致空回复，若干范围控制退步被共同标记；平均总token约为强提示1.94倍、整次等待约2.06倍。按冻结规则暂停扩样，先定位机制与适用范围，不把少数照应正例称为稳定优势。

实际端点生成132次：开发36次、正式96次，不重试、不按结果补样。两份方法标签隐藏的自动化评审不是人类体验评价；非劣界也是作者预设的工程阈值，不是经用户验证的容忍差距。24个手写情境不足以证明罕见失效低于5%，也不代表全部真实使用分布。

## 材料

| 文件 | 内容 |
| --- | --- |
| [论文](paper.md) / [事先方案](protocol.md) / [冻结清单](protocol.json) | 研究问题、非劣标准、扩展规则与源文件SHA-256 |
| [材料](materials.json) / [材料说明](materials_notes.md) / [人格规格](profiles.json) | 6开发、24正式情境，8族、两人格 |
| [三种强提示](prompts.json) / [选择结果](selection.json) / [开发集全部回复](development_transcripts.md) | 有限开发预算内的基线优化，不是理论最优声明 |
| [统一量表](rubric.md) / [校准材料](calibration.json) / [校准结果](calibration_gate.json) | 等级锚点、分类、作者预期与独立编码门控 |
| [开发数据](development/results.json) / [完整调用](development/raw_calls.json.gz) | 36条实际开发回复及完整请求响应 |
| [正式数据](run/results.json) / [完整调用](run/raw_calls.json.gz) / [完整对话](transcripts.md) | 96个分配单元、全部实际正文和失败 |
| [匿名评审输入](reviews/main_inputs.json) / [映射](reviews/main_key.json) | 评分所见完整必要上下文，方法元数据隐藏 |
| [评审1](reviews/main_ratings_1.json) / [评审2](reviews/main_ratings_2.json) | 独立自动化任务的原始分维评价，保留分歧 |
| [统计](summary.json) / [描述性补充](supplementary_summary.json) / [数据字典](data_dictionary.md) / [环境](analysis_environment.json) | 8项共同主要终点、配对计数、置信界、成本和解释范围 |
| [审计](audit_report.json) / [文件清单](artifact_manifest.json) | 数据关联与最终字节校验，不证明语义真值 |

## 图表

| 图 | 内容 |
| --- | --- |
| [图1](figures/fig1_quality_noninferiority.png) | 分人格、分评审的人格与情境差值及不劣界 |
| [图2](figures/fig3_generation_costs.png) | 正式集实际token与整次等待，首字未测 |

论文保留质量区间与生成成本两幅主图；分族差异和风险界以正文及统计数据呈现。每图同时提供SVG。图表从归档评分与调用重算，不用手填的柱高替代数据。

## 离线复核

```powershell
python -m pip install -r docs/validation/noninferiority_pilot_2026_11_10/requirements.txt
python -B -X utf8 docs/validation/noninferiority_pilot_2026_11_10/test_benchmark.py
python -B -X utf8 docs/validation/noninferiority_pilot_2026_11_10/analyze.py --self-test
python -B -X utf8 docs/validation/noninferiority_pilot_2026_11_10/analyze.py --out data/noninferiority-reproduction
python -B -X utf8 docs/validation/noninferiority_pilot_2026_11_10/audit.py
```

重算不调用模型，可生成全部派生图。新位置的`summary.json`应与归档一致；字体、绘图库或SVG元数据可能影响图像字节，原图字节由交付清单保存。在线复现须使用新研究副本和新输出，重新报告模型版本与随机采样；不覆盖已封存结果。临时密钥仅从进程环境读取，未写入研究材料。
