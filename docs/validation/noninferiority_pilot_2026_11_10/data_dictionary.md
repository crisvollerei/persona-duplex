# 数据与统计口径

## 输入与冻结

- `materials.json`：6个development情境、24个test情境。`facts`为共同资料，`history`为共享合成初态，`query`为本轮输入；`memories`为应用预置的来源与版本记录，`memory_scope`为应用指定范围。
- `profiles.json`：两个人格的自然语言规范、voice及人工设定参数，不是心理量表或完整官方角色说明。
- `prompts.json`：三种直接SAY候选。`protocol.json.hashes`绑定生成、评分、分析代码与输入；`implementation_snapshot/`为实际使用源码。
- `selection.json`：开发集分数、错误评审项计数、选定候选、时间及对应文件哈希。`error_ratings`以评审项计，不等于独立错误个数。

## 输出与成本

`development/results.json.records`含36条，`run/results.json.records`含96条。完整解析与交付记录集中保存于`results.json`，每条以`id`关联同阶段`calls/<id>.json`。`raw_calls.json.gz`逐字保存与calls列表相同的JSON观测，未保存授权头。

| 字段 | 含义 |
| --- | --- |
| case_id / profile_id / method | 情境、人格和生成路线；正式集每情境四条 |
| status | ok为非空实际正文；empty_reply为没有交付正文；error为编排或生成失败 |
| raw / say | 完整模型返回文字与用户实际可见文字，二者不混作质量评价 |
| validity | 框架本地协议检查；直接回复路线为null，不能解释成不合规 |
| mechanical | 预设字面片段、精确JSON值、最终行检查；不判断人格或语义正确 |
| selected_memory_ids / memory_context | 共同检索所得ID与送入生成的完整上下文 |
| evidence_sha256 | facts、history与memory_context的共同摘要，逐配对核对相同 |
| framework_ledger | 实际框架来源、轮次及解析记录；仅框架有此字段 |
| elapsed_s | 单元编排开始至完成的整次等待，包含本地处理，不是首字时间 |

`calls`保存完整request与response，requested model与response model分别报告。HTTP正常完成不等于框架成功交付；usage以提供方返回为准。成本包括本地记录的132次端点生成，不包括两份端点外自动化评审的未计量计算。没有首字流式数据，不把整次等待称作首字时间。

`ground_truth.required_numbers`仅为材料中的语义核对提示，本版`mechanical()`未使用该字段。`final_line`采用逐字匹配，可能因为句号等差异失败；机械通过率与实际履行、事实评价必须分别阅读，不能把标点差异全部称为逻辑错误。

## 评分与判定

`reviews/*_inputs.json`为方法元数据隐藏后的实际输入；`*_key.json`将匿名ID关联到原始record。两个`*_ratings_<coder>.json`分别保留独立自动化任务的原评分，不合并成单一真值。`calibration.json`含作者预期，不能交给评审；`calibration_gate.json`记录预设门控。

人格与情境为1—5或null，其余分类见[量表](rubric.md)。`summary.json.main.primary_quality`有2人格×2维×2评审共8项，每项按24个情境配对。框架减基线为正表示框架较高，主非劣线为−0.30；t与bootstrap下界均保留。

`failure_risk`先判断同人格配对中的框架特有不利事件，再对同情境两人格取any，分母为24。confirmed_labels_only与unknown_and_missing_unfavorable分开；CP精确上界不把零事件解释为零风险。`overall_noninferiority_supported`同时要求表达与稀有风险，不由质量均分替代。

分族每组只有3个情境；局部差异是预定分类的描述，不是经多重比较校正的稳定优势。扩展决定由冻结规则计算，本批不增加生成。
