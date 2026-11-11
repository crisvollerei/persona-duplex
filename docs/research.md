# 研究资料与复现

[文档导航](README.md) · [项目首页](../README.md)

## 当前研究进展

**[人格工程能否做到与强提示相近？从对话体验、动作归属到状态证据](validation/synthesis_2026_11_11/paper.md)**

覆盖上次发布之后的九项研究。先读正文中的三张表：真人愿意收到哪条回复、谁执行了动作且是否真的完成、缩短生成路径节省多少用量。已测任务上的接近不等于整体非劣。

[逐项观测](validation/synthesis_2026_11_11/observations.json) · [离线复算](validation/synthesis_2026_11_11/reproduce.py)

## 历次研究

| 阶段综述 | 覆盖的问题 |
| --- | --- |
| [从配置执行到表达不劣（11 月 10 日）](validation/research_synthesis_2026_11_10.md) | 情境策略执行、默认框架与优化强提示的非劣性先导比较 |
| [人格工程能控制什么（11 月 7 日）](validation/research_synthesis_2026_11_07.md) | 判断与事实、记忆更正、行动选择与开放人格表达 |

### 情境执行与非劣性阶段的三项研究

| 材料 | 关注的问题 | 数据范围 | 入口 |
| --- | --- | --- | --- |
| 请求履行与额外推进 | 同文换请求、分维编码、边界明确性和后端复用 | 12对24项手写材料，两份自动化编码，2次服务批量标注；主对照未执行 | [论文](validation/context_behavior_2026_11_10/paper.md) · [数据与复现](validation/context_behavior_2026_11_10/README.md) |
| 情境执行与实际表达 | 请求存在性、共同执行器、配置执行与实际文字、用户更正、反事实重放 | 24项新校准、48段四轮对话、192条回复、194次新请求；六个同模板实例 | [完整阶段论文](validation/situated_policy_2026_11_10/paper.md) · [数据与复现](validation/situated_policy_2026_11_10/README.md) |
| 与优化强提示的非劣性比较 | 独立开发集选基线、人格与情境分别检验、稀有失效界和真实成本 | 36条开发回复、24个新情境两人格共96个正式单元；132次请求，不扩样 | [论文](validation/noninferiority_pilot_2026_11_10/paper.md) · [数据与复现](validation/noninferiority_pilot_2026_11_10/README.md) |

### 本阶段的九项分研究

| 材料 | 关注的问题 | 数据范围 | 入口 |
| --- | --- | --- | --- |
| 新实例与记忆更新 | 新数字和实体、冲突与撤销、真实固化后的更正召回 | 150 个单元、186 条输出、210 次请求，四幅图 | [论文](validation/generalization_2026_11_06/paper.md) · [数据与复现](validation/generalization_2026_11_06/README.md) |
| 多轮机制与人格行动 | 回答契约、来源操作、开发与保留题、双任务匿名行为编码 | 520 个单元、680 条计划输出、664 条完整输出，791 次请求，七幅图 | [论文](validation/iterative_2026_11_06/paper.md) · [材料](validation/iterative_2026_11_06/README.md) |
| 共享资料下的强传统提示对照 | 两方法各自的中性参照、相同人格语义和资料包、匿名表达编码与实际成本 | 192 条事实输出、32 条开放回复，224 次请求，五幅图 | [论文](validation/fair_rag_2026_11_07/paper.md) · [数据与复现](validation/fair_rag_2026_11_07/README.md) |
| 可执行策略与受限表达 | 程序选择、状态权限、实际动作偏离及受限组合的体验取舍 | 78条计划在线输出、77条完整输出，87次请求；32项零新增API重放，四幅图 | [论文](validation/executable_policy_2026_11_07/paper.md) · [数据与复现](validation/executable_policy_2026_11_07/README.md) |
| 有限组合与多轮体验 | 参数化句式、交付历史避重与整段自然度 | 18个六轮条件、108条主输出，实际12份不同对话；另108条机械重放，零API请求，三幅图 | [论文](validation/composed_expression_2026_11_07/paper.md) · [材料](validation/composed_expression_2026_11_07/README.md) |
| 受约束的信息追问 | 应用问题槽、主动选问、拒答与未知、共同执行器及提示说明 | 180次请求；主研究及确认147条完整对话输出，另72个决策探针，三幅图 | [论文](validation/inquiry_policy_2026_11_07/paper.md) · [材料](validation/inquiry_policy_2026_11_07/README.md) |
| 自然候选与匿名行为偏好 | 实际语言取舍、候选覆盖和模型自标签的独立编码 | 12次请求、72条候选；预检停止，主对照未执行，四幅图 | [论文](validation/candidate_choice_2026_11_07/paper.md) · [数据与复现](validation/candidate_choice_2026_11_07/README.md) |
| 候选覆盖与标签修订 | 原生成与语义计划、生成自标签与分离证据标注的配对验证 | 32次请求、96条候选、96次离线选择；局部指标改善，质量与标签门槛未全通过，四幅图 | [论文](validation/candidate_revision_2026_11_07/paper.md) · [数据与复现](validation/candidate_revision_2026_11_07/README.md) |
| 人格条件化评价 | 人格可见、方法隐藏；人格、情境、事实逻辑及明确边界分开评价，风格不作硬门槛 | 36次请求、24条最终回复；84条在线文本与12项手写校准，四幅图 | [论文](validation/persona_evaluation_2026_11_07/paper.md) · [数据与复现](validation/persona_evaluation_2026_11_07/README.md) |

### 上次提交已收录的前期研究

| 材料 | 关注的问题 | 数据范围 | 入口 |
| --- | --- | --- | --- |
| 初步实证研究 | 人格配置、生成协议、关系修复、记忆和显式参数推动 | 65 条对话输出及参数状态快照 | [论文](validation/validation_study_paper.md) · [数据](validation/experiment_results.json) |
| 重复对照与边界验证 | 九条件答案交付、单因素参数、受控记忆、局部检查器及真实固化补充 | 528 个主单元；主实验与补充共 516 条对话输出、744 次请求 | [论文](validation/followup_2026_11_05/paper.md) · [完整材料](validation/followup_2026_11_05/README.md) |
| 框架修订的定向验证 | 原话证据保留、指定格式和流式交付一致性 | 36 个单元、52 条对话输出、60 次请求；另重放 400 条历史框架流 | [验证报告](validation/framework_revision_2026_11_05/README.md) · [数据](validation/framework_revision_2026_11_05/results.json) |

对话输出数与测试单元数不同：局部检查器任务有请求和结果标签，但不计用户对话输出；真实记忆重复则包含训练、固化和新会话探针。各材料中的分母以对应方案为准，不将不同指标合并成一个正确率。

## 事实、记忆与强提示对照

重复对照研究使用相同的服务接口模型生成中性、蓝鱼和白龙表达条件。请求标识为 `deepseek-chat`，归档响应标识为 `deepseek-flash`；两种社区参考人设不是两个底层模型品牌的性能比较。人设资料、来源边界及研究者操作化见[社区人设参考](validation/followup_2026_11_05/persona_sources.md)。

论文同时报告严格结论格式、所问结论内容、协议结构、交付一致性、记忆细节和状态结果。事后模型辅助标注与预先固定指标分开保存；它们不是人类真值，也不能证明全文全部断言正确。

框架定向验证中，载入真实固化记忆后的精确双数字召回为 4/4，空白对照回答未知为 4/4；在线交付与保存文本一致为 52/52，本地协议有效为 50/52。该小样本覆盖具体合成条件，仍不足以证明一般安全性、长期记忆效果、人格与判断的因果独立或人格成长。

新实例研究中，事实任务的指定末行结论为普通提示35/48、默认框架37/48，不同表达配置的差异方向不一致；六条真实更新工作流的新会话均召回新数字，空白对照均答未知。框架协议有效136/138，交付一致138/138。27条严格格式失败输出另有事后条件匿名的单一 Codex 内容复核，原始评分保持不变；这不是全文事实正确率或人类人格评价。

多轮研究对格式、记忆和人格行动分别测量。无歧义格式任务出现天花板，不能据此宣称契约提高生成正确率；源码更正与撤销由来源和主题关联控制，原始退步及中断均保留。`intervention` 高低值在未请求建议组出现不同的具体行动建议率，双份 Codex 编码及其分歧同时报告，不替代人类体验评价或人格与判断硬隔离的证明。

共享资料研究给强传统提示与默认框架提供逐字相同的证据包及相同人格语义。两方法的蓝鱼、白龙条件各为32/32个目标命中；唯一自动目标错误位于传统中性条件，不能据此解释为人格条件优势。框架实际总token用量为传统方案的2.723倍。两套方法匿名Codex评价中，整体偏好接近、平局最多；角色评分和自然度分开保留，不能证明人类体验非劣。该研究使用短合成资料和实体内全部六份文档，不测量检索质量、RAG压缩收益或长上下文饱和。

## 程序动作与实际交付

可执行策略研究进一步区分动作记录与实际执行：在已确认建议禁令组，强提示和程序选择后的自由生成均有8/8条建议；受限组合的同组为0/8，但同16条回复的Codex自然度均分下降。后者是事后离线编译，另16条只在请求逐字一致时复用旧回复，不是新的在线重复。受限执行只覆盖特定非建议动作，事实问答与其他自由动作不获得同样保证。

有限表达的多轮对照呈现负结果：相邻新撰句式重复27/45降为0/45，两套Codex评价的语义重复感仍均为4.67/5；自然度从3.00变为2.78／3.00。亲近和正式风格可以区分，不等于整体体验改善。旧方法三个配置副本完全相同，每方法九条件不能当作九个独立场景。候选composed保持显式实验选项，默认表达不变。

信息追问研究将偏好选择与共同权限执行分开。程序流程取得10个明确答案、尊重2次预设拒答；原提示的33次冷却拒收含24次连锁拒收，不能当作独立失败。固定共同状态中，明确解释计数后的提示36/36符合预期，后续六条完整流程也与程序可见交互逐字相同。结果支持对已定义策略作可复算执行，未证明模型不能遵守这些规则；程序路径省去本试验中的逐轮选问调用，不包含内容理解、问题审核或语义确认成本，也不证明一般人格或真实用户体验优势。

## 核查保存的数据

后续研究保存逐任务记录、实际调用、可获得的响应元数据、文本增量和运行源码哈希。请优先阅读：

- [数据字典](validation/followup_2026_11_05/data_dictionary.md)：字段含义、分母、缺失与评分边界。
- [统计汇总](validation/followup_2026_11_05/summary.json)及[结果表](validation/followup_2026_11_05/results_tables.md)：可重新计算的派生结果。
- [固定方案](validation/followup_2026_11_05/protocol.md)及[真实固化补充方案](validation/followup_2026_11_05/supplement_protocol.md)：条件与测量对象。
- [归档源码](validation/followup_2026_11_05/implementation_snapshot/README.md)：论文实际评估的实现，不要求现行代码永远不变。
- [材料审计](validation/followup_2026_11_05/audit_report.json)：检验材料之间的对应关系，检查通过不等于所有实验成功。

运行时的私人账本不应直接发布。早期研究主要使用合成任务，公开逐轮记录与源码快照用于核查这些特定实验；真人研究仅公开匿名数值、方法和来源哈希，原文、理由与私有执行记录不自动发布。

## 离线重算与在线重跑

以下命令在仓库根目录执行。核心框架运行时无第三方依赖，研究绘图额外依赖 matplotlib：

```powershell
python -m pip install -r docs/validation/followup_2026_11_05/requirements.txt
python -B -X utf8 docs/validation/followup_2026_11_05/analyze.py
python -B -X utf8 docs/validation/followup_2026_11_05/audit_results.py
python -B -X utf8 docs/validation/framework_revision_2026_11_05/analyze.py
python -B -X utf8 docs/validation/generalization_2026_11_06/analyze.py --data docs/validation/generalization_2026_11_06/run --out data/generalization-analysis
python -B -X utf8 docs/validation/generalization_2026_11_06/audit.py
python -B -X utf8 docs/validation/iterative_2026_11_06/analyze.py --out data/iterative-analysis
python -B -X utf8 docs/validation/iterative_2026_11_06/audit.py
python -B -X utf8 docs/validation/fair_rag_2026_11_07/analyze.py --out data/fair-rag-analysis
python -B -X utf8 docs/validation/fair_rag_2026_11_07/audit.py
python -B -X utf8 docs/validation/executable_policy_2026_11_07/analyze.py --out data/policy-analysis
python -B -X utf8 docs/validation/composed_expression_2026_11_07/analyze.py --out data/composed-analysis
python -B -X utf8 docs/validation/inquiry_policy_2026_11_07/analyze.py --out data/inquiry-analysis
python -B -X utf8 docs/validation/inquiry_policy_2026_11_07/replay.py --out data/inquiry-replay
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/analyze.py --out data/candidate-choice-analysis
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/replay.py --out data/candidate-choice-replay
python -B -X utf8 docs/validation/candidate_revision_2026_11_07/analyze.py --out data/candidate-revision-analysis
python -B -X utf8 docs/validation/candidate_revision_2026_11_07/replay.py --out data/candidate-revision-replay-new
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/analyze.py --out data/persona-evaluation-analysis
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/replay.py
python -B -X utf8 docs/validation/persona_evaluation_2026_11_07/audit.py
```

离线分析重算已有材料，不调用模型。绘图会受字体环境影响，优先使用 Microsoft YaHei、SimHei 或 Noto Sans CJK SC。

已在进程环境设置 `PERSONA_API_KEY` 后，可把后续对照实验独立重跑到一个尚不存在的目录：

```powershell
python -B -X utf8 docs/validation/followup_2026_11_05/reproduce.py --out data/reproduction-001 --implementation study --run
```

`--implementation study` 使用归档的研究实现，`--implementation current` 使用现行框架。两个选项都构建独立运行树，保留原始发布证据。未传 `--run` 时只准备新目录，不发请求。详细操作见[研究目录说明](validation/followup_2026_11_05/README.md)。

源码、条件与原始请求支持复核本次结果；远端采样、服务版本、缓存及调度仍可能变化，在线重跑不保证逐字相同。内容标注需针对新输出重新组织，不能沿用旧标签。

共享资料对照的隔离重跑使用其归档实现与固定方案；默认只准备目录，追加`--run`才会发送请求。具体材料及操作见[该研究说明](validation/fair_rag_2026_11_07/README.md)。

初步研究的配图可通过 [generate_figures.py](validation/generate_figures.py) 重绘，需要 Pillow；配图中的固定示意与适用边界见该论文。各研究都应区分绘图复现、材料核对和重新执行在线实验。
