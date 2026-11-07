# 受约束信息追问：程序策略与提示词决策

[研究总览](../../research.md) · [接口说明](../../inquiry-policy.md) · [论文](paper.md)

本研究将应用审核的问题槽交给程序或提示词选择，再由同一执行器验证并交付。主研究、共同状态探针和强化提示完整流程确认分别保存，共180次API调用、160172个实际总token，147条完整对话输出；72次状态探针不计为对话输出。

程序能够按已定义策略执行选问、等待、拒答与恢复，并省去选择模型调用。原提示曾出现冷却拒收及其连锁反馈；补充精确定义后，强化提示的六条完整流程与程序可见交互逐字一致。该结果不支持“模型做不到”或框架全面优于传统提示聊天。研究范围与局限见[论文](paper.md)。

## 材料导航

| 材料 | 内容 |
| --- | --- |
| [paper.md](paper.md) | 研究问题、三阶段方法、结果、分母及局限 |
| [protocol.md](protocol.md) / [protocol.json](protocol.json) | 主研究方案和冻结文件哈希 |
| [cases.json](cases.json) / [profiles.json](profiles.json) / [inquiry_config.json](inquiry_config.json) | 任务、用户响应规则、关注偏好及选问配置 |
| [selector_prompt.txt](selector_prompt.txt) | 主研究的提示词决策规则 |
| [results.json](results.json) | 24条主流程、144次计划步骤，含33个未交付项 |
| [raw_calls.json.gz](raw_calls.json.gz) / [runtime_ledgers.json.gz](runtime_ledgers.json.gz) | 主研究实际请求响应及运行账本 |
| [review_inputs.json](review_inputs.json) / [review_codebook.md](review_codebook.md) | 方法映射隐藏的完整主研究轨迹及编码规则 |
| [ratings_1.json](ratings_1.json) / [ratings_2.json](ratings_2.json) / [review_key.json](review_key.json) | 两套Codex标签及分析映射，非人类评价 |
| [decision_probe/protocol.md](decision_probe/protocol.md) / [inputs.json](decision_probe/inputs.json) / [results.json](decision_probe/results.json) | 36个共同状态上的两提示配对探针 |
| [decision_probe/clarified_prompt.txt](decision_probe/clarified_prompt.txt) / [raw_calls.json.gz](decision_probe/raw_calls.json.gz) | 强化提示与72次探针实际调用 |
| [clarified_flow/protocol.md](clarified_flow/protocol.md) / [results.json](clarified_flow/results.json) | 强化提示六条完整流程确认，不新增体验编码 |
| [clarified_flow/raw_calls.json.gz](clarified_flow/raw_calls.json.gz) / [runtime_ledgers.json.gz](clarified_flow/runtime_ledgers.json.gz) | 确认阶段36次调用与账本 |
| [summary.json](summary.json) / [audit_report.json](audit_report.json) | 三阶段派生统计及材料关系核查 |
| [data_dictionary.md](data_dictionary.md) | 字段含义、计数单位与各阶段分母 |
| [implementation_snapshot/persona](implementation_snapshot/persona) | 实际评估的冻结实现 |
| [benchmark.py](benchmark.py) / [analyze.py](analyze.py) | 主实验生成与离线统计绘图 |
| [replay.py](replay.py) / [replay_report.json](replay_report.json) | 复用108条已存响应，零网络重放主研究及确认阶段的30条流程 |
| [analysis_environment.json](analysis_environment.json) / [requirements.txt](requirements.txt) | 分析环境和绘图依赖 |
| [artifact_manifest.json](artifact_manifest.json) | 发布文件哈希清单 |

三幅图分别展示[信息取得](figures/fig1_information.png)、[实际交付](figures/fig2_delivery.png)和[共同状态决策](figures/fig3_shared_states.png)。图中的强化提示来自独立保存的确认阶段，不替换原提示结果。

## 数据口径

- 主研究程序与不追问各六条流程、36个计划步骤；原提示十二条流程、72个计划步骤。完整输出为36＋36＋39＝111。
- 原提示的33次未交付均为冷却拒收，包含九条流程中的首次拒收和24次随后连锁拒收，不是33个独立首次错误，也不是API服务失败。
- 共同状态探针为36份固定payload×两种提示，共72次调用，不执行新对话。
- 完整流程确认为六条流程、36次调用和36条完整输出；与主研究相加得到147条完整对话输出。
- 明确答案只计应用已确认的answered原话；拒答和模糊回答不填为答案。双字段记录完整不等于行为全部正确，不完整也可能是尊重拒答的正常结果。
- 两套Codex标签只针对主研究24条完整轨迹；连贯性不等于人格自然度或人类体验，确认阶段没有继承旧标签。
- 所有实际请求标识为deepseek-chat；180次归档响应标识均为deepseek-flash。总token包含输入和输出，不是仅输出token。

原始轨迹使用相同初态、规则和候选，但交互分叉后历史不同。只有共同状态探针逐payload匹配输入。数据是合成任务，不代表真实用户资料。

## 离线核查与绘图

以下命令在仓库根目录运行，不调用在线模型：

```powershell
python -m pip install -r docs/validation/inquiry_policy_2026_11_07/requirements.txt
python -B -X utf8 docs/validation/inquiry_policy_2026_11_07/analyze.py --out data/inquiry-analysis
```

分析读取已保存三阶段数据和两套主研究标签，核查冻结哈希、策略选择、实际执行、模拟用户回应与结果字段，重新生成summary、audit_report及三幅图。绘图使用matplotlib和DejaVu Sans字体，不重新评分。

要重放主研究和确认阶段的全部对话状态，可使用以下零网络命令：

```powershell
python -B -X utf8 docs/validation/inquiry_policy_2026_11_07/replay.py --out data/inquiry-archived-replay
```

它只在完整messages数组逐字一致后复用108条已存选择响应，核对30条流程、180个计划步骤和147条完整交付，不发出新请求，也不构成新增独立样本。

只重跑确定性的程序与不追问流程，可使用：

```powershell
python -B -X utf8 docs/validation/inquiry_policy_2026_11_07/benchmark.py --out data/inquiry-offline-reproduction
```

`benchmark.py`默认`--mode offline`，运行12条六轮流程，不发送API请求。使用本目录冻结源码和新输出目录，原始发布数据不会被覆盖。它不重跑提示词条件，也不自动重跑两个补充阶段。

## 需要明确选择的在线重跑

以下命令会联网并产生API费用；它们不是离线核查步骤。认证通过进程环境`PERSONA_API_KEY`或交互式隐藏输入提供，不写入论文、配置或公开数据。每次使用尚不存在的输出目录。

主研究的全部条件，包括72次原提示决策请求：

```powershell
python -B -X utf8 docs/validation/inquiry_policy_2026_11_07/benchmark.py --mode all --out data/inquiry-online-reproduction
```

固定共同状态探针会发送72次请求，强化提示流程确认会发送36次请求。两个脚本直接执行在线研究，没有默认离线准备模式：

```powershell
python -B -X utf8 docs/validation/inquiry_policy_2026_11_07/decision_probe/run.py --out data/inquiry-probe-reproduction
python -B -X utf8 docs/validation/inquiry_policy_2026_11_07/clarified_flow/run.py --out data/inquiry-confirmation-reproduction
```

每阶段方案和输入分别冻结，调用不自动重试；原失败保留。服务版本、缓存、调度和采样可能改变在线结果，因此源码一致不保证响应逐字相同。新回复不能直接使用本次标签作为新评价；固定探针的选择正确性则可按保存的规则重新计算。
