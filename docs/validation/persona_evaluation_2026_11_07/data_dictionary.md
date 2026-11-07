# 数据字典

## 固定材料

- `protocol.md/json`：设计、计数、调用预算、评审与分析方法、材料哈希和实际执行源码范围。
- `profiles.json`：两份完整人格规格及人工目标映射；`shorekeeper_source.toml`为所依据的仓库配置快照。人格评审使用文字规格，不看到`target`。
- `cases.json`：六个合成场景；`history/user`是来源，`facts`是来源摘要，`boundaries`是明确原话。`controls`驱动覆盖提醒；`evaluation_notes`是作者的材料说明，不发送给生成器或评审。
- `rubric.md`：新量表，风格不构成质量门槛。历史自然度评分不在这里改写。
- `fixtures.json`：12个手写校准项。`fixture_key.json`保存作者预期与同文换人格配对；评审不可见。校准不是在线样本或一般准确率真值。

## 在线与选择记录

`run/results.json.records`每项对应一个情境、人格、方法条件，共计划24项。`method=direct`有一次生成；`coverage`有一次六候选生成和一次分离标签调用。`status`与`label_status`分别保留失败。

- `say`：该条件最终文本；候选路线严格取所选原文，无后续润色。
- `candidates`：池中六条文字及生成自标签。
- `labels`：分离标注器的原始提议、逐字引用校验错误及接受的逐维标签；`null`不回退成自标签。校验不能证明语义正确。
- `selected_id`：确定性L1规则所选候选。规则使用固定人格目标，不感知任务例外，不执行事实或边界过滤。生成阶段的约束与选择器本身的能力必须分开。
- `random_selected_id`：同池单次确定性伪随机抽样，用于零新增调用的离线诊断；不属于主24条最终回复。
- `label_id_mapping`：候选原ID与标注时随机别名的映射；标注请求不含原自标签、人格或目标坐标。
- `generation_call/label_call`：对应`run/calls`及压缩归档中的实际请求ID。

`run/raw_calls.json.gz`保存完整请求/响应、输出、时间、usage、请求模型和响应模型。请求授权头不保存；`run/manifest.json`与`execution_plan.json`记录运行环境、计划顺序与预算。并发请求的完成顺序不同于计划顺序，记录ID用于连接数据。

## 评审与统计

`reviews/inputs.json`按(情境、人格、文字)去重并混排在线文本与校准项。输入只包含匿名ID、人格文字规格、上下文与回复。`review_key.json`用于恢复来源及候选映射；不同人格不能共享人格评分。两份`ratings_*.json`为独立方法隐藏的Codex编码，完整保留证据与分歧。

`persona_fit/context_fit`为1–5整数或未知；`grounding`为supported/violation/uncertain；`boundary`为respected/violation/uncertain。支持仅表示提供材料范围内未发现问题，不宣称全面事实正确。边界结果须区分存在明确禁令与未设置禁令；无禁令的respected不作为额外成功样本。

`style`为描述性多标签；`attention/initiative`是实际行为的独立编码。两个行为整数不等同人格符合度，不把机械感、字数或泛用自然度转成质量门槛。`evidence`含逐字定位与解释。

主比较按两人格×两方法、每组6个场景分别报告每位评审；同池随机选择12项另列。候选标签匹配以候选总数为分母，未知不记正确并另报；候选数不代表独立场景数。评分差异属于这批合成场景和同类模型评审，不是人类体验或总体效应。

成本使用真实usage，每条候选路线计入完整pool+labels，不除以候选数；同池离线随机诊断无额外请求。报告的路线时延是顺序阶段实际耗时之和，非用户侧端到端计时。

`summary.json`与四图来自`analyze.py`。`replay_report.json`核验归档请求/选择；`audit_report.json`核验材料关联、证据引用、凭证模式及旧研究完整性；它们通过不表示实验结果全部良好。`analysis_audit_report.json`是派生统计自身的校验。`artifact_manifest.json`列出交付文件哈希，排除清单本身。
