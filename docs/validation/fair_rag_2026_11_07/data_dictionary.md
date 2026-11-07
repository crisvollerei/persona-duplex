# 数据字典与统计口径

本目录保存`fair_rag_2026_11_07`研究材料。JSON使用UTF-8；原始调用为gzip压缩的UTF-8 JSON数组。时间戳为带时区的ISO 8601，记录时间为UTC+8；耗时以单调时钟计算，单位为秒。文本长度按Python `len`计Unicode字符，不称作汉字数或tokens。

## 1. 观测层级与关联

| 层级 | 键 | 数量 | 说明 |
| --- | --- | ---: | --- |
| 问题 | `case_id` | 20 | 16个事实问题，4个开放交流问题 |
| 任务族 | `family` | 8个事实族及开放组 | 内部研究元数据，不进入被测模型可见资料 |
| 来源实体 | `entity` / `context_id` | 20 | Q101等不含任务语义的编号 |
| 来源文档 | 文档`id` | 120 | 每实体六份，供两方法共同查看 |
| 试次 | `trial_id` | 224 | 一个问题×方法×表达配置×重复 |
| 实际调用 | `trial_id` + `call_id` | 224 | 本研究每试次恰调用一次 |
| 左右配对 | 配对`id` | 112 | 同问题、同配置、同重复的两方法实际SAY |
| 原始编码 | 配对`id` + `a`/`b` | 每套224个回复 | 两套编码单独保存 |

`method`为`traditional`或`framework`，`profile`为`neutral`、`blue`或`dragon`，`replicate`为0或1。不能把同题的配置和重复视为独立任务样本。执行结果按完成顺序保存，不能依数组位置关联条件；应使用明确的键。

## 2. 冻结任务与来源

### `protocol.json`

保存研究标识、前置固定时间、请求模型和采样设置、并发数、计划与硬调用上限、主次指标、停止规则及材料哈希。`seed`用于本地分配，`server_seed`为null；两者不表示服务端生成可精确重放。`runner_sha256`对应执行时的脚本快照。`profile_hashes`和`hashes`用于核查人格及其他固定材料。

### `cases.json`

数组每项代表一道问题。

| 字段 | 含义 |
| --- | --- |
| `id`、`family`、`variant` | 内部问题键、任务族和实例编号 |
| `entity` | 实际引用的无语义实体编号 |
| `user` | 原始用户输入，实际请求逐字使用 |
| `kind` | 自动答案类型；`open`不设事实金标 |
| `gold` | 研究参考结论，仅供离线评分，不提供给生成模型或匿名编码者 |
| `required_source_ids` | 完成问题所需来源，用于资料包存在性检查 |
| `required_conditions` | 题目要求保留的关键条件，供语义评价；自动词面命中另报 |

### `corpus.json`与`retrieval.json`

语料保留合成来源正文和内部诊断字段。`retrieval.json`按问题键保存实际上下文、来源ID、排序、资料包长度及哈希。`documents`可以含内部元数据，`visible_documents`仅含模型实际可见的`id`、`subject`、`date`和`text`。原始`context`是追加到请求中的完整资料数据块，`context_sha256`是其UTF-8内容哈希。

每实体恰有六份资料，top-k为六，且未触发长度截断，因此本次获得完整的实体资料包。`required_sources_present`只说明预定来源在包中，不说明资料可靠、检索在大语料中表现良好或模型实际使用了全部资料。

### 其他固定文件

`execution_plan.json`保存全部224个指定试次；`configurations.json`是三配置的完整展开值；`traditional_prompts.json`保存传统模板；`personas/`保留两个人格文件。`implementation_snapshot/`保留执行使用的框架代码，`manifest.json`列出各源码哈希。`runner_snapshot.py`是实际执行器副本。

## 3. 执行结果与原始调用

### `results.json`

根字段包含`started_at`、`completed_at`、`protocol_sha256`、`records`、`calls_count`、`raw_sha256`、`request_errors`和`reserved_calls`。`records`中每项为一个试次：

| 字段 | 含义 |
| --- | --- |
| `trial_id`及设计字段 | 关联执行计划与问题、方法、配置、重复 |
| `status` | 执行状态；不能因失败或缺失删除计划试次 |
| `config_hash`、`context_sha256` | 实际配置与资料包的对应哈希 |
| `say` | 实际对用户可见的完整回复 |
| `delivered_chunks` | 实际交付增量；框架仅含可见SAY增量 |
| `delivery_match` | 拼接交付增量是否等于最终SAY |
| `protocol_valid` | 框架三段协议有效性；传统为null |
| `validity`、`head`、`warnings` | 框架解析与诊断结果；不作独立事实真值 |
| `evaluation` | 冻结自动提取器产生的评分，见下表 |
| `elapsed_s` | 完整试次处理时间，含本地组织与解析；不是首字延迟 |
| `calls_count`、`call_ids` | 实际前台调用数及关联键 |
| `error`、`failed_delivery` | 若执行失败，保存异常及已交付内容 |

### `raw_calls.json.gz`

解压后每项是一条实际模型调用，包含`trial_id`、`call_id`、`request.messages`、`request.settings`、`context_sha256`、开始时间、状态、原始`chunks`、完整`output`、调用级`elapsed_s`和`response`。

`request`不含认证头。`response`保留实际返回模型、响应指纹、结束原因及用量。请求模型`deepseek-chat`与返回模型`deepseek-flash`分别报告，不用请求名替换观测到的模型名。框架的原始`output`包含完整三段，`results.records[].say`只含可见部分。

用量字段中的`prompt_tokens`、`completion_tokens`和`total_tokens`采用服务返回值。缺失用量或字段保留为未知，不填零、不外推账单。框架completion包含HEAD与TAIL，所以其生成tokens不能只由SAY长度推算。

## 4. 自动评分字段与分母

| `evaluation`字段 | 类型与含义 |
| --- | --- |
| `extracted` | 从实际回复提取的目标内容；可为空 |
| `content_covered` | 是否成功提取可评分目标 |
| `target_correct` | 已覆盖时的目标匹配布尔值；未覆盖为null |
| `target_success` | 已分配试次的可提取命中指示；未覆盖为false，用于下界统计 |
| `conclusion_label_present` | 是否存在要求的结论标签 |
| `format_ok` | 是否满足结论末行形式 |
| `strict_correct` | 目标命中与末行格式联合满足 |
| `required_condition_words_present` | 预定必要条件的词面命中；不适用为null，不是语义真值 |

事实指标的总体分母为192，每方法96，每方法×配置32。开放部分32个输出不设事实目标，相关指标分母为0、比率为null。覆盖内正确率以实际已提取数为分母；全分母命中率以全部计划事实试次为分母。没有输出不等于没有错误，也不算成功尊重用户边界。

协议有效性的计划分母仅为112个框架试次。交付一致性分别报告计划与实际返回分母。未知、失败、未覆盖、明确目标错误和格式错误保持不同类别。

## 5. 方法匿名配对与原始编码

### 固定输入与解码

| 文件 | 内容 |
| --- | --- |
| `review_protocol.json` | 编码规则和分配的前置固定哈希、评价者性质 |
| `review_codebook.md` | 字段规则及允许值 |
| `pair_assignment.json` | 生成前确定的左右分配，包含方法解码 |
| `review_contexts.json` | 以无语义实体ID去重的问题、实际可见来源、关键条件及创作表达参考 |
| `pair_inputs.json` | 每对`id`、`context_id`、`profile`及`a`/`b`实际SAY；无方法、金标或自动分数 |
| `pair_key.json` | 配对到试次与方法的解码表，不提供给编码者 |
| `pair_missing.json` | 计划配对、实际完整导出数与缺失原因 |

### `pair_ratings_1.json`与`pair_ratings_2.json`

各文件是112项数组，每项包含`id`、`a`、`b`、`preference`、`reason`。`a`与`b`的字段为：

| 字段 | 允许值及口径 |
| --- | --- |
| `target_verdict` | 事实题为correct、incorrect、no_answer、ambiguous；开放题为not_applicable |
| `target_answer` | 仅抽取回复实际表达的所问结论；不得自行代答；未知与不适用保留文本标签 |
| `unsupported_assertion` | 布尔值；观察全文中来源或输入未支持的现实、心理、能力或确定性断言 |
| `necessary_conditions` | yes、no、not_applicable；按关键条件是否实际保留判读，不要求机械重复已完成条件 |
| `boundary_respected` | yes、no、unclear、not_applicable；用于明确拒绝建议、否认情绪读法或拒绝替其决定的语境 |
| `role_fidelity` | 蓝鱼、白龙为1—5整数；中性必须为null；不按目标正确直接加分 |
| `naturalness` | 1—5整数；评价回应自然度和语境贴合程度 |

`preference`为A、B、tie或unclear，`reason`为一句主要理由。偏好不是角色分，也不是人格质量非劣结论。两个文件分别分析；分歧保留，不经讨论合成标签。

语义目标判读与自动提取并列，`target_answer`精确对照只作为额外抽取诊断；二者都不覆盖原始自动字段。必要条件和边界统计单列适用数，不能把大量not_applicable混入成功率分母。两套编码的精确标签一致率包含分类字段和评分完全相等的比例，不是两任务独立性或评价正确性的证明。

## 6. 派生汇总与图表

`summary.json`由`analyze.py`生成，保存设计完成度、分组事实与开放结果、调用及用量、资料包检查、每项一致性检查、自动目标对比、匿名评价、图表路径和解释边界。`result_tables.md`为其中文可读表格。原始结果和原始标签均不回写。

`audit.py`单独核查材料并重放112个框架原始流；`audit_report.json`保存独立检查项、检查数、失败数及重放覆盖。它与分析器的2091项检查有不同范围，不将两者相加解释为行为样本。`review_inputs_manifest.json`保存实际给评价者的输入哈希；`artifact_manifest.json`记录最终发布材料的相对路径、字节数及SHA-256，可用于检查归档后是否发生变化。哈希说明字节一致，不说明内容真实或方法优越。

`requirements.txt`固定绘图依赖。`reproduce.py`默认在新的`data/`子目录准备同证据重采样环境，只有显式`--run`才调用在线模型；`reproduction_origin.json`记录新副本来源及是否请求网络。新输出与新标注不覆盖本目录归档。

主自动对比以各方法自身中性条件为参照，按八任务族配对聚类bootstrap4000次；蓝鱼与白龙分别报告。人格评分差异按配对统计，开放题在评分聚类中作为开放组，簇均值加权与逐输出均值的口径不同，不能混用。所有区间均为当前固定样例的探索性描述，不用于确认性显著、等效或非劣检验。

五幅图分别呈现目标与格式、中性参照交互、tokens与处理时间、两套人格评分、配对偏好及语义边界。图中返回用量均值仅依据有该字段的实际调用；若将来出现缺失，图表不得用零替代未知。
