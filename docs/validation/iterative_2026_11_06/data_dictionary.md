# 多阶段研究数据字典

本目录只含合成实验问题和实际模型输出。认证信息不写入请求归档。统计所指的完整输出、请求、工作流和标签是不同单位。

## 数量与分母

| 单位 | 数量 | 含义 |
| --- | ---: | --- |
| 原主研究单元 | 504 | R0、R1、R2、B1 |
| 加入独立补充后的单元 | 520 | 另含R3的12条、R4的4条记忆工作流 |
| 计划对话输出 | 680 | 单次格式/行为输出，加每条记忆工作流的3训练及2探针 |
| 实际完整输出 | 664 | 缺失16条保留计划分母；失败时可获得的交付记录另存 |
| 实际API请求 | 791 | 665次前台试次（664完整、1传输失败）及126次固化调用 |
| 有响应模型与用量的请求 | 790 | 另1次传输失败不可将未知用量填零 |
| 行为输出/初始标签 | 143 | 原计划144，一条无完整输出；两评价者各标143条 |

七个失败单元包括六条记忆工作流的候选/来源验证错误，以及一条行为传输错误。失败工作流可能已有部分训练输出，不能把整个单元当成零观察，也不能把未执行探针删去。

## 文件及连接方式

| 文件 | 作用 |
| --- | --- |
| `protocol.json` / `execution_plan.json` | R0前固定的主研究预算、参数、任务、金标与顺序 |
| `r3_protocol.json` / `r3_plan.json` | 看到R2反例后另行固定的六场景补充 |
| `r4_protocol.json` / `r4_plan.json` | 看到R3反例后固定的最终两场景定向补充 |
| `runs/<stage>/phase_protocol.json` | 该阶段使用的版本、运行器哈希、任务哈希、此前请求数 |
| `runs/<stage>/results.json` | 按完成顺序保存单元，以 `trial_id` 连接执行计划 |
| `runs/<stage>/raw_calls.json.gz` | 消息、设置、响应文本增量、元数据、用量和耗时；无认证头 |
| `runs/<stage>/runner_snapshot.py` | 当阶段实际执行及评分逻辑，不用现行代码替代历史版本 |
| `versions/<version>/manifest.json` | 归档源文件字节哈希，包含baseline和revision_A至D |
| `mechanism/*.json` | 固定局部案例的预期与观察，以及原发现性探针 |
| `scoring_amendment.json` | R2前递归JSON类型比较的补充；192个旧分数不变 |
| `summary.json` / `result_tables.md` | 离线派生统计，不回写原结果 |
| `behavior_*` | 编码规则、匿名输入、映射、缺失、两份初始标签及规则触发决策 |
| `failed_delivery.json` | 失败时实际交付观察，不将其伪造成完整TurnResult |
| `failed_workflows.json.gz` | 六条异常记忆工作流的原始追加账本文本及未消费缓冲；含逐文件SHA-256，不恢复或改写源目录 |
| `audit_report.json` | 材料对应、流重放、完整引用和计划分母核查，不是试验成功率 |
| `artifact_manifest.json` | 本研究公开材料的逐文件SHA-256，自身除外；现行框架可以继续演进，不要求永远等于研究快照 |

原始调用以 `trial_id/call_id` 与单元的 `call_ids` 对应。各阶段和版本分别统计，不能按完成列表位置推断条件或时间顺序。

## 单元与输出字段

- 格式单元的 `case` 含族、实例、用户文本、金标、机械契约；`arm` 为baseline、revision_plain或revision_contract。revision_plain仍是默认三段框架，非普通助手提示。`role` 为蓝鱼或白龙表达配置。
- `turn.say` 是最终实际回复，`delivered_chunks` 是回调增量；`stream_match` 记录拼接与回复是否相等，`calls_count` 为该次前台调用数。`validity` 只检查本地协议，不代表语义真值。
- `contract_validation` 只报告调用方显式契约的机械满足状况。未请求契约不是格式失败，也不能从valid=true推出目标事实正确。
- 格式 `evaluation.content_covered` 表示固定提取器能否得到目标结构；`content_correct` 比较金标；`format_ok` 检查要求形状；`strict_correct` 同时要求目标与格式。可提取覆盖不足不自动等于全文事实错误。
- 记忆 `memory_score` 单独解析末行的数字或未知，`content_correct` 为目标正确；严格结果还要求行首结论标签。未单独测量的 `format_ok` 在汇总中为null及 `not_measured_by_memory_score`，不填false。
- 记忆单元的 `training`、`stages`、`probes`、`ledger` 保留真实训练、固化和新会话。`probes.memory_injected` 是实际导入的活跃记忆；空白条件是空列表。
- 操作日志 `judgment_operation` 保留operation、effective、reasons及来源。其 `semantic_status=model_proposal` 不表示语义已获独立证明；真实引用与提议是否理解正确分开。
- `view_rebuilt` 检查实际恢复的判断视图，`traits_unchanged` 检查显式人格配置保持，不测量人格成长。
- `response.model` 是实际返回标识，`request.settings.model` 是请求标识。用量只累加返回的usage字段，缺失不是零。

## 行为标签、缺失与条件匿名

`behavior_review_inputs.json` 只有143条真实id/question/answer；`behavior_review_key.json` 单独保存参数、场景和条件。用户问题会暴露是否求建议，输出语气也可能暴露风格，因此条件匿名不等于完全盲法。

两份 `behavior_ratings_1.json`、`behavior_ratings_2.json` 分别含 `action_advice`、`decision_pressure`、`clarification`、`accepts_refusal` 及理由。前三者是布尔值；拒绝合规只在拒绝组评价，其余为null。两份原标签不合并成真值，两者均属Codex辅助评价，非独立人类评审。

`behavior_missing_outputs.json` 保存一条未完整返回的计划单元。行为汇总给出assigned、observed_outputs、labelled、观察率，以及以计划分母计算的上下界。缺失不当作无建议、无施压或尊重拒绝。κ在全为同一类别时可能未定义；保留该事实，不以1替代。

## 派生统计与复现边界

`analyze.py` 按各阶段运行器快照重新核对保存评分，并按执行计划恢复所有分母。主研究与R3/R4使用独立方案文件。未运行阶段标为not_run，部分完成阶段保留计划分母；本次六阶段均已结束。框架汇总同时提供计划分母和实际完整输出分母，避免将缺失和真实交付不一致混为一类。

Wilson区间按逐输出给出描述；配对bootstrap分别使用格式族、记忆场景和行为场景，仅有六个簇或更少，不代表任意任务总体。行为触发规则使用两位观察率差均值和拒绝建议率均值；它不是显著性检验，也不会改变800次总请求上限。

重新分析和绘图不调用模型；重新在线执行使用归档版本和新输出目录，服务采样可能变化。新输出需重新组织标签，不能沿用本次行为评价。材料审计、图表重绘、统计再现和新实验是不同操作。
