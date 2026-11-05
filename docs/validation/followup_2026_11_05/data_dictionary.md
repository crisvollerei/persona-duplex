# 数据字段与统计边界

## 主结果

`experiment_results.json` 保存 528 个测试单元。其 `records` 中的每个单元都有唯一 `trial_id`，原始评分保持运行时内容；派生分析见 `summary.json` 与 `scoring_amendment.json`。

| 字段 | 含义 |
| --- | --- |
| experiment / condition / case_id | 实验、配置条件与固定任务编号 |
| replicate | 同一条件与任务的四次重复，索引 0–3 |
| order | 本地随机排序后的计划位置；实际完成时间可因并发而不同 |
| status=ok | 执行器完成该单元；不等于协议有效或任务回答正确 |
| turns[].say | 引擎最终保存的回复文本 |
| turns[].delivered_chunks | 实际用户回调收到的逐段文本 |
| turns[].stream_match | 回调拼接与最终 say 是否完全相等；不作内容正确性判断 |
| turns[].valid | 本地结构有效性；普通提示组为 null，不适用 HEAD/SAY/TAIL 指标 |
| turns[].head | 内部参考字段，不是模型真实心理，也不是独立真值 |
| call_ids | 关联 raw_calls.json.gz 中的 global_call_id |
| evaluation.say_correct | 冻结的原实现结论匹配结果；原清洗器误删小数点，不能直接当作修正后主结果 |
| evaluation.direct_correct | 原实现可解析 direct 子集上的匹配，缺失为 null |
| memory_injected | 记忆主实验中研究者控制的判断，证据标为 synthetic_control，不能冒充真实训练所得 |
| run_manifest | 生成配置、实际适配器条件、源码哈希及运行信息 |

分析器在内存中增加 `say_correct_decimal_fixed` 与 `direct_correct_decimal_fixed`，保留小数点后按相同标准答案重算；修正前后差异及 25 个受影响试验在 `scoring_amendment.json` 中。缺少结论行、尖括号等不符合原严格规则的输出仍按原接受范围处理。严格格式匹配与实际所问内容是否答对需要分开解释。

`manifest.local_randomization_seed` 只控制本地计划排序。旧命名字段 `server_seed_supported=false` 在本运行器中表示未配置远端采样种子，并非已经验证提供方不支持种子；对应方案字段为 `server_seed=null`。

## 原始调用

`raw_calls.json.gz` 解压后为 716 条调用的 JSON 数组：

- `request.messages`、`request.arguments` 和 `request.settings` 保存送入适配器的文本、工具 schema、阶段预算覆盖及默认设置；认证请求头不记录。
- `response` 保存提供方可获取的请求标识、实际模型名、完成原因、fingerprint 与 usage；缺失字段不当作零值。
- `chunks` 是提供方返回的文本增量，不是完整网络抓包或全部原始 SSE 字节。
- `output` 为适配器输出，`route` 为 text、stream 或 tool，`status` 为调用执行结果。
- 四个 persona 试验因表达阶段结构无效而省略后续观察，因此实际调用数少于每个都执行三阶段的名义值。

## 补充、标注与分析集

`supplement_results.json` 保存四个端到端记忆重复，各自包含三条真实训练输出、两次实际固化请求和两个空历史探针。全部 20 条对话与 28 次调用单独保存；其 `replay` 字段比较真实账本重建前后状态。

`blind_*_items.json` 是隐藏条件标签的评分输入，`blind_*_ratings.json` 保存内容标签及证据。事实与建议两项任务分别由不同 Codex 子代理完成，每项只有一位评审，没有同项双评。`annotation_summary.json` 公开揭盲后的关联，属于事后探索性模型辅助评价。原始评分不因这些标签被改写。

主分析集仅使用完整独立执行的记录。`attempt_01_partial/interruption.json` 保留未完整采集的预备执行排除信息，其中相互重叠的检查点不能相加，也不与主分析集合并。

运行清单中的 Git 提交标识与逐文件源码哈希应共同使用。`implementation_snapshot/persona/` 保存通过哈希核对的研究实现，历史流式回放使用该快照；现行框架与研究版本的差异只作版本标识，不影响历史材料对应关系的判断。
