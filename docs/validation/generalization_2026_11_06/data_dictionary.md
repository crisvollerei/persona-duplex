# 数据字典

本研究使用合成输入，真实在线输出；每个请求均归档。不存在个人对话数据或认证请求头。

| 文件 | 含义 |
| --- | --- |
| `run/protocol.json` | 请求前固定的参数、预算、评分、配置及源码哈希 |
| `run/execution_plan.json` | 150 个随机交错单元，含 16 个事实实例的标准答案 |
| `run/results.json` | 按完成顺序追加的观察；以 `trial_id` 与计划连接，不将列表位置当条件 |
| `run/raw_calls.json.gz` | 210 次请求的消息、设置、输出增量、响应元数据、返回用量和耗时；无认证头 |
| `run/runner_snapshot.py` | 实际运行及评分代码，字节哈希对应固定方案 |
| `run/implementation_snapshot/` | 21 份实际运行的核心源码；`manifest.json` 给出逐文件哈希及当时 Git HEAD |
| `summary.json` / `results_tables.md` | 离线重算的统计与表格，不能回写原评分 |
| `format_failure_review_*` | 原始严格评分失败的事后补充复核，单独保留匿名输入、映射、规则、标签和汇总 |
| `audit_report.json` | 流重放、引用、哈希、分母及材料对应检查；不是全部试验成功的证明 |
| `artifact_manifest.json` | 公开研究文件的原始字节 SHA-256，自身除外 |

## 单元、输出与分母

150 单元 = 96 事实单元 + 48 受控记忆单元 + 6 真实更新工作流。186 条输出 = 前两组 144 条 + 工作流训练 24 条 + 新会话探针 18 条。前定结论评分覆盖 162 条；训练输出没有指定结论金标。210 次请求 = 186 次前台生成 + 24 次两阶段记忆固化。

框架协议和交付分母为 138，包含全部 `text` 事实、受控记忆、训练与探针。48 条普通提示输出没有 HEAD/SAY/TAIL，不参与该分母。普通提示的回调内容由相同增量直接拼接，相等不是独立的流式框架检验。

## 核心字段

- 事实与受控记忆单元的 `case` 保存模板、固定实例、问题和金标；`config`、`config_hash` 为实际表达配置。真实更新工作流另存训练、阶段和探针，其表达配置为固定方案中的中性配置。
- `turn.say` 是实际交付文本，`delivered_chunks` 是回调增量；`validity` 检查本地协议。二者均不代表事实真值。
- `evaluation.strict_correct` 同时要求正确目标、单位与末行结论标签。`labelled_content_correct` 仅放宽末行位置；它也不能读取没有标签的正文。两个指标均不评价所有断言。
- `call_ids` 与原始调用的 `trial_id/call_id` 一一连接；`mode` 为 `stream` 或 `text`。`response.model` 是实际返回标识，`request.settings.model` 是请求标识；二者分别报告。
- `response.usage` 是服务返回的用量，不等于实际账单。本次 210 次均返回用量；重跑缺失时不以零填充。
- 受控记忆的 `source=synthetic_control`，原话的 `evidence_scope=specified` 表示研究者精确指定的合成证据，不表示真实学习成功。
- 真实更新的 `training`、`stages`、`probes` 和 `ledger` 保留旧设置、更新、实际固化和隔离探针；`memory_injected` 是探针真正收到的记忆。
- `rebuild_equal` 检查实际重建视图；`reopened_ledger_equal` 检查重开 Store 的重放；`traits_unchanged` 不涉及自动人格成长效果。

计划失败、缺失输出和未执行探针不会从分母删除。当前数据没有执行异常。事后复核只覆盖 27 条未满足严格评分的输出，由一个条件匿名的 Codex 子代理完成，不是人类真值；其标签不覆盖原有评分。
