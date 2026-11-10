# 数据字典

- `calibration_inputs.json`：24项，id为混排匿名ID；context为固定资料，user为当前输入，reply为待评价原文。
- `calibration_key.json`：匿名ID映射到original_id和四维expected。原ID的P01—P12为配对，a/b为变体。作者预期不是语义真值。
- `calibration_reviews/ratings_*.json`：每位独立自动化评审编码者24项，四维及reason。fulfills_request与adds_agenda允许bool/null；fact_status与boundary_status各为三态。两任务不等于两名人类或两个独立模型家族。
- `calibration_run/calls/`与`raw_calls.json.gz`：同一两次实际调用的逐文件版与压缩集合。包含请求messages、参数、响应元数据/正文、usage、起止及耗时，不保存Authorization。两种存储不计为四次请求。
- `calibration_run/results.json`：两次服务编码的结构化结果与协议哈希；每次24项。输入顺序与服务采样同时变化，不能识别单独的顺序效应。
- `calibration_gate.json`：两独立自动化评审结果各自的字段、变化配对、负对照、作者指定缺陷计数、差异明细与passed；decision遵守事先条件。backend_repeats只诊断、不参与门控。
- `summary.json`：派生统计。`field_correct`为符合作者预期，不命名为客观准确率；`no_request_null`为作者预期无请求项上null的复现；`codex_exact_agreement`、`backend_repeat_agreement`是相互一致，不能替代真值。
- `costs.actual_requests`只计本地实验服务请求；`total_tokens`只加可得usage，不包括独立自动化评审编码成本。requested model与response model分别解释。
- 独立自动化评审读完整量表和说明，服务API读`mechanism.LABEL_SYSTEM`精简指令；模型、指令详略与执行形式混杂，不能将两配置分歧单独归因为后端能力。
- `main=null`：主试验没有执行，不是全部失败，也不能把144计划单元计入实际样本。
- `cases.json`、`profiles.json`：主试验的计划材料；候选kind/focus/effects和人格数值由作者设计，没有作为观察结果。
- `protocol.json`：首次调用及盲评之前的冻结文件SHA-256和参数；`artifact_manifest.json`：交付时全部研究文件的字节哈希，排除清单自身，两者职责不同。

配对变化只检查作者预期改变的行为字段，两项都匹配才算该对复现；负对照要求同对两项的四维均匹配。null在本批同时允许“不适用”和“不确定”，论文指出该设计的局限；不回写原数据来消除混用。unknown与冻结预期不同时不计一致，但不称作客观错误。
