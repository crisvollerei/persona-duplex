# 数据字典与统计单位

## 校准

`calibration_inputs.json`含24项作者材料，匿名ID、context、user、reply是评审可见内容。`calibration_key.json`映射原ID、pair_id、contrast、probe与expected；预期是作者构念假设，不是语义真值。

20项明确材料进入解释门槛；两对探针共4项整对排除，其中也含明确配对参照，不能称四项都是歧义。六个字段逐维报告；明确缺陷只统计明确子集内作者指定的冲突与直接禁令违规。`calibration_gate.semantic_eligible`只控制语义解释强度，不取消机械、状态和成本研究。

两次服务批量标注是同一24项的重复，不是48个独立情境。两份独立自动化评审不等于两个人类或两个独立模型家族；各配置读取同一rubric可见正文，其他系统上下文仍可能不同。

## 主试验

`run/results.json.episodes`为48段轨迹：6模板实例×2人格×2方法×2重复，每段4轮。`records`为192个生成单元，与episodes逐轮一致。六实例共享流程结构，不代表六独立任务；同一轨迹内各轮及同模板重复相关。

- `payload`：该轮实际提供的事实、请求、权限、候选、人格、state、history和draw。
- `state_before`：接受当前输入前的模拟备忘。
- `user_update` / `state_after_user`：应用确认的用户更正或权限变更及其结果。
- `program_selected_id`：程序按共享规则算出的目标，两方法都记录；prompt不会在请求中得到此结果。
- `model_reported_id`：模型JSON自报选择，不是语言行为真值。
- `executed_id`：共同执行器实际接受的候选；权限或请求范围拒绝时为空。program路线同样执行合法的模型报告ID，不自动改成程序目标。
- `state_after_action`：只有作者候选effects写入的模拟结果，不证明现实动作发生。
- `rule_match` / `allowed` / `request_scope_match`：机械对照，不是人格、事实或用户体验成功。
- `say`：实际生成文字，未自动改写；错误或缺失单列，不补样。
- `status` / `execution_error`：模型/结构与执行器错误分别记录；不将全部错误统称网络故障。

两路初态、规则与输入类别一致，但各自选择可使后续state/history不同；不能宣称整段逐字同状态。

## 评审与派生结果

`reviews/inputs.json`按（完整上下文、人格、实际文字）去重，共180项，映射回192个生成单元。提供行动前状态和独立执行器确认的application_result以核对模拟变化；不提供方法、动作ID、抽样值或另一份评分。文字风格仍可能透露来源，元数据隐藏不保证完全盲法。

六维行为编码之外，人格与情境各为1—5等级或null，focus为practical/feeling/detail/mixed/unclear。每位评审单独报告，不合成为客观真值或加权总质量。实际SAY与动作ID可能不一致，事实/边界错误不能由人格高分抵消。

`summary.json`是派生描述；`counterfactual_replay.json`在已观察payload上改变参数或去掉请求范围检查，零新增API、零新增文本，不与在线样本混计。共同状态的反事实选择不证明新模型生成或跨任务泛化。

`calls/*.json`与对应`raw_calls.json.gz`保存同一组请求响应的两种存储，不能重复计调用数。请求模型、实际响应模型、usage、时延分别保存。成本只计本地实验端点，不含端点外自动化评审；48条轨迹的延迟取各自调用耗时之和，不冒充真实用户端到端延迟。

本研究新请求最多194次（2校准+192生成）；此前预检2次另计，全阶段实际调用数应分别列明，旧预检和新校准材料不能合并计算准确率。
