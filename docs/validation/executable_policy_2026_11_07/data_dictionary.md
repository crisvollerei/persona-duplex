# 字段、阶段与分母

## 首阶段

`protocol.json`和`execution_plan.json`固定70个单元：64个受控输出、6个自动提取工作流，最多76次计划请求。`results.json`保留全部70项，其中69完整、一项提取后拒收。`raw_calls.json.gz`解压为75条真实请求，均不含认证头；请求模型与实际响应模型分别保存。

| 结果字段 | 含义 |
| --- | --- |
| id/case_id/profile/method/replicate/kind | 对应预先指定单元，controlled与automatic分开 |
| content | 受控部分的相同手写ContentFrame；自动部分为真实提取结果，失败时可为null |
| policy_config | 四个数值与固定种子；不作为心理量表 |
| state_before/state_after | 应用初始化及实际结束后的结构化状态 |
| input_hash | 用户、来源、内容帧和初始状态的统一序列化哈希；同对照必须相同 |
| decision | 程序选择，传统为null；不能当作实际执行标签 |
| say/delivered_chunks/delivery_match | 实际完整回复与交付增量；失败只有partial |
| status/error_type | 执行状态，解析或控制拒收与请求异常区分 |
| calls_count | 实际模型请求数，失败不补齐到计划数 |
| elapsed_s | 完整单元处理时间，不是首字延迟 |

原始调用包含trial_id、call_index、stage、messages、settings、真实output及表达chunks、时间、状态和response。response保留模型、指纹、用量和结束原因。首阶段64条手写内容帧的表达调用每方法32次；6条自动工作流实际11次请求，不能将两者混为相同调用成本。

## 状态补充

`state_validation/protocol.json`固定4项工作流、8条输出及12次请求上限，使用自己的实现快照。每项control为普通文本提取与自由表达；关闭重开后probe给出结构化内容帧，再表达一次。初始状态各自为空，程序参数相同。

`results.json.workflows[].stages[]`保留phase、status、say、decision、calls_count、state、permission与处理时间。permission按预先指定的UTC+8受控时间计算，不表示真的等待到次日。cases中的expected_permission/revision仅供离线核查，不进入模型请求。

原始调用按确定执行顺序关联每个阶段的calls_count。协议的temperature/max_tokens描述表达配置；提取实际为温度0、1300tokens，逐请求settings与执行器保留真实条件。全部12次请求另算，不能把补充状态4/4当作正文执行4/4。

## 事后受限重放

`bounded_replay/results.json`是原32个程序条件的离线再执行，network_calls为0。realization为bounded_atoms的16条是新组合文本；free_model的16条为旧响应复用。后者只有完整messages逐字相同时才接受，byte_checked_reused_requests记录16项通过。

original_id关联原单元；original_reply_sha256绑定旧文本；unchanged_reply说明是否相同；replayed_expression_calls仅是离线复用次数，不是新增API调用，不重复计算服务用量。新语句不作为模型生成结果伪装，旧模型回复不称作独立重复。

32项重放中只有16条文本改变，新的匿名编码只覆盖这16条。同16条原自由回复的自然度从原标签关联计算；不是将新16条与不匹配的原32条比较。重放执行器按实际运行字节保存在runner_snapshot.py；reproduce_bounded.py在隔离树中使用这些字节与归档实现，原公开结果不覆盖。

## 匿名编码

首阶段review_inputs包含69条完整回复；未返回输出不伪造、不判为“没有建议”。review_key将匿名ID对应单元，评价任务不可见。ratings_1/2各保存全部69条的advice、focus、target、unsupported、naturalness和reason。

受限重放另有16条输入、解码和两套标签。两个任务重复参与后续编码，不称为独立人类评价；原69条标签不改写，分歧不通过协商消除。

| 统计 | 分母 |
| --- | --- |
| 无边界开放交流的实际建议 | 每方法/介入值8条（2场景×2关注×2重复） |
| 关注符合配置 | 每方法/关注值12条（3个非事实问答场景×2介入×2重复） |
| 已有建议禁令的实际建议 | 每方法8条 |
| 事实目标保持 | 每方法8条，提供正确内容参考，不评价从头推理 |
| 首阶段自然度及无依据断言 | 每方法32条受控输出 |
| 受限表达新回复 | 16条，其中禁令子组8条 |
| 状态补充 | 4个独立工作流，不与语言标签混合 |

首要关注的mixed与unclear分别保留；建议unclear不当作no；未知状态不填成功；自然度不等于角色质量。全文附加断言、指定目标、状态和机械交付分别计量。

## 派生材料

summary.json由analyze.py重算；图表由同一脚本生成。所有比例都是固定小样例的描述，没有确认性显著、等效或非劣结论。全文研究分母保持各自定义，不合成一个总人格成功率。
