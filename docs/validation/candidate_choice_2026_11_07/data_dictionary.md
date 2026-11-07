# 数据字典

所有情境为合成材料。没有真人对话、用户隐私或 API 密钥；时间是程序读取的实际 UTC 时间。研究中的候选选择是独立原型，不是现有 Session 的线上调用。

| 文件 | 单位与主要字段 |
| --- | --- |
| cases.json | 12个情境；history/user 是对话，facts/boundaries 是研究者提供且所有方法共享的信息。evaluation_notes 仅说明标注边界，不发送给生成模型。 |
| profiles.json | 三种匿名偏好配置；target 为两维工程目标，description 为相同含义的中文描述。不是心理量表得分。 |
| protocol.json | 首次请求前冻结的方案、输入、提示、程序哈希；fixed_at 是冻结时间。 |
| pools/results.json | 每情境一个池记录；status 是接口和严格JSON解析状态，candidates 是未经语义过滤的完整模型候选。attention/initiative 是模型自标签。 |
| main/results.json | 若初审通过才存在；每条是情境×配置×方法条件。say 为实际保存的最终文本，selected_id 指向候选，direct 无候选ID。status 非ok 时不虚构交付文本。 |
| pools/raw_calls.json.gz、main/raw_calls.json.gz | 各阶段实际请求/响应列表。request 包含完整messages及采样参数；response 保存服务原始JSON、模型标识和usage；不保存Authorization。elapsed_s 是本次端点耗时，不含排队、标注和研究调度。 |
| 各阶段 manifest.json、execution_plan.json | 环境、方案哈希、阶段关联及固定顺序；并发返回顺序可能不同。server_seed=null 表示没有要求远端确定性。 |
| reviews/pool_inputs.json、pool_groups.json | 去掉目标人格、自标签和方法的匿名候选评价材料。文本ID对应唯一（情境、文本）；组ID只供候选覆盖初审。 |
| reviews/ratings_pool_1.json、ratings_pool_2.json | 两套独立Codex实际文本编码；null表示行为维度不可判定，不能当0或1。评分及判据见review_codebook.md。 |
| reviews/gate_ratings_1.json、gate_ratings_2.json | 独立候选可用性及语义取舍初审，规则见candidate_review_codebook.md。 |
| gate.json | 整批继续门槛的计算结果及输入哈希；不包含逐条候选过滤。 |
| reviews/direct_inputs.json、ratings_direct_1.json、ratings_direct_2.json | 主对照新增文本及独立编码；与候选文本相同的输出沿用同一条文本编码，避免重复评价。 |
| review_key.json、pool_group_key.json | 分析用映射，编码者不读取。candidate_ids 是同场C01—C06，output_ids 是条件ID。多个条件可共用一个文本ID。 |
| summary.json | 从原始记录与两套编码重算的统计；未执行阶段不会被补成完成。random_sensitivity为100个固定种子对已编码候选的离线索引诊断，不是新增对话或已执行主对照。protocol_planned_main_outputs为方案原定144条件，main_planned_outputs为实际启动阶段的排程量，本次为0。 |
| audit_report.json | 输入哈希、条件覆盖、请求重建、选择和原文保留等材料检查；通过不等于人格效果成立。 |
| analysis_environment.json | 离线分析与绘图环境。 |
| artifact_manifest.json | 最终材料的逐文件SHA-256；不包含自身。 |

API调用数、实验条件数、候选数和唯一文本数必须区分。12个池最多产生72条候选；四方法各36条件，共144个计划输出，其中大量文本复用；主对照最多调用72次模型，而非144次。首次池生成与主对照最多合计84次请求。

token来自服务usage，输入和输出之和是total_tokens；未提供的字段不推算为零。部署成本为每次回复完整承担候选生成或直接生成成本；实验中共享池的摊销数字另列。Codex标注调用发生在代理任务中，不计入本研究端点调用数，其不可取得的费用不伪装为已测量。

本材料不包含随机种子在服务端生效的保证，不测量流式交付，也不将自标签、精确公式执行或无模型重写解释为语义安全保证。
