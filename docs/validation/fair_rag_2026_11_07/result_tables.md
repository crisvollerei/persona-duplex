# 共享检索资料的公平对照：离线统计

主自动指标只评价固定可提取目标；缺失、覆盖与格式分开。条件匿名辅助编码不替代原始评分或人类评价。

| 方法 / 配置 | 计划/返回 | 内容覆盖 | 已分配目标命中下界 | 标签存在 | 末行格式 | 联合严格 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| traditional_neutral | 32/32 | 32/32 | 31/32 | 32/32 | 32/32 | 31/32 |
| traditional_blue | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| traditional_dragon | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| framework_neutral | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| framework_blue | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| framework_dragon | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |

## 目标覆盖内正确及未覆盖界限

| 方法 / 配置 | 覆盖内正确 | 已覆盖明确不匹配数 | 未评分目标数 | 已分配目标成功可能区间 |
| --- | ---: | ---: | ---: | --- |
| traditional_neutral | 31/32 | 1 | 0 | [96.9, 96.9] |
| traditional_blue | 32/32 | 0 | 0 | [100.0, 100.0] |
| traditional_dragon | 32/32 | 0 | 0 | [100.0, 100.0] |
| framework_neutral | 32/32 | 0 | 0 | [100.0, 100.0] |
| framework_blue | 32/32 | 0 | 0 | [100.0, 100.0] |
| framework_dragon | 32/32 | 0 | 0 | [100.0, 100.0] |

## 配置相对中性的可提取命中分数损失与方法差异

损失为本方法中性命中分数减人格命中分数；交互为框架损失减传统损失。正交互表示框架的下降更大，不称为神经推理污染。未覆盖是未评分，命中下界差异不能直接解释为推理错误差异。

| 对比 | 百分点估计 | 95% 八族聚类bootstrap描述区间 | 未评分识别界限 | 簇数 |
| --- | ---: | --- | --- | ---: |
| persona_target_score_loss_traditional_blue | -3.1 | [-9.4, 0.0] | [-3.1, -3.1] | 8 |
| persona_target_score_loss_framework_blue | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 8 |
| persona_loss_framework_minus_traditional_blue | 3.1 | [0.0, 9.4] | [3.1, 3.1] | 8 |
| persona_target_score_loss_traditional_dragon | -3.1 | [-9.4, 0.0] | [-3.1, -3.1] | 8 |
| persona_target_score_loss_framework_dragon | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 8 |
| persona_loss_framework_minus_traditional_dragon | 3.1 | [0.0, 9.4] | [3.1, 3.1] | 8 |
| target_framework_minus_traditional_neutral | 3.1 | [0.0, 9.4] | [3.1, 3.1] | 8 |
| target_framework_minus_traditional_blue | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 8 |
| target_framework_minus_traditional_dragon | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 8 |

## 请求、成本与实际输出

状态：`complete`；实际请求224；完整输出224/224；单元异常0。

| 方法 / 配置 | 请求 | 已知用量条数 | 平均输入tokens | 平均输出tokens | 完整输出均秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| traditional_neutral | 32 | 32 | 1149.00 | 31.56 | 0.96 |
| traditional_blue | 40 | 40 | 1350.90 | 28.27 | 0.98 |
| traditional_dragon | 40 | 40 | 1325.90 | 37.65 | 0.90 |
| framework_neutral | 32 | 32 | 3267.00 | 179.88 | 1.86 |
| framework_blue | 40 | 40 | 3478.90 | 168.93 | 1.81 |
| framework_dragon | 40 | 40 | 3453.90 | 180.50 | 1.91 |

框架协议：112/112（计划），112/112（返回）；交付一致：112/112（返回）。传统提示无相同三段协议，不计入协议分母。

资料包是实体过滤后的全部六份资料，词法排序只影响顺序；不将相同证据输入解释为检索算法质量或压缩优势。

材料对应检查：2091，失败0。

## 条件匿名配对评价

状态：`complete`。

### 评价任务 1

| 方法 / 配置 | 已标回复 | 角色一致性均分 | 自然度均分 | 无依据断言 | 边界标签 yes/no/unclear/不适用 |
| --- | ---: | ---: | ---: | ---: | --- |
| traditional_neutral | 32 | 不评分 | 4.91 | 2/32 | [0, 0, 0, 32] |
| traditional_blue | 40 | 4.22 | 4.85 | 1/40 | [5, 1, 0, 34] |
| traditional_dragon | 40 | 4.58 | 4.42 | 1/40 | [5, 1, 0, 34] |
| framework_neutral | 32 | 不评分 | 4.69 | 0/32 | [0, 0, 0, 32] |
| framework_blue | 40 | 4.42 | 4.92 | 1/40 | [6, 0, 0, 34] |
| framework_dragon | 40 | 4.88 | 4.80 | 2/40 | [6, 0, 0, 34] |

全部配对偏好计数：`{"tie": 77, "traditional": 18, "framework": 17}`。含中性配对，允许平局与不确定；均分不能证明人格非劣。

| 配置 | 传统偏好 | 框架偏好 | 平局 | 无法判断 |
| --- | ---: | ---: | ---: | ---: |
| blue | 5 | 6 | 29 | 0 |
| neutral | 8 | 2 | 22 | 0 |
| dragon | 5 | 9 | 26 | 0 |

| 方法 / 配置 | 已分配/已标注 | correct | incorrect | no_answer | ambiguous | 可判断中正确 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| traditional_neutral | 32/32 | 31 | 1 | 0 | 0 | 31/32 |
| traditional_blue | 32/32 | 32 | 0 | 0 | 0 | 32/32 |
| traditional_dragon | 32/32 | 32 | 0 | 0 | 0 | 32/32 |
| framework_neutral | 32/32 | 32 | 0 | 0 | 0 | 32/32 |
| framework_blue | 32/32 | 32 | 0 | 0 | 0 | 32/32 |
| framework_dragon | 32/32 | 32 | 0 | 0 | 0 | 32/32 |

### 评价任务 2

| 方法 / 配置 | 已标回复 | 角色一致性均分 | 自然度均分 | 无依据断言 | 边界标签 yes/no/unclear/不适用 |
| --- | ---: | ---: | ---: | ---: | --- |
| traditional_neutral | 32 | 不评分 | 4.84 | 2/32 | [0, 0, 0, 32] |
| traditional_blue | 40 | 4.25 | 4.85 | 1/40 | [5, 0, 1, 34] |
| traditional_dragon | 40 | 4.50 | 4.42 | 1/40 | [5, 0, 1, 34] |
| framework_neutral | 32 | 不评分 | 4.59 | 0/32 | [0, 0, 0, 32] |
| framework_blue | 40 | 4.83 | 4.67 | 3/40 | [6, 0, 0, 34] |
| framework_dragon | 40 | 4.92 | 4.75 | 1/40 | [6, 0, 0, 34] |

全部配对偏好计数：`{"tie": 59, "framework": 27, "traditional": 26}`。含中性配对，允许平局与不确定；均分不能证明人格非劣。

| 配置 | 传统偏好 | 框架偏好 | 平局 | 无法判断 |
| --- | ---: | ---: | ---: | ---: |
| blue | 11 | 10 | 19 | 0 |
| neutral | 11 | 2 | 19 | 0 |
| dragon | 4 | 15 | 21 | 0 |

| 方法 / 配置 | 已分配/已标注 | correct | incorrect | no_answer | ambiguous | 可判断中正确 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| traditional_neutral | 32/32 | 30 | 1 | 0 | 1 | 30/31 |
| traditional_blue | 32/32 | 32 | 0 | 0 | 0 | 32/32 |
| traditional_dragon | 32/32 | 32 | 0 | 0 | 0 | 32/32 |
| framework_neutral | 32/32 | 32 | 0 | 0 | 0 | 32/32 |
| framework_blue | 32/32 | 32 | 0 | 0 | 0 | 32/32 |
| framework_dragon | 32/32 | 32 | 0 | 0 | 0 | 32/32 |

### 两套标签的精确一致

| 字段 | 全部标签精确相同 | 排除双方均不适用后的相同 |
| --- | ---: | ---: |
| role_fidelity | 126/160 | — |
| naturalness | 201/224 | — |
| target_verdict | 223/224 | 191/192 |
| unsupported_assertion | 217/224 | — |
| necessary_conditions | 223/224 | 23/24 |
| boundary_respected | 222/224 | 22/24 |
| preference | 83/112 | — |

角色分排除中性null；分类适用分母取至少一套认为适用的回复。大量不适用标签不作为实际语义一致的证据。
