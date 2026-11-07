# 数据字典与计数单位

本材料使用合成中文情境，不包含真人对话或认证密钥。旧研究的原始数据、论文与哈希清单保持不变。

| 文件 | 内容 |
| --- | --- |
| protocol.md / protocol.json | 首次在线请求前冻结的假设、门槛、输入及源码哈希；baseline_origin核验原提示逐字复用。 |
| cases.json | 8个新情境，6个无行为禁令、2个有明确边界；controls是研究者确认的边界注释，不是模型心理推断。 |
| profiles.json | 与前轮相同的三种匿名操作化目标，只用于离线选择与覆盖统计；生成和标注模型均不可见。 |
| common_prompt.txt / baseline_prompt.txt / coverage_prompt.txt / label_prompt.txt | 实际共同规则、两生成条件和分离标注提示。 |
| mechanism.py / selection.py / benchmark.py | 实际执行的研究原型；前者编译覆盖意图、重编号、验证引文结构并选择，后者解析原始候选。 |
| run/results.json | 16个池记录。每池包括candidates（生成文本和self标签）、labels（验证后的分离标签）、label_id_mapping（匿名编号对应关系）和selections（两标签来源×三配置）。 |
| run/raw_calls.json.gz | 全部实际请求/响应。request含完整messages、模型和采样设置；response保留服务原始JSON及usage；elapsed_s为单次请求耗时。无Authorization值。 |
| run/manifest.json / execution_plan.json | 执行环境、方案哈希、固定顺序；并发完成顺序不保证与排程一致。 |
| review_codebook.md / reviews/inputs.json | 两个评价任务共享的规则和去重匿名文本；不含生成条件、计划、任何标签来源或目标人格。 |
| reviews/ratings_1.json / ratings_2.json | 两套独立Codex文本编码，不是人类评价或独立模型家族；保留差异。 |
| review_key.json | 匿名ID与（case,text）的映射；references将它关联到pool_id/candidate_id，评价者不读取。 |
| summary.json / audit_report.json | 从原始材料与两套标签重算的统计与材料核查。 |
| analysis_environment.json / artifact_manifest.json | 分析环境及最终文件哈希；清单不包含自身。 |
| replay.py / replay_report.json | 仅在完整请求逐字匹配后重放归档响应，零新增网络调用、零新增样本。 |

## 字段含义

`generation`只有baseline、coverage。`label_source`只有self、evidence；它们共享同一池文本。行为数值0/1/2的含义见codebook，不是心理量表。

每条`labels`记录中的`proposal`是原始分离标注提议；`attention`和`initiative`是经过结构及逐字片段校验后的值。`null`表示没有可用标签，可能来自模型主动保留未知或程序拒收。`validation_errors`保留拒收原因，不触发重试或回退自标签。引用存在不证明语义解释正确。

`focus_evidence`中kind为content或experience，quote必须是候选原文连续片段；混合关注需两种kind和不同片段。`initiative_quote`在标签为0时必须是整条回复；这种格式约束并不证明全文真的没有推进行为。

`selections`保存确定性选择：仅当两维均有标签时才有资格参与距离比较，选最小距离、同分按原candidate_id升序，原文输出。没有可用选项时显式abstained，不虚构回复或填零距离。程序没有依据盲评质量事后过滤候选。

## 分母与成本

8情境×2生成=16池；每池6条，共96候选。它们并非96个独立场景。每池2种标签来源×3种配置，共96次离线选择；这些选择不是新生成，也不是新用户交互。32次API调用为16次池生成和16次分离标注。

候选自然度、适切性和可用性按生成条件统计，每条件48条，不因两种标签来源翻倍。主覆盖比较是六个开放情境×三配置=18个目标单元；两个有边界情境单列。

标签指标同时报告全计划候选分母下的精确匹配（每生成条件48）、非unknown覆盖和已取得标签中的一致率。验证失败/未知不算全分母成功。评价者自己无法判别时也不当作正确标签，另列评价覆盖。选择距离均值必须附实际选择覆盖；没有选择不按0距离处理。

单次通常部署的self路径承担完整池生成成本，evidence路径承担完整池生成加分离标注成本。研究对三个配置复用同池，不代表产品每条只需承担三分之一；六条候选也不是六次用户交付。两阶段请求耗时相加仅为重构量，不代表真实用户端到端时延。Codex评价任务的不可取得token费用不计为零。
