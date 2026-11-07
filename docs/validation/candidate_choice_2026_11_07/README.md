# 自然候选选择与匿名行为偏好

[研究总览](../../research.md) · [论文](paper.md) · [研究方案](protocol.md) · [数据字典](data_dictionary.md)

本研究在十二个合成社交情境中检验自然候选能否承载显式行为偏好。目标人格不可见的模型生成六条候选，程序或强提示从相同候选选择，另设强提示直接生成和固定随机选择。原型独立保存在本目录，不修改框架默认会话路径。

评价针对实际文字，候选自标签只作机制记录。两套独立Codex编码、分歧、失败及成本均保留，不替代人类体验研究或长期人格验证。完整方法和继续门槛在首次在线调用前冻结。

实际完成12次请求、72条候选，共9637 token。两套初审均为63/72条可用、9/12池有至少两种适切取舍，略低于冻结门槛65条与10池，因此在候选阶段停止，未执行后续72次请求及144个主对照条件。没有由本试验得出的程序对强提示优劣结论。

关注点自标签与两套文本编码的一致率为38.89%与44.44%；自然度均分为4.47与4.63。自然表达、语义覆盖和标签可靠性分别报告，完整解释及门槛局限见[论文](paper.md)。四幅图分别展示[候选预检](figures/fig1_pool_gate.png)、[语义标签](figures/fig2_semantic_labels.png)、[事后候选覆盖](figures/fig3_candidate_coverage.png)和[实际调用成本](figures/fig4_pool_cost.png)，另提供同名SVG。

## 材料

| 材料 | 内容 |
| --- | --- |
| [protocol.md](protocol.md)、[protocol.json](protocol.json) | 设计、事前门槛和冻结输入哈希 |
| [cases.json](cases.json)、[profiles.json](profiles.json) | 12个合成情境、3种匿名偏好配置 |
| [common_prompt.txt](common_prompt.txt)、[pool_prompt.txt](pool_prompt.txt)、[selector_prompt.txt](selector_prompt.txt)、[direct_prompt.txt](direct_prompt.txt) | 完整共同及分阶段提示 |
| [selection.py](selection.py)、[benchmark.py](benchmark.py) | 本次实际执行的选择原型与在线运行器 |
| [pools/results.json](pools/results.json)、[pools/raw_calls.json.gz](pools/raw_calls.json.gz) | 未经过语义筛选的全部候选及原始调用 |
| [review_codebook.md](review_codebook.md)、[candidate_review_codebook.md](candidate_review_codebook.md) | 实际文本编码、候选初审规则 |
| [reviews/pool_inputs.json](reviews/pool_inputs.json)、[reviews/pool_groups.json](reviews/pool_groups.json) | 人格、方法与自标签隐藏的候选评价输入 |
| [review_key.json](review_key.json)、[pool_group_key.json](pool_group_key.json) | 编码完成后用于分析的映射 |
| [ratings_pool_1.json](reviews/ratings_pool_1.json)、[ratings_pool_2.json](reviews/ratings_pool_2.json) | 两套独立文本编码，非人类评价 |
| [gate_ratings_1.json](reviews/gate_ratings_1.json)、[gate_ratings_2.json](reviews/gate_ratings_2.json)、[gate.json](gate.json) | 独立候选初审及整批停止计算 |
| [summary.json](summary.json)、[audit_report.json](audit_report.json) | 派生统计与材料核查 |
| [analysis_environment.json](analysis_environment.json)、[artifact_manifest.json](artifact_manifest.json) | 分析环境与最终文件哈希 |
| [replay_report.json](replay_report.json) | 十二次完整请求匹配后的零网络重放 |
| [prepare_reviews.py](prepare_reviews.py) | 匿名材料和整批门槛生成，不调用API |
| [analyze.py](analyze.py)、[replay.py](replay.py) | 离线统计绘图、完整请求匹配后的归档重放 |
| [test_study.py](test_study.py) | 解析、候选选择及原文输出边界的离线测试 |

## 离线复核

在仓库根目录执行。绘图需要matplotlib，运行器与选择器只依赖Python标准库。

```powershell
python -m pip install -r docs/validation/candidate_choice_2026_11_07/requirements.txt
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/analyze.py --out data/candidate-choice-analysis
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/replay.py --out data/candidate-choice-replay
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/test_study.py
```

重放要求完整请求逐字相同才复用归档响应；它没有新增在线样本。在线服务的模型版本、采样和调度可能变化，源代码一致不保证重新生成的回复逐字一致。

## 独立在线重跑

在线命令需要明确追加`--run`，会发送请求并产生费用；不加该参数只准备计划。输出阶段目录必须不存在。认证只通过进程环境`PERSONA_API_KEY`或隐藏输入，不写进配置或公开材料。

```powershell
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/benchmark.py --phase pools --out data/candidate-new/pools --run
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/prepare_reviews.py pools --data data/candidate-new
```

按两个codebook对新回复独立编码，把新标签存入`data/candidate-new/reviews/`，文件名与本研究相同。旧标签不能用于新回复。两位编码者分别完成`ratings_pool_1/2.json`和`gate_ratings_1/2.json`后计算固定门槛：

```powershell
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/prepare_reviews.py gate --data data/candidate-new
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/benchmark.py --phase main --pools data/candidate-new/pools/results.json --gate data/candidate-new/gate.json --out data/candidate-new/main --run
python -B -X utf8 docs/validation/candidate_choice_2026_11_07/prepare_reviews.py direct --data data/candidate-new
```

整批门槛失败时主对照拒绝启动；不得删除差候选、补生成或只保留通过的场景。若继续，对新的直接回复另外完成`ratings_direct_1/2.json`。分析器和归档重放默认复核本目录已发布材料；新研究的标签及材料应独立保存并报告，不能替换本次记录。本次主对照未执行，因此本目录没有`main/`及直接回复评价文件。
