# 候选覆盖与行为标签修订验证

[研究总览](../../research.md) · [论文](paper.md) · [冻结方案](protocol.md) · [数据字典](data_dictionary.md)

本研究在8个新情境上，把生成规范和标签来源拆成两因素：原六候选提示与程序编译的语义覆盖计划；生成时自标签与另一次看不到原标签/计划的证据标注。两个标签来源共享同池文本。程序只验证引用对应和格式，不把它们当成语义真值；未知不回退、不补生成。

实际完成32次请求、16个池、96条候选、96次离线选择，共35521 token。论文与统计分别评价候选覆盖、质量、标签可靠性、实际选择和成本。默认Session、异步记忆和其他已发布实验均未修改。

覆盖计划把开放情境可用目标覆盖从3/18、4/18提高到12/18、10/18，但两套编码的自然度分别下降0.46、0.31分；可用比例为81.25%、75.00%，未过事前门槛。分离标注使coverage候选的关注点精确匹配均增加7/48条，主动性却均减少3/48条；额外标注路径约需原self路径2.04倍token。因此两组件均保留为实验原型，不扩大当前完整机制，不将局部指标改善写成整体人格优势。

四幅图分别呈现[覆盖与质量](figures/fig1_coverage_quality.png)、[标签取得与匹配](figures/fig2_labels.png)、[实际离线选择](figures/fig3_offline_selection.png)和[完整成本](figures/fig4_cost.png)，另有同名SVG。结果细节、部分选择质量收益及局限见[论文](paper.md)。

## 研究材料

| 材料 | 内容 |
| --- | --- |
| [protocol.md](protocol.md)、[protocol.json](protocol.json) | 事前冻结的两因素方案及扩展门槛 |
| [cases.json](cases.json)、[profiles.json](profiles.json) | 新情境、明确边界与匿名偏好目标 |
| [mechanism.py](mechanism.py)、[selection.py](selection.py)、[benchmark.py](benchmark.py) | 本次实际运行的改进原型与运行器 |
| [run/results.json](run/results.json)、[run/raw_calls.json.gz](run/raw_calls.json.gz) | 全部候选、两类标签、离线选择与原始请求响应 |
| [review_codebook.md](review_codebook.md)、[reviews/inputs.json](reviews/inputs.json) | 方法与标签隐藏的实际文本评价输入 |
| [ratings_1.json](reviews/ratings_1.json)、[ratings_2.json](reviews/ratings_2.json)、[review_key.json](review_key.json) | 两套独立Codex编码及分析映射 |
| [prepare_reviews.py](prepare_reviews.py)、[analyze.py](analyze.py) | 匿名材料准备及离线统计绘图 |
| [replay.py](replay.py)、[replay_report.json](replay_report.json) | 完整请求匹配的零网络重放 |
| [test_mechanism.py](test_mechanism.py) | 边界计划、匿名化、证据验证、未知、失败与原文选择测试 |
| [summary.json](summary.json)、[audit_report.json](audit_report.json) | 完整派生指标、两项门槛判断及材料核查 |
| [analysis_environment.json](analysis_environment.json)、[artifact_manifest.json](artifact_manifest.json) | 分析环境与最终文件SHA-256清单 |

## 离线复核

在仓库根目录运行，不需要API密钥：

```powershell
python -m pip install -r docs/validation/candidate_revision_2026_11_07/requirements.txt
python -B -X utf8 docs/validation/candidate_revision_2026_11_07/analyze.py --out data/candidate-revision-analysis
python -B -X utf8 docs/validation/candidate_revision_2026_11_07/replay.py --out data/candidate-revision-replay-new
python -B -X utf8 docs/validation/candidate_revision_2026_11_07/test_mechanism.py
```

归档重放只在完整请求逐字一致时复用原响应，核对16池、32次请求与96次离线选择；不构成新增在线重复。图表从已保存数据重新计算，不重新评分。

## 重新在线采样

以下命令需要明确追加`--run`，最多发送32次请求并产生费用；不加该参数只准备执行计划。输出目录必须不存在。密钥只经`PERSONA_API_KEY`进程环境或隐藏输入提供，不写入配置和研究数据。

```powershell
python -B -X utf8 docs/validation/candidate_revision_2026_11_07/benchmark.py --out data/candidate-revision-new/run --run
python -B -X utf8 docs/validation/candidate_revision_2026_11_07/prepare_reviews.py --data data/candidate-revision-new
```

新文本必须按codebook组织新的独立匿名评价，保存到新数据根目录的`reviews/ratings_1.json`和`ratings_2.json`，不能沿用本次编码。分析器与重放脚本默认读取本研究归档目录；新研究材料需独立保存并报告。服务版本、采样、缓存和调度可能变化，源码相同不保证回复逐字相同。
