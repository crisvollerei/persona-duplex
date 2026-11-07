# 有限表达组合与多轮体验

[研究总览](../../research.md) · [接口说明](../../executable-policy.md)

本研究比较固定倾听句式与按参数组合的有限表达。18个六轮条件产生108条主输出，另108条只用于不中断恢复核验；共12份不同对话，零API调用。结果显示语气可区分、机械重复减少，但多轮自然度没有改善，语义重复感没有降低。新方式保留为显式composed实验，不替换默认实现。

| 材料 | 内容 |
| --- | --- |
| [paper.md](paper.md) | 研究方法、分母、结果、负结果解释和局限 |
| [protocol.md](protocol.md) / [protocol.json](protocol.json) | 输出前固定的方案和代码/输入哈希 |
| [cases.json](cases.json) / [profiles.json](profiles.json) | 三场景与三组表达参数 |
| [results.json](results.json) | 主输出、动作、关注、表达计划、状态、恢复比较 |
| [runtime_ledgers.json.gz](runtime_ledgers.json.gz) | 18个主条件的原始运行账本 |
| [review_inputs.json](review_inputs.json) / [review_codebook.md](review_codebook.md) | 方法映射隐藏的多轮编码材料和规则 |
| [ratings_1.json](ratings_1.json) / [ratings_2.json](ratings_2.json) | 两套Codex标签，均非人类评价 |
| [review_key.json](review_key.json) | 编码结束后用于分析的条件映射 |
| [summary.json](summary.json) / [audit_report.json](audit_report.json) | 自动派生统计和材料对应检查 |
| [figures](figures) | 三幅论文图表 |
| [benchmark.py](benchmark.py) / [analyze.py](analyze.py) | 冻结实现重跑与离线统计绘图 |
| [data_dictionary.md](data_dictionary.md) | 字段和分母说明 |
| [artifact_manifest.json](artifact_manifest.json) | 发布材料文件哈希 |

在仓库根目录重算，输出到新目录，不覆盖已发布数据：

```powershell
python -B -X utf8 docs/validation/composed_expression_2026_11_07/benchmark.py --out data/composed-reproduction
python -m pip install -r docs/validation/composed_expression_2026_11_07/requirements.txt
python -B -X utf8 docs/validation/composed_expression_2026_11_07/analyze.py --out data/composed-analysis
```

benchmark使用本目录归档的实现，任何模型调用都会使实验失败。它同时产生108条主输出和108条恢复核验输出；后者只比较确定性结果，不扩充体验样本。相同代码和输入下results/review_inputs/review_key可逐字比较；运行账本有新ID和实际写入时间，不要求压缩文件完全相同。

analyze核对冻结输入，读取已保存输出及两套标签，重算统计与图表，不重新评分。主分析仅需标准库，绘图需要matplotlib；图采用其DejaVu Sans字体，图中文字为英文，解释以中文论文为准。若改代码、输入或回复，不可沿用这批标签作新结果。
