# 人格判断与表达分工：后续对照实验

入口为[研究论文](./paper.md)。本目录保存固定方案、在线数据、原始调用、统计分析、图表与单一模型辅助盲评，研究日期为 2026-11-05。

## 资料

- [主方案](./protocol.md) / [机器可读方案](./protocol.json)：528 个随机排序测试单元，四次重复。
- [执行计划](./execution_plan.json)：全部条件、合成输入、标准答案与分组。
- [主结果](./experiment_results.json)：496 条对话输出及 32 次检查器测试。
- [主原始调用](./raw_calls.json.gz)：716 条请求与响应，含文本增量；gzip 解压后为 JSON 数组。
- [端到端记忆补充方案](./supplement_protocol.md)、[补充结果](./supplement_results.json) 与 [补充调用](./supplement_raw_calls.json.gz)：四次重复、20 条对话输出、28 次实际调用。
- [统计汇总](./summary.json)、[结果表](./results_tables.md)、[配图](./figures/) 与 [评分校正](./scoring_amendment.json)。
- [数据字典](./data_dictionary.md)：执行成功、结构有效、流式一致、严格评分与缺失字段的区别。
- [条件掩蔽事实复核](./blind_fact_ratings.json)、[建议行为标注](./blind_advice_ratings.json) 与 [揭盲汇总](./annotation_summary.json)：均为单一 Codex 子代理的事后探索评价，不是人类实验。
- [人设来源与操作化](./persona_sources.md)：两套实验配置及中文社区参考来源。
- [数据一致性审计](./audit_report.json)：核对数据是否忠实反映结果，检查通过不表示所有试验成功。
- [材料哈希清单](./artifact_manifest.json) 与 [离线重绘核对](./reproduction_check.json)：校验文件完整性及同环境下图表重生成结果。
- [研究源码快照](./implementation_snapshot/README.md)：保留本文评估的框架版本，供历史结果核对与独立重跑。

两个角色由同一接口模型生成。请求名为 deepseek-chat，归档主运行与补充调用均返回 deepseek-flash；这里不是不同模型品牌之间的性能比较。

## 离线复核

在仓库根目录运行，Python 核心框架依旧仅使用标准库，绘图另外依赖 matplotlib：

```powershell
python -m pip install -r docs/validation/followup_2026_11_05/requirements.txt
python -B -X utf8 docs/validation/followup_2026_11_05/test_study.py
python -B -X utf8 docs/validation/followup_2026_11_05/analyze.py --self-test
python -B -X utf8 docs/validation/followup_2026_11_05/analyze.py
python -B -X utf8 docs/validation/followup_2026_11_05/audit_results.py
```

分析程序读取原始数据，在内存中应用有记录的小数点评分校正，并重新生成派生统计与图表，不改写主结果。中文图优先使用 Microsoft YaHei、SimHei 或 Noto Sans CJK SC；字体环境会影响布局。历史审计使用归档源码快照，现行框架可以继续演进；私有逐试验存档存在时会作额外核对，其缺失不影响公开材料的审计。

## 独立重跑在线实验

运行器从进程环境变量读取密钥。先设置 `PERSONA_API_KEY`，然后使用一个尚不存在的数据目录：

```powershell
python -B -X utf8 docs/validation/followup_2026_11_05/reproduce.py --out data/reproduction-001 --implementation study --run
```

默认 `--implementation study` 复制归档的研究实现；`--implementation current` 则复制现行框架，用于相同条件下的新版本比较。两种方式都在新的独立运行树中执行，源码哈希对应实际加载的文件，结果位于所选目录的 `docs/validation/followup_2026_11_05/` 下。原始发布证据保留在本目录。未传 `--run` 时只准备一个新目录；准备后的目录已存在，实际执行需另选名称。

远端模型采样、服务版本和请求调度不能完全固定，重跑不承诺逐字相同。模型辅助标注需单独复做，新运行不会套用旧标注。原始接口运行器为 [run_experiments.py](./run_experiments.py) 和 [supplement_memory.py](./supplement_memory.py)，其源码哈希记录在对应结果中。
