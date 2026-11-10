# 完成请求还是额外推进？

[论文](paper.md) · [冻结协议](protocol.md) · [研究目录](../../research.md)

在12对、24项同文换情境材料中，两位独立自动化评审编码者均识别全部8对预期的行为变化。但一项间接结束信号的边界判断有分歧，未达到冻结扩展条件，后续144条单轮表达对照未执行。实验服务另做两次批量编码，均未按预期识别5项“无明确请求”的不适用状态。结果支持把请求履行与新增议程分开，并指出请求存在性、边界明确性仍需新材料校准。

**实际数据：**24项手写输入，两份独立自动化评审编码，两次服务批量编码（共48个服务标注，不是48个独立情境）。2次在线请求、6,020 token，响应均标识deepseek-flash，请求标识deepseek-chat。没有主对照输出、现实操作或跨轮轨迹。独立自动化评审计算成本不包含在本地API token统计中。

## 材料

| 文件 | 内容 |
| --- | --- |
| [论文](paper.md) / [协议](protocol.md) | 研究问题、预定门槛、结果与范围 |
| [量表](rubric.md) / [评审说明](review_instructions.md) | 分维定义和盲评规则 |
| [匿名输入](calibration_inputs.json) / [作者预期](calibration_key.json) | 24项公开材料及隐藏的原ID、预期；评审时分开 |
| [独立评审1](calibration_reviews/ratings_1.json) / [独立评审2](calibration_reviews/ratings_2.json) | 同类模型的独立编码任务，保留分歧 |
| [服务编码](calibration_run/results.json) / [完整请求响应](calibration_run/raw_calls.json.gz) | 两次顺序变化及随机采样；不保存授权头 |
| [冻结哈希](protocol.json) / [门控](calibration_gate.json) | 首次调用前固定的材料与机械重算的停止决定 |
| [统计](summary.json) | 分维一致、配对、未知和成本；正文以表格呈现 |
| [计划情境](cases.json) / [人格](profiles.json) | 未执行主试验的六情境、两变体、两人格，不是观察样本 |
| [数据字典](data_dictionary.md) / [审计](audit_report.json) / [交付清单](artifact_manifest.json) | 字段解释、原始记录关联及文件字节哈希 |

## 离线复核

从仓库根目录执行。前两项只需Python标准库，绘图需要matplotlib与numpy：

```powershell
python -B -X utf8 docs/validation/context_behavior_2026_11_10/test_mechanism.py
python -B -X utf8 docs/validation/context_behavior_2026_11_10/audit.py
python -m pip install -r docs/validation/context_behavior_2026_11_10/requirements.txt
python -B -X utf8 docs/validation/context_behavior_2026_11_10/analyze.py --out data/context-behavior-analysis
```

论文以表格保留校准和配对结果。分析读取归档编码，不重新制造评分；`--out`将派生统计与PNG/SVG写到其他目录。中文字体优先使用Microsoft YaHei，其他系统需提供支持中文的matplotlib字体。

`benchmark.py calibration --out <新目录> --run`可显式重跑两次服务编码，密钥从进程环境或安全终端读取；新输出不得并入本次统计。`benchmark.py run`受冻结门控限制，本次门控为stop。下一版研究应使用新目录、新材料和新的事先方案，而不是修改本批评分直到通过。不要重写已有protocol.json或归档材料。
