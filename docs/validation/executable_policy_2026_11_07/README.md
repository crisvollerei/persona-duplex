# 可执行人格策略：数据与复现

本研究区分程序决策、状态权限和实际语言执行。两个在线部分实际共87次调用、77条完整输出；另有零新增API的32项事后受限重放，其中16条新组合、16条复用messages完全相同的旧响应。

**主要观察：**自由生成即使接收程序选择的respond，在已有建议禁令组仍有8/8条建议；受限组合的相应子组为0/8，但同16条回复的自然度均分有所下降。结果不证明整体人格或通用事实优势。

| 材料 | 入口 |
| --- | --- |
| 正式论文 | [paper.md](paper.md) |
| 首阶段固定方案 | [protocol.md](protocol.md)、[protocol.json](protocol.json) |
| 字段、分母与阶段 | [data_dictionary.md](data_dictionary.md) |
| 首阶段原始结果与调用 | [results.json](results.json)、[raw_calls.json.gz](raw_calls.json.gz) |
| 任务、配置与强传统提示 | [cases.json](cases.json)、[profiles.json](profiles.json)、[baseline_prompt.txt](baseline_prompt.txt) |
| 普通输入补充任务 | [automatic_cases.json](automatic_cases.json) |
| 实际源码与执行器 | [源码哈希](implementation_snapshot/manifest.json)、[runner_snapshot.py](runner_snapshot.py) |
| 匿名输入、解码与两套标签 | [规则](review_codebook.md)、[输入](review_inputs.json)、[解码](review_key.json)、[编码1](ratings_1.json)、[编码2](ratings_2.json) |
| 状态范围与重启工作流 | [固定方案](state_validation/protocol.json)、[任务](state_validation/cases.json)、[结果](state_validation/results.json)、[调用](state_validation/raw_calls.json.gz)、[源码](state_validation/implementation_snapshot/manifest.json) |
| 受限表达事后重放 | [结果](bounded_replay/results.json)、[源码](bounded_replay/implementation_snapshot/manifest.json)、[实际执行器](bounded_replay/runner_snapshot.py) |
| 新16条回复的另行编码 | [输入](bounded_replay/review_inputs.json)、[编码1](bounded_replay/ratings_1.json)、[编码2](bounded_replay/ratings_2.json) |
| 可重算汇总 | [summary.json](summary.json)、[analyze.py](analyze.py) |
| 离线分析环境 | [analysis_environment.json](analysis_environment.json)、[requirements.txt](requirements.txt) |
| 独立核查与归档字节清单 | [audit.py](audit.py)、[audit_report.json](audit_report.json)、[artifact_manifest.json](artifact_manifest.json) |

## 离线分析

以下命令在仓库根目录执行，读取现有材料，不调用模型。绘图需要Matplotlib和系统中文字体。

```powershell
python -m pip install -r docs/validation/executable_policy_2026_11_07/requirements.txt
python -B -X utf8 docs/validation/executable_policy_2026_11_07/analyze.py --out data/policy-analysis
python -B -X utf8 docs/validation/executable_policy_2026_11_07/audit.py
```

可追加`--no-plots`仅计算统计。分析检查冻结哈希、计划覆盖、原始回复、交付、实际共享输入、程序决策重算、状态补充及源码对应。材料一致不等于行为正确，检查数量不作为人格效果样本数。

## 离线重现受限组合

```powershell
python -B -X utf8 docs/validation/executable_policy_2026_11_07/reproduce_bounded.py --out data/policy-bounded-reproduction
```

目标目录须尚不存在。工具复制归档实现及原始数据，隔离运行原重放执行器：16项直接组合；其余16项逐字比较完整messages后使用原始增量。没有网络请求，也不将旧用量重复计为新增消费。新生成的模型回复若messages不同，不允许复用旧文本或旧编码。

## 重新在线采样

两个在线阶段可分别准备归档实现和固定任务的隔离副本，默认不访问网络：

```powershell
python -B -X utf8 docs/validation/executable_policy_2026_11_07/reproduce.py --phase main --out data/policy-online-rerun
python -B -X utf8 docs/validation/executable_policy_2026_11_07/reproduce.py --phase state --out data/policy-state-rerun
```

只有显式追加`--run`才会发出请求，分别最多76次或12次计划请求；需要进程环境中的PERSONA_API_KEY，也可交互输入。目标目录须尚不存在，新采样需要新的编码，不能沿用本研究标签。远端采样与服务版本变化可能造成不同结果；原发布材料不会被覆盖。

## 图表

- [实际成本](figures/fig1_cost.png)：仅64条手写内容帧的受控部分。
- [实际建议](figures/fig2_realized_advice.png)：两套匿名编码的实际建议比例。
- [实际关注](figures/fig3_realized_focus.png)：首要关注符合配置的比例。
- [受限执行](figures/fig4_bounded_execution.png)：同16个非建议动作的原自由回复与离线受限组合。

论文中的自由表达、状态修订和受限表达各有独立实现快照；现行PolicySession默认bounded，不能用现行默认静默覆盖原自由表达研究。核心实现、有限控制语法与调用成本见[接口说明](../../executable-policy.md)。
