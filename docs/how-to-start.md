# 使用说明

## 环境与安装

需要 Python >= 3.11，运行时无第三方依赖。以下命令在项目目录中执行，使用 PowerShell。

```powershell
python -m pip install .
python -X utf8 -m persona --help
python -X utf8 -m unittest discover -s tests -t .
```

安装后也可使用 `persona-duplex` 命令。Windows 下建议启用 UTF-8 模式。

## 离线演示

```powershell
python -X utf8 -m persona demo -d data/demo
```

演示使用固定脚本，展示输出解析、观察结算、关系修复、后台记忆和状态重建，不联网。
演示使用默认的单次生成 `text` 协议，不验证多调用实验或一致性检查。
演示数据不适合用于判断模型质量或参数效果。

## 连接模型

```powershell
$env:PERSONA_API_KEY = "sk-xxx"
$env:PERSONA_BASE_URL = "https://api.deepseek.com/v1"
python -X utf8 -m persona chat -p shorekeeper -m deepseek-chat -d data/my-session
```

端点需要兼容 OpenAI 接口。默认 `text`：一次前台调用生成 HEAD（含 direct）、SAY、
TAIL（含 pending）；默认流式输出，不额外审查、失败重试或自动切换协议。
每轮事件只写入本地缓冲，记忆提炼在后台异步进行，不是每轮必等的串行调用。
输入空行或 Ctrl-C 退出。
API key 不应写入人格配置或提交到仓库。

## 人格与数据目录

可选人格：`default / service / guarded / shorekeeper / ache`。
每个用户和人格使用不同目录；目录中的关系、判断和参数状态会在后续会话中继续使用。
需要从配置初值开始时，选择新的目录，不删除已有账本。

```powershell
python -X utf8 -m persona chat -p ache -m deepseek-chat -d data/my-user-ache
```

自定义人格可以参考 `personas/` 中的 TOML 文件，并将配置文件路径传给 `-p`。
常用参数包括 `bias`、`attention`、`legibility_max` 和 `params` 下的行为参数。
`legibility_max=0` 只禁止人格基准之外的额外位移，不会移除人格语感。
1 使用轻微局部变化策略，2 允许更大的表达转移；这两个级别是提示约束，不是程序测得的语义幅度。
身份文本只作表达参考，不作为现实身份或证据；简短、少干涉等偏好不限制 direct
保留必要条件和不确定性。允许有依据且可否认的试探性问法，不机械删除“我猜”句式。
`special` 的自然语言 `when` 只作说明，必须另列非空 `keywords = ["音乐", "弹琴"]`，
只有本轮用户文本匹配关键词时才注入。配置中的未知参数名、越界数值和重叠站位区间会报错。

## 选择协议

```powershell
python -X utf8 -m persona chat -p shorekeeper --protocol text -m deepseek-chat -d data/sk-text
python -X utf8 -m persona chat -p shorekeeper --protocol tool -m deepseek-chat -d data/sk-tool
python -X utf8 -m persona chat -p shorekeeper --protocol persona -m deepseek-chat -d data/sk-persona
```

- `text`（默认）：一次生成 HEAD、SAY、TAIL。按输出结构分工，人设仍可能影响基础判断。
- `persona`：显式三次调用实验。判断层不直接读取人格配置，表达层保留人格。
- `tool`：显式多调用实验。带人设的内部参考工具，再生成人格回复和观察结果。
- `auto`：显式兜底。文本协议连续失效时切换到 tool，不建立判断隔离。

`persona` 正常每轮三次调用。CLI 使用工具传输，工具失败默认报错，不自动换格式。
库接口未配置 `tool_llm` 时直接使用 JSON 文本，这是选定的正常路径，不是失败后的兜底。
基础判断或表达失败时不另行生成回复，也不发送内部素回答。
失败阶段和诊断保存在 `generation_failure` 账本记录中；流式已交付的部分不会重来。
流式中断后只记录实际已交付片段，不据残句生成新的观察；观察失败也不重新生成回复。

以下额外手段仅用于显式选择的 `--protocol persona`，两个开关相互独立、默认关闭：

- `--check-fidelity`：先缓冲人格候选，增加一次无人设事实一致性检查，通过才交付。
  未通过、结果未知或检查失败均不交付；不自动修复或重写候选。
  检查事实冲突与误导性遗漏，不要求把内部读法、建议和理由全部说出来。
- `--json-fallback`：工具调用失败后，允许改用同阶段 JSON 文本，不退回混合人设协议。

库接口需指定 `protocol="persona"`，再以 `SessionConfig(fidelity_check=True, json_fallback=True)`
显式开启。其他协议不接受这两个开关，避免无声忽略。
`--no-judge` 只关闭 LLM 话题判定与模型记忆提炼，不控制上述开关。
`--no-stream` 关闭流式请求，不改变所选协议与正常调用数。
`tool` 正常三次调用，需要工具支持；失败可能回退到 `text`，仅在显式选择该协议时发生。
这里的调用数只计前台生成；异步话题判定及后台固化是独立任务，也可能调用模型。
记忆线程默认每 300 秒尝试固化；会话退出时默认补跑一次，不属于每轮回复必经步骤。
无模型固化只记模型读法的诊断计数，不将重复猜测当作用户倾向；这些记录不会进入
前台用户判断或冻结快照。账本保留其来源，模型提炼也不把 need、direct、SAY 当成用户证据。

默认文本协议须按 HEAD、SAY、TAIL 顺序各输出一次。缺失、重复、乱序或 HEAD 前有正文时，
安全可见的回复仍可保留，但该输出不更新 TAIL 状态、不进入记忆提炼，也不复用错误元数据。
不会因此自动重试或补一次审查。`persona` 的表达阶段只需 HEAD、SAY，观察仍是独立阶段；
表达结构无效时跳过观察调用，保留失效诊断。

## 固定长期判断的实验条件

```powershell
python -X utf8 -m persona chat -p shorekeeper -m deepseek-chat -d data/sk-fixed --freeze-memory --condition fixed-judgments
```

`--freeze-memory` 使用会话开始时的长期判断快照，不启动后台或退出固化，手动固化也会报错。
有效事件仍保存在缓冲中，历史、关系和 pending 仍正常更新；它不是“全部记忆关闭”。
库接口是 `SessionConfig(memory_mode="frozen", condition="fixed-judgments")`。
`--condition` 随配置和请求记录实验标签，不自动创建相同初态或完成因果对照。
每个条件使用独立目录；在不同初态或不同模型设置下取得的统计不能直接归因于某个旋钮。

配置了记忆模型时，请求、格式或叙事生成失败会保留事件等待后续处理，不自动改用
need 计数。无模型时使用有来源标记的启发式路径；`--no-judge` 不等于关闭所有记忆。
需要显式退回计数时，可使用库选项 `Consolidator(heuristic_fallback=True)`。

## 查看内部参考

```powershell
python -X utf8 -m persona trace -p shorekeeper -m deepseek-chat -d data/my-session
```

trace 展示 `direct`、`baseline`、`displace`、结算与修复信息，属于开发者界面。

```text
direct   目标为中性素回答；默认同次生成不保证完全不受人格影响
baseline 当前人格对本轮输入的常态表达
say      用户实际看到的回复
```

普通人格化不计额外位移，`displace: none` 是常态。未获得明确证据的观察可以一直保持未知，
直到过期；连续短回复或同话题不会自动结算。

设计上，人格影响注意、站位、亲疏和表达选择，不只影响措辞。
可以把劝阻降成问题、转移关心的表达或保留一句没说完的话，不要求完整复述内部建议。
这与改变事实判断不同：现实断言不能造假，计算不能改错，未知不能当作保证；
实际建议或操作也不能因省略必要条件而变得错误或误导。
默认同次生成的分工不是模型上下文硬隔离，也不通过额外审查替主路径纠错。
`persona` 实验另传必要约束；只有显式开启一致性检查才缓冲候选。
检查仍由模型完成，不能当作正确性或人格效果的独立证明。

## 查看报告与恢复状态

```powershell
python -X utf8 -m persona stats -d data/my-session
python -X utf8 -m persona calibrate -d data/my-session
python -X utf8 -m persona lift -d data/my-session
python -X utf8 -m persona rebuild -d data/my-session
```

- `stats` 的位移率只包含本地协议有效且有明确人格基准的记录；无效、中断与未知口径单列。
- `calibrate` 显示观测和诊断，不提供参数有效性或陪伴效果的独立证明。
- `lift` 按 claim 原文首见/重复分组，仅提供描述性差值。
- `rebuild` 恢复实际提交的状态、限额结果与已记录的有限历史窗口，不重跑模型或填造旧数据。

启动会补完已写入账本而尚未发布的本地提交；后台会恢复暂存的未处理事件，并跳过已经
提交消费的事件。损坏的缓冲保留为 `.invalid`，可对照 `buffer_error` 排查。
新判断与对应陪伴记录原子提交；模型失败时保留待处理事件。
`ledger.jsonl` 可含 `commit` 封套，程序读取请使用 `Store.read_ledger()` 取得展开事件；
`judgments.jsonl` 只用于旧版导入，新判断以账本为准。

`run_start` 保存运行配置与哈希、协议和已知模型设置；每轮关联实际请求、输出及可获得的
响应元数据。自定义模型条件或提供方内部状态仍可能未知；本地有效性不代表语义或事实真值。
这些记录包含原文和内部参考，应按敏感数据保管，不要直接公开。

后台默认不自动推动人格参数。判断记忆、陪伴记录和人格参数是不同状态，记忆增加不要求
参数随之变化。显式参数更新接口用于实验，需要调用方自行提供依据并评价结果。
默认保持参数基线；可靠的自动成长信号与更新策略尚未接入。
