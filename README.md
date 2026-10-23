# persona-duplex

轻量对话人格框架。Python >= 3.11，纯标准库，支持 OpenAI 兼容端点。
通过人格配置、分阶段生成和追加式账本管理对话表达与会话状态。

## 快速开始

在项目目录中运行，以下示例使用 PowerShell：

```powershell
# 离线演示，不需要 API key
python -X utf8 -m persona demo -d data/demo

# 真实对话，默认流式输出
$env:PERSONA_API_KEY = "sk-xxx"
$env:PERSONA_BASE_URL = "https://api.deepseek.com/v1"
python -X utf8 -m persona chat -p shorekeeper -m deepseek-chat -d data/shorekeeper

# 开发者视图：显示内部参考、观察结果和修复状态
python -X utf8 -m persona trace -p shorekeeper -m deepseek-chat -d data/shorekeeper

# 运行离线测试
python -X utf8 -m unittest discover -s tests -t .
```

也可通过 `python -m pip install .` 安装，使用 `persona-duplex` 命令。
UTF-8 模式用于避免 Windows 终端的符号编码问题。演示使用手写脚本，不代表真实模型效果。

## 工作原理

```text
用户输入 + 对话上下文
  -> direct：不带人格的素回答
  -> baseline：当前人格在这个情境下的常态表达
  -> say：实际输出，可选择在 baseline 之外作额外位移
```

`direct -> baseline` 是普通人格化；`baseline -> say` 才是位移。
相对人格基准无额外位移是常态，`displace: none` 仍保留人格语感。
HEAD 的 `baseline` 是本轮表达参考，不是永久固定的台词；配置中的 `traits.anchor`
则是人格参数的基线。基准文本与位移类型由模型给出，文本相似度检查仅用于诊断。

用户只看到 SAY。HEAD、TAIL 和生成脚手架由解析与流式过滤层隔离；`trace` 是例外的
开发者观察界面，不应直接提供给最终用户。生成失败后，已输出的内容不会重新生成。

## 人格与生成协议

内置 `default`、`service`、`guarded`、`shorekeeper`、`ache` 五套 TOML 配置。
人格由身份文本、语感例句、注意力、站位和行为参数共同描述：

- `bias`：把握度偏移，影响站位选择。
- `attention`：只读表层或同时考虑潜台词。
- `legibility_max`：相对人格基准的位移上限；0 保留人格，但不作额外位移。
- `traits`：行为参数，包含 `anchor / value / plasticity / drift / lo / hi`。
  提示词包含当前数值和两端语义。

| 协议 | 正常调用数 | 分工 |
|---|---:|---|
| `text`（默认） | 1 | 一次生成 HEAD、SAY、TAIL |
| `tool` | 3 | 内部参考工具、带人格和上下文的 SAY、观察工具 |
| `persona` | 3 | 素回答工具、人格层 HEAD + SAY、观察工具 |
| `auto` | 按路径 | 文本协议连续失效时切换到 tool |

`persona` 的素回答层不直接注入人设，只接收用户历史和可解析的历史 `direct`，
不转发人格化的历史 SAY 或内部协议字段。用户原话与 need 分类仍可能间接影响结果，
因此该边界不保证素回答完全中性。

```powershell
python -X utf8 -m persona chat -p shorekeeper --protocol persona -m deepseek-chat -d data/sk-persona
```

文本协议结构：

```text
<<<HEAD>>>
need: 暂定的需求读法
confidence: 对读法的把握度
direct: 中性的素回答
baseline: 当前人格通常会怎样回应
displace: 相对 baseline 的额外位移；none 是常态
<<<SAY>>>
实际说出口的话
<<<TAIL>>>
settle: 有证据才结算，否则 - 或 watch
pending: 可选的后续观察假设
delta: 受限的关系增量
```

## 观察与记忆

需求读法是生成回复的参考，不要求每轮记录预测或立即结算。

| 情况 | 处理 |
|---|---|
| 明确提供所预测的内容 | 可记录 `CONFIRMED`，附证据 |
| 明确反驳所预测的内容 | 可记录 `REFUTED`，附证据 |
| 仅同话题、换题、连续短回复或时间经过 | 保持 `WATCH` |
| 超过保留期 | `EXPIRED`，没有对错结论 |
| 语域突变 | 独立记录暂定的疑似受伤信号，不覆盖预测真假 |

可选语义判定器异步运行；失败或超时时，字面兜底只观察相关性，不据此判对错。
模型 TAIL 的 confirmed/refuted 必须提供非空证据。队列总量默认最多 8 条，保留期默认
72 小时。关系增量使用白名单和单轮限幅。

后台提炼关于用户的可推翻判断，并记录角色的陪伴经历。自我叙事中的“我”指角色，
不将用户经历当成角色自身的经历。默认后台不自动调整 `bias`、`traits.value` 或
`traits.anchor`；参数更新、`push_traits()` 和 `push_anchors()` 供显式实验调用。

## 作为库使用

```python
from persona.config import PersonaConfig
from persona.store import Store
from persona.session import Session
from persona.llm import openai_compatible, openai_stream

cfg = PersonaConfig.load("personas/shorekeeper.toml")
store = Store("data/my-user-shorekeeper")

with Session(cfg, store, openai_compatible("deepseek-chat"),
             stream_llm=openai_stream("deepseek-chat")) as session:
    result = session.say("今天好累啊")
    print(result.say)
```

`Session` 退出时补跑后台固化。每个用户、人格和实验条件应使用独立数据目录。

## 持久化与报告

```text
data/<会话目录>/
  ledger.jsonl      追加式观测与已发生变更
  state/            物化视图
  judgments.jsonl   判断记忆
  self-story.md     角色的陪伴记录
  buffer/           尚未固化的事件
```

turn 记录包含人格基准、位移口径、人格名和有效参数快照。叙事记录和基线推动频率也有
账本依据。历史数据缺失的字段保持未知；没有入账的独立叙事文件在重建时保留。

| 命令 | 用途 |
|---|---|
| `stats` | 位移、预测校准、声明检测、修复和参数状态统计 |
| `calibrate` | 观测分布、判定覆盖和耗时诊断 |
| `lift` | 按 claim 原文首见/重复分组的命中率差值 |
| `rebuild` | 按账本恢复已记录的状态 |

位移率只计有明确人格基准及对应口径的记录，缺失部分单列未知。`lift` 是描述性统计，
不是有/无记忆的因果实验。`rebuild` 不按新参数重跑模型，也不重新判定旧事件。
模型判定和修复状态不是独立真值，命中率不等于用户认可度。

完整命令示例见 [使用说明](docs/how-to-start.md)。

## 项目结构

- `persona/`：配置、提示词、协议、解析、会话、记忆、账本与报告。
- `personas/`：随安装包分发的人格配置。
- `tests/`：离线单元测试和回归测试。
- `docs/`：使用说明。

## 限制

- 单个 store 没有完整的多用户/多人格隔离。
- 重启不完整恢复生成历史、话题窗口等进程内上下文。
- 参数快照不包含全部模型与采样条件，不能单独保证实验可复现。
- 人格基准、语义判定、疑似受伤检测和叙事均可能有误差；行为效果需要独立评价。
- 不提供自动人格成长或完整的参数反事实实验管线。

## 许可

MIT
