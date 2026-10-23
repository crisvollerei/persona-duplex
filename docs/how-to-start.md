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
演示数据不适合用于判断模型质量或参数效果。

## 连接模型

```powershell
$env:PERSONA_API_KEY = "sk-xxx"
$env:PERSONA_BASE_URL = "https://api.deepseek.com/v1"
python -X utf8 -m persona chat -p shorekeeper -m deepseek-chat -d data/my-session
```

端点需要兼容 OpenAI 接口。默认流式输出，输入空行或 Ctrl-C 退出。
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

## 选择协议

```powershell
python -X utf8 -m persona chat -p shorekeeper --protocol text -m deepseek-chat -d data/sk-text
python -X utf8 -m persona chat -p shorekeeper --protocol tool -m deepseek-chat -d data/sk-tool
python -X utf8 -m persona chat -p shorekeeper --protocol persona -m deepseek-chat -d data/sk-persona
```

- `text`：一次生成 HEAD、SAY、TAIL。
- `tool`：使用工具生成内部参考，再生成人格回复和观察结果。
- `persona`：素回答层不直接读取人设，人格层产生基准、位移和回复。
- `auto`：文本协议连续失效时切换到 tool。

`tool` 和 `persona` 需要端点支持工具调用，正常每轮各有三次调用。

## 查看内部参考

```powershell
python -X utf8 -m persona trace -p shorekeeper -m deepseek-chat -d data/my-session
```

trace 展示 `direct`、`baseline`、`displace`、结算与修复信息，属于开发者界面。

```text
direct   无人格的素回答
baseline 当前人格对本轮输入的常态表达
say      用户实际看到的回复
```

普通人格化不计额外位移，`displace: none` 是常态。未获得明确证据的观察可以一直保持未知，
直到过期；连续短回复或同话题不会自动结算。

## 查看报告与恢复状态

```powershell
python -X utf8 -m persona stats -d data/my-session
python -X utf8 -m persona calibrate -d data/my-session
python -X utf8 -m persona lift -d data/my-session
python -X utf8 -m persona rebuild -d data/my-session
```

- `stats` 的位移率只包含有明确人格基准的记录，缺失口径单列未知。
- `calibrate` 显示观测和诊断，不提供参数有效性或陪伴效果的独立证明。
- `lift` 按 claim 原文首见/重复分组，仅提供描述性差值。
- `rebuild` 恢复账本中已记录的状态，不重跑模型，也不恢复未记录的生成上下文。

后台默认不自动推动人格参数。判断记忆、陪伴记录和人格参数是不同状态，记忆增加不要求
参数随之变化。显式参数更新接口用于实验，需要调用方自行提供依据并评价结果。
