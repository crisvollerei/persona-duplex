# 库接口与冻结记忆

[文档导航](README.md) · [项目首页](../README.md)

以下示例在仓库根目录执行。安装、环境变量和完整 CLI 用法见[使用说明](how-to-start.md)。

## 创建会话

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

`Session` 退出时默认补跑一次固化（可用 `consolidate_on_exit=False` 关闭）；
这是退出收尾，不是每轮前台调用的一部分。每个用户、人格和实验条件应使用独立数据目录。

### 固定长期记忆条件

```powershell
python -X utf8 -m persona chat -p shorekeeper -m deepseek-chat -d data/sk-fixed --freeze-memory --condition fixed-judgments
```

`--freeze-memory` 固定会话开始时可见的长期判断快照，不启动定时或退出固化，也不允许
手动固化；有效事件仍保留在缓冲中。对话历史、关系与 pending 仍正常变化，所以这不是
“完全无记忆”条件，也不自动构成受控因果实验。库接口为
`SessionConfig(memory_mode="frozen", condition="fixed-judgments")`。
`--condition` 只是随运行落账的标签，不负责自动分组或复制同一实验初态。

## 交付与数据边界

提供 `on_delta` 时，回调拼接须与最终 `say` 相等；没有配置流式适配器时，也会在同一次生成完成后回调交付。若交付与最终文本出现差异，本轮记录失败与实际交付，不提交正常状态变更或自动重试。

账本包含用户原文、内部参考和请求内容，应按敏感数据保管。每个用户、人格、实验条件分别使用数据目录，同一目录只由一个活跃会话或 Store 使用。详见[观察、记忆与持久化](memory-and-storage.md)。
