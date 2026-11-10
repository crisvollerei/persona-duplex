# 库接口与冻结记忆

[文档导航](README.md) · [项目首页](../README.md)

以下示例在仓库根目录执行。安装、环境变量和完整 CLI 用法见[使用说明](how-to-start.md)。

另有显式的[可执行人格策略接口PolicySession](executable-policy.md)，在表达前由程序选择行动和关注点。它与本页的默认三段生成Session分开；根据内容是否已结构化及动作是否由程序受限表达，共0—2次模型调用。

该实验的`realization="composed"`选项使用`ExpressionConfig`控制有限措辞，并返回可追溯的`expression_plan`。默认仍为bounded；composed的风格差异不代表多轮自然度已改善，详细范围与结果见对应接口文档。

`InquiryConfig`与`QuestionSlot`可显式启用[信息追问](inquiry-policy.md)。应用审核候选并确认回答语义；`QuestionChoice`支持以外部提议替代选问偏好，提议仍须通过共同权限检查。默认不启用该机制。

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

### 本地记忆检索

前台默认按当前问题检索全部有效版本，最多5条，不额外调用模型。可在库接口设置预算和应用作用域：

```python
from persona.session import SessionConfig

options = SessionConfig(
    memory_limit=5,
    memory_budget_chars=12000,
    memory_scope="work",
)
# 应用已确认的局部偏好；时间窗口可另传valid_from、expires_at。
store.add_judgment("工作阅读器为ReaderPro。", .8, memory_scope="work")
with Session(cfg, store, openai_compatible("deepseek-chat"), options=options) as session:
    print(session.chat("我之前选的阅读器是什么？"))

matches = store.recall_judgments("工作阅读器", scope="work", budget_chars=12000)
```

预算以完整记录的JSON字符数计，不是token上限；超出预算的原话不被截短。`memory_limit=0`或`memory_budget_chars=0`可关闭前台注入。`memory_mode="frozen"`固定完整候选快照，之后仍按问题检索并检查时间窗口。

`Session(..., memory_retriever=rank_ids)`或`Store.recall_judgments(..., retriever=rank_ids)`支持自定义排序器。函数接收`(query, candidates)`，返回候选中唯一的记忆ID列表；不能返回不存在、失效、来源未支撑或其他范围的ID。候选为副本，最终正文由原记录取得；非法返回直接报错，不切换其他模型或重试。自定义排序器自身的服务成本由应用负责。默认实现只有本地词面相关性，不保证语义检索或回答正确。

## 显式回答格式

调用方可以为单轮回复提供 `ReplyContract`。声明会编译为本轮 SAY 的格式要求，完成后对实际交付文本做本地校验。普通聊天不必声明；`contract=None` 保持原来的请求消息和默认一次前台生成。

```python
from persona import ReplyContract

with Session(cfg, store, openai_compatible("deepseek-chat")) as session:
    result = session.say(
        "1200元扣除450元和125元后剩多少？简要说明，最后一行写结论和单位。",
        contract=ReplyContract.conclusion(unit="元", answer_type="number"),
    )
    print(result.say)
    print(result.contract_validation)

    result = session.say(
        "本轮数据：B仓库在苏州，库存3箱。只返回city和count两个JSON字段。",
        contract=ReplyContract.json_object(
            required_keys=("city", "count"),
            value_types={"city": "string", "count": "integer"},
        ),
    )
    print(result.contract_validation)
```

`Session.say`、`chat`、`say_stream` 和 `TurnEngine.turn` 都接受可选的关键字参数 `contract`。`chat` 仍只返回文本；需要校验状态时使用 `say`，或读取 `say_stream` 迭代对象完成后的 `.result`。

| 契约 | 可检查的要求 |
| --- | --- |
| `ReplyContract.conclusion(prefix="结论：", unit="元", answer_type="number")` | 最后非空行的逐字前缀、非空答案、单位及十进制数字形状 |
| `ReplyContract.conclusion(answer_type="time")` | 两位小时和分钟的 `HH:MM` 形状及24小时制范围 |
| `ReplyContract.json_object(required_keys=(...), value_types={...})` | 完整JSON对象、必需键和指定字段类型 |
| `ReplyContract.json_array(item_type="integer")` | 完整JSON数组及每个根数组元素的类型 |
| `ReplyContract.markdown_table(("项目", "金额"), min_rows=2)` | 完整表格、表头顺序、分隔行、列数和最少数据行 |

结论契约默认 `allow_unknown=True`，允许前缀后直接写“未知”而不附单位；是否有依据、应否回答未知仍须另行评价。`answer_type` 可选 `text`、`number`、`integer`、`time`，不会自动补算答案。

JSON类型使用 `string / number / integer / boolean / object / array / null`；`value_types` 中声明的键也必须出现。检查拒绝重复键、非有限数字、代码围栏及前后解释，布尔值不当作整数。字段类型检查是浅层检查，不递归核对对象中数组元素的类型；对象允许额外键。表格只校验结构，不核对格内数值、行名含义或内容是否遗漏。

`result.contract_validation` 独立于 `result.validity`：

```python
{
    "status": "satisfied",  # not_requested / satisfied / violated
    "valid": True,          # 未声明时为 None
    "reasons": [],
    "scope": "mechanical_format_not_semantic_truth",
}
```

格式满足只表示符合声明的机械要求。例如“结论：600元”可满足数字和单位要求，却仍可能算错。校验不会改写已经交付的文字、从内部参考补答或触发重试；违约时保留实际文本和具体原因。协议与交付有效时，也不会仅因格式违约一概撤销本轮独立有效的观察或 TAIL。契约声明和校验结果分别记录到轮次账本，原始用户文本仍用于历史、记忆和语义观察。

## 交付与数据边界

提供 `on_delta` 时，回调拼接须与最终 `say` 相等；没有配置流式适配器时，也会在同一次生成完成后回调交付。若交付与最终文本出现差异，本轮记录失败与实际交付，不提交正常状态变更或自动重试。

OpenAI兼容流适配器按SSE事件聚合多行`data:`，支持跨字节分块的UTF-8与CRLF；即使事件没有文本或`choices`为空，也保留其中的模型、usage和结束元数据。提供方未返回的用量保持缺失，明确返回的零值不会被当作缺失；这些记录不独立证明提供方计费或模型身份。

### 字面正文通道

未转义的内部字段行和协议标记仍保留给解析器。需要在SAY中展示`direct:`字典项、协议示例、代码块或`（内心：……）`等小说文字时，生成端可使用一个JSON字符串块：

```text
<<<LITERAL>>>"direct: English adjective meaning immediate"<<<END_LITERAL>>>
```

测试适配器或应用构造输出时可直接使用帮助函数：

```python
from persona import encode_literal_say

wire = encode_literal_say('```yaml\ndirect: true\n```\n（内心：终于回来了。）')
# 将wire放在SAY正文内；tool表达阶段也接受该字面块。
```

提示中包含该写法，默认调用次数不变。解析器只在整个块闭合且JSON类型为字符串时解码；流式期间按完整块交付，块外普通文字仍增量输出。坏JSON或未闭合块不作为普通正文降级交付，明确的模型控制token仍过滤。模型能否正确选择并输出字面块尚需真实任务评价，这不等于任意无标记输出都能安全保留。

最终`result.say`是可显示的字符串；字面来源随返回值在内部保留，历史使用编码形式，轮次账本在必要时同时记录`say`和`say_wire`。自行序列化并恢复时应保留wire再交给解析器，不能仅把解码后的保留字段重新当成模型原始输出。

账本包含用户原文、内部参考和请求内容，应按敏感数据保管。每个用户、人格、实验条件分别使用数据目录，同一目录只由一个活跃会话或 Store 使用。详见[观察、记忆与持久化](memory-and-storage.md)。
