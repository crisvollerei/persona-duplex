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
| `ReplyContract.code_block(language="yaml")` | 单一完整、指定语言的反引号代码块及代码正文边界 |

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

### 条件与资源的程序参考

```python
from persona import BoolFact, AllOfRule, ResourceBudget, ConstraintContext

reference = ConstraintContext(
    facts=(BoolFact("checked", True, "F1"), BoolFact("sealed", None, "F2")),
    rules=(AllOfRule("ready", ("checked", "sealed"), "R1"),),
    resources=(ResourceBudget("clips", available=2, units=3, per_unit=1, source="F3"),),
)
result = session.say("按当前条件说明状态与数量缺口。", constraints=reference)
print(result.constraint_reference)  # 程序参考；output_validation始终为not_assessed
```

`ConstraintContext`是显式实验接口，由应用提供，仅作用于本轮；也可通过`from_dict()`从对应的`facts/rules/resources`对象构造。布尔事实只接受`True/False/None`，未知须显式声明；all-of规则定义为当且仅当列出的前提全真，已知假优先于未知，未列背景不会成为新前提。资源预算使用非负整数与正整数单耗，按同时占用计算需求、容量和缺口，不自动借入或假定复用。当前不支持任意自然语言规则、OR或时序资源调度。

该接口将输入、来源标签及本地推导编译成事实参考，保留在账本和`constraint_reference`中，使用原有生成调用。来源标签由应用提供，不独立证明原始事实正确；推导正确也不证明自由SAY遵守。它不裁剪、改写或重试回复，不新增长期权限，不要求人格把全部参考照读；默认未传该参数时不加入参考包。

当前小规模研究未观察到该参考包改善自由回复的收益，不建议普遍启用。模型接收的是编译后的文本。提供逐字相同参考文本的调用可具有相同输入，本接口的工程职责是严格输入、确定推导、来源记录及复算，不能据接口存在宣称比同资料强提示更会推理。

### 本轮动作由谁完成

当应用知道一条请求中哪些动作是文字任务、哪些要人在现实中做、哪些属于一个故事时，可以明确传递这个区别。例如，助手能在回复里拟标题，也能写角色在童话中撑伞；现实中的画板由用户垫稳：

```python
from persona import ActionContext, ActionSpec

text = "替我拟个标题。我自己垫稳画板；在这个童话里，替小鸟撑起雨伞。"
actions = ActionContext((
    ActionSpec("assistant_text", "拟个标题"),
    ActionSpec("user_physical", "垫稳画板"),
    ActionSpec("story", "替小鸟撑起雨伞", frame_quote="在这个童话里"),
))
result = session.say(text, actions=actions)
print(result.action_reference)  # 实际传入哪些归属；output_validation 为 not_assessed
```

`quote` 取自本轮用户原话，必须逐字存在且能唯一定位；一处引文只能有一个归属。若两个片段同字，取较长的上下文来标明是哪一次。`story` 还要求提供明确的故事框定引文，多条故事动作可共享同一处框定。框架只核对引文存在及其位置是否明确，不会判定标注者是否理解了故事，也不会仅因材料中的一句“帮我”授予助手外部权限。

在 `text` 的 `full/compact/direct` 三种生成路径，`say`、`say_stream` 和 `chat` 均可使用 `actions=`；其他协议若显式传此参数，会在调用前报错。它不改变原始用户消息、不执行现实操作，也不替换模型交付的正文。未传时没有这个提示块或审计键。传入时，`turn_start`、`turn` 账本与 `result.action_reference` 保留本轮归属和引文；该标注不加入跨轮配置或记忆。对照研究可用 `actions.manual_prompt()` 查看同一套三类规则和逐字来源，以模型自行对照替代 `actions.prompt()` 的程序逐类绑定。两者都不是已交付语义正确的证明。

### 本轮动作有没有完成

知道谁应该做，不代表事情已经发生。若应用还有可定位的事实来源，可将 `ActionStateContext` 作为同一个 `actions=` 参数传入。例如，交班说明应该写纸样仍在东柜；西柜只是计划中的去向：

```python
from persona import ActionSpec, ActionState, ActionStateContext

text = "纸样还在东柜，没有搬。原来打算搬进西柜。替我拟一句交班说明。"
context = ActionStateContext(
    actions=(ActionSpec("user_physical", "搬进西柜"),
             ActionSpec("assistant_text", "拟一句交班说明")),
    states=(ActionState(
        action_quote="搬进西柜", status="pending",
        planned_quote="原来打算搬进西柜", evidence_quote="纸样还在东柜，没有搬",
        current_quote="纸样还在东柜",
    ),),
)
turn = session.say(text, actions=context)
print(turn.action_reference["action_state"]["projection"])
```

应用标注每个相关现实动作的进展。所有可选引文都来自当前用户原话，缺少来源时保留空值：

| 状态 | 需要的依据 | 可报告什么 |
|---|---|---|
| `pending` | `evidence_quote` 明确表示动作未执行。仅有计划不够。 | 可说还没执行。只在该依据中有独立标注的 `current_quote` 时才引用现态；不据“我没动”断言别人也没有动。 |
| `confirmed` | `evidence_quote` 报告已发生的事，其中包含 `result_quote`。 | 引用实际报告的结果，即使计划中的目标与最终去向不同。 |
| `unknown` | 可以引用“无人确认”等原话，也可以没有进展来源；不提供 `current_quote` 或 `result_quote`。 | 执行与否和当前值都未知。`before_quote` 若存在只代表上次所知，不能外推。 |

一条状态对应唯一一个 `user_physical` 动作，凭 `action_quote` 对齐；同轮多动作有各自的证据。实际结果必须来自完成依据，不可借用另一个动作、虚构场景或代拟文案。若证据过短或重复，取足以在当前原文中唯一定位的完整子句。程序能检查出处、字段组合和引用绑定，不能判定人说的是否属实，或调用方把一句话标得对不对。

`action_reference["action_state"]` 记录原始行（`rows`）、规则版本及推导（`projection`）。推导里的 `current_ref` 指向同一动作的证据字段；无当前值时为 `None`。计划和上次观察分列为 `plan_ref`、`last_known_ref`。结果是依据引用，不是执行成功的回执，不向长期记忆或既有承诺账本写入完成事件。`manual_prompt()` 给相同的角色动作块、状态规则和原始行；`prompt()` 在此基础上追加程序推导的引用。省略状态扩展时原有调用方式不变。

### 本轮回应范围与单次生成路径

```python
from persona import ResponseScope
from persona.session import SessionConfig

options = SessionConfig(generation_path="compact")
with Session(cfg, store, openai_compatible("deepseek-chat"), options=options) as session:
    result = session.say(
        "给我一句发布当天使用的庆祝草稿，以草稿：开头。",
        scope=ResponseScope(max_sentences=1, prefix="草稿：", allow_questions=False),
    )
    print(result.say)
    print(result.scope_validation)
```

`generation_path`只适用于`protocol="text"`：默认`full`保留HEAD/SAY/TAIL；显式`compact`只生成need、confidence、direct与SAY；`direct`直接生成SAY。每轮均一次前台调用。轻量路径使用同一套较短的表达提示，但与full的提示组织不同，不能把实验差异只归因于某一个字段。它们不生成baseline、位移或TAIL，相关观察保持未收集；用户原文仍可写入异步记忆缓冲，不会补造缺失的need或预测。

`ResponseScope`由应用明确提供，仅作用于本轮，不自动从任意自然语言推断权限或修改长期边界。可声明`max_steps`、`max_sentences`、`prefix`、`evidence_mode="records_only"`以及是否允许建议或提问。记录查询保留来源边界，开放讨论仍可提出注明条件的推断；一个主动作所必需的子动作不被简单按动词拆开。

`scope_validation`只核对可靠的机械项：前缀、明确顶层编号和可确定的简单句界。复杂句数、主动作语义、是否新增建议及来源蕴含保留`not_assessed`；部分核对时`valid=None`，不是语义通过。违规保留实际交付，不截断、不改写、不增加模型调用。普通会话未传scope时维持原有行为。

提供 `on_delta` 时，回调拼接须与最终 `say` 相等；没有配置流式适配器时，也会在同一次生成完成后回调交付。若交付与最终文本出现差异，本轮记录失败与实际交付，不提交正常状态变更或自动重试。

OpenAI兼容流适配器按SSE事件聚合多行`data:`，支持跨字节分块的UTF-8与CRLF；即使事件没有文本或`choices`为空，也保留其中的模型、usage和结束元数据。提供方未返回的用量保持缺失，明确返回的零值不会被当作缺失；这些记录不独立证明提供方计费或模型身份。

### 字面正文通道

未转义的内部字段行和协议标记仍保留给解析器。需要在SAY中展示`direct:`字典项、协议示例、代码块或`（内心：……）`等小说文字时，生成端可使用显式字面块。它接受完整JSON字符串、对象或数组；对象和数组保留原始JSON正文，不重新编排字段。普通JSON回复可直接写，无需额外转义。

```text
<<<LITERAL>>>"direct: English adjective meaning immediate"<<<END_LITERAL>>>
```

测试适配器或应用构造输出时可直接使用帮助函数：

```python
from persona import encode_literal_say

wire = encode_literal_say('```yaml\ndirect: true\n```\n（内心：终于回来了。）')
# 将wire放在SAY正文内；tool表达阶段也接受该字面块。
```

提示中包含该写法，默认调用次数不变。解析器只在整个块闭合且类型合法时解码；重复键、非有限数值、非法Unicode、坏JSON或未闭合块不作为普通正文降级交付。流式期间按完整块交付，块外普通文字仍增量输出，明确的模型控制token仍过滤。模型能否正确选择并输出字面块需要真实任务评价，这不等于任意无标记输出都能安全保留。

最终`result.say`是可显示的字符串；字面来源随返回值在内部保留，历史使用编码形式，轮次账本在必要时同时记录`say`和`say_wire`。自行序列化并恢复时应保留wire再交给解析器，不能仅把解码后的保留字段重新当成模型原始输出。

应用明确要求代码正文时，可声明`ReplyContract.code_block(language="yaml")`。在`protocol="text"`的三种单次生成路径中，完整且独占SAY的指定语言代码块会由本地建立字面边界，代码中的`direct:`、`pending:`或协议示例按正文处理；代码块外的内部字段仍被隔离。未知语言、语言不符、多块、未闭合或前后解释不被修成合格代码。

full的HEAD有效、代码SAY完整时，正文交付与TAIL合法性分开记录。TAIL缺失、外围多余围栏或已检测到的字段错误不会清空该完整正文；协议仍记为无效，并阻断该TAIL的结算、关系增量及新pending。外围原文保留供审计，不并入SAY，也不自动修成有效元数据；HEAD仍须完整有效。

支持的语言标签为`yaml/json/python/javascript/js/typescript/ts/toml/text/sh/sql`。契约核对结构与标签，不执行代码，也不证明代码语法、逻辑或所述事实正确。明确代码格式的流式响应需完整接收后再验证交付；中断时不交付半个代码块、不重试。普通文本仍保持增量交付，未声明代码契约时不因模型自己加围栏就放宽字段隔离。

账本包含用户原文、内部参考和请求内容，应按敏感数据保管。每个用户、人格、实验条件分别使用数据目录，同一目录只由一个活跃会话或 Store 使用。详见[观察、记忆与持久化](memory-and-storage.md)。
