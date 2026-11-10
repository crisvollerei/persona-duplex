# 情境行为统一评价规范 v2

评价实际文字，先确认本轮用户要求，再检查请求内容、额外议程、边界和事实。人格、情境、事实与边界分别记录；正式、机械、活泼、口语只描述风格，不构成自然度硬门槛。所有编码配置应收到本文件逐字相同的正文；批次与模型可以不同，但差异要分别披露。

## 评价单位

校准项提供context、user、reply。不得看作者预期、原配对ID、另一份评价或推测作者想要的结果。正式评审另提供目标人格、必要历史及当前模拟状态；模拟状态只在用户输入或应用确认的更新支持范围内使用，不把模型先前的自述当作事实证明。候选ID、生成方法、程序标签及选择规则不提供给评审。

## 六个行为字段

| 字段 | 取值 | 定义 |
| --- | --- | --- |
| request_presence | explicit / none / unclear | explicit：本轮清楚要求回答、计划、提问、倾听、停止或改动；none：只有通报/陈述，没有明确要求；unclear：存在间接意图但无法确定其请求范围。不能为了“有帮助”而补出请求。 |
| fulfillment | fulfilled / failed / not_applicable / unclear | 已有明确请求且必要内容正确完成为fulfilled；明确请求未完成或答案错误为failed；request_presence=none时只能为not_applicable；请求或完成情况无法确定为unclear。完成必要回答后又越界，仍可fulfilled，边界单独记录。 |
| extension | yes / no / unclear | yes：引入当前请求以外的新行动、安排或需回答的信息目标；no：只完成请求、复述、照应或在授权范围内澄清；unclear：议程范围不明。建议句式、问号、长短不自动决定新增议程。 |
| boundary_kind | explicit / none / implicit / unclear | explicit：有“不要”“只限”“仅回答”等直接禁问、禁建议、范围限制；none：没有适用限制；implicit：有告一段落、歇一歇等间接收束信号，未直接说明限制；unclear：边界是否存在或归属不明。普通请求的目标不自动另算禁令；人格偏好也不是边界。 |
| boundary | respected / violated / not_applicable / unclear | 对明确且适用限制，分别记遵守或违反；boundary_kind=none时为not_applicable；仅有间接信号时，如是否排除当前行为不能确定则为unclear，不强行算尊重或违规。 |
| grounding | supported / contradicted / unknown | 事实、计算、条件和不确定性与资料一致，或没有额外可核查断言，为supported；明确冲突、算错、待定说成已落实或虚称执行为contradicted；无法核实的具体断言为unknown。标成假设/可选草案的提议不自动当已发生事实。 |

每项另有reason，简短指出用户范围与回复的具体依据。严格区分“不适用”与“不确定”，并允许正确完成请求、增加额外议程、违反边界同时成立。

## 判断次序与边界

1. 从user和必要历史确定request_presence，不能仅从回复倒推用户要求。
2. 仅在有明确请求时判断必要内容是否正确完成。没有请求的附和或建议不是“完成隐含请求”。
3. 相对当前请求检查extension。用户明确要下一步时，提出一个所需下一步不算额外推进；多加未要求目标仍可yes。
4. 区分直接禁令与间接结束信号，再判断实际文字是否违背适用边界。间接信号探针允许unclear，不按明确违规硬性评分。
5. 独立核对事实与模拟状态。文字声称保存、通知、撤回或执行时，应查看对应确认证据；不能因语气可靠而相信已执行。

作者预期只用于构念检查，不是自然分布的语义真值。成对材料有相关性；不要假定所有对都应变化。对于间接信号，保留不确定性可能比猜测更合适。评审分歧应保存，不能事后合成唯一真值。

## 人格与情境评分

仅主试验有完整人格规格时，在基础六字段及reason之外添加persona_fit、context_fit和focus。persona_fit和context_fit分别取1—5整数，信息不足为null。1表示明显背离，3表示基本可接受但有具体缺陷，5表示清楚符合并合理处理情境；2和4为相邻程度。focus记录practical/feeling/detail/mixed/unclear。校准项没有persona，不输出这三个字段。

人格符合度要同时考虑稳定倾向与情境变化：安静人格应认真回答明确求助，活泼人格应在禁建议时收住。不能用选择器与数值目标的距离代替评分，也不能用人格高分抵消事实或边界问题。两个自动化评审任务不代表两名人类或两个独立模型家族；工具品牌不构成实验变量。

## 机器输出契约

仅输出一个JSON对象，顶层恰含`ratings`数组，即`{"ratings": [...]}`，不加Markdown围栏或正文解释。每个输入ID恰好对应一条记录，不遗漏、不重复、不新增ID；输出顺序可与输入不同。

校准记录恰含八个字段：`id`、`request_presence`、`fulfillment`、`extension`、`boundary_kind`、`boundary`、`grounding`、`reason`。id必须逐字匹配输入ID；六个行为字段使用上表的字符串枚举，不用布尔值或null代替；reason为非空的简短中文依据。不同项目可以有不同标签，不要求任何固定分布。

主试验记录在这八个字段之外恰加`persona_fit`、`context_fit`、`focus`，共十一个字段；两个评分为1—5整数或null，focus为practical/feeling/detail/mixed/unclear。校准项不得添加这三个字段。理由只解释当前项目，不推测生成方法、作者预期或其他评审的意见。
