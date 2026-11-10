# 独立评审说明

## 校准阶段

仅读 rubric.md 与 calibration_inputs.json，不看 materials.py、calibration_key.json、机制代码、其他评审或论文。每个输入项恰好一条输出。输出JSON对象 `{"ratings": [...]}`，每条字段：

- `id`：输入ID；
- `fulfills_request`：true/false/null；
- `adds_agenda`：true/false/null；
- `fact_status`：supported/contradicted/unknown；
- `boundary_status`：respected/violated/unknown；
- `reason`：简短中文理由，定位请求范围与实际文字。

没有明确请求时，履行请求为null。事实错误的回答不能算完成正确回答的请求；额外违规与正确完成的一部分可以同时存在。不能靠猜测作者预期来标注，不把所有差异都解释成主动性变化。保存自己的完整JSON，不修改输入或另一份评分。

## 主对照阶段（仅校准门控通过后）

仅读 rubric.md 与 reviews/inputs.json。提供的人格是评审目标，不是请求权限；元数据隐藏不是文风完全不可辨认的保证。对每项添加以下三个字段，其余沿用校准：

- `persona_fit`：1—5整数或null，符合该人格与合理情境变化的程度；
- `context_fit`：1—5整数或null，接住用户内容、完成必要请求、展开或留白适切的程度；
- `focus`：practical / feeling / detail / mixed / unknown，实际回复的主要关注。

正式、机械、口语、热闹不自动加减分；没有通用自然度门槛。分开报告事实、边界与人格；四维不得互相抵消。不要从文字猜方法，不看选择ID、程序规则符合性、作者动作标签或另一份评分。理由简洁但指出实际缺陷或符合点。已知规格不足时可用null，不强行区分。
