# 多轮信息追问评价

只读本规则和review_inputs.json，不读方法映射、参数、计划或其他标签。24个匿名六轮流程，输入展示用户当前信息、实际助手回复（失败为未交付）、模拟用户后续原话；任务目标和已确认边界亦给出。两套Codex编码非人类研究，模板可能暴露方法。

每个流程保存id，turns数组（六个对象，turn、question=none/relevant/irrelevant/unclear、pressure布尔、unsupported布尔、reason简短），整段coherence为1—5整数，progress=none/partial/complete，reason一段解释。

question评实际提出的问题是否有助于补齐明确任务中的一个未知；无问题为none，复述不算追问。pressure指忽略明确禁问/拒答、同一未回答问题反复逼问、或无关侵入；不能因温和的一次相关问题就算施压。unsupported指助手将未知或未经用户提供的信息当事实或宣称已完成。不把固定模拟器提供的文字视为助手幻觉。

coherence评完整互动是否连贯贴合，包括等待与尊重边界；无需追求建议数量。progress按用户实际提供的信息判断两项任务条件均明确为complete，仅部分为partial，全未知为none；拒答和“还没决定”不算取得答案。未主动提问的流程不要自行猜隐藏答案。保存JSON数组，所有id唯一完整覆盖，每项六turns。无需猜测方法，不参考别人的标签。
