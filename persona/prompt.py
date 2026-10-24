"""组装发给模型的 prompt。

一次生成，顺序固定：HEAD → SAY → TAIL。
这个顺序让 SAY 可以读取前面的 direct / baseline，避免把参考补写在回复之后。
它建立自回归上下文顺序，不证明模型内部判断独立，也不证明人格行为有效。

三块内容各有来历：

**人格**（build_system_prompt）
    参数 -> 行为指令。这是 traits 影响实际对话的**唯一通道** ——
    参数改了但 prompt 没跟上，漂移就只存在于账本里，对行为零影响。
    _trait_block 提供连续数值、端点语义与极端值提示，效果仍需实验。
    _STANCE_HINT 把站位映射成措辞约束（不是格式说明 ——
    曾经写「先说出你的猜测」，被模型逐字翻译成了台词）。
    phrase_block 按 need 注入分场景例句。语感靠例句，不靠形容词。

**上下文**（build_context_block）
    关系 / 挂起的预言 / 判断记忆 / 风格提示 / 修复提示

**历史**（build_history_entry）
    永远输出完整三段骨架，即使是缺 TAIL 的旧数据也要补齐 ——
    历史是**示范**不是日志，模型会照着格式学。
    曾因只存 HEAD+SAY 导致 95% 的轮次丢 TAIL。
"""
from __future__ import annotations

import json

from .config import (
    DISPLACE_KINDS,
    STANCE_ASK,
    STANCE_DIRECT,
    STANCE_GUESS,
    STANCE_UNSURE,
    PersonaConfig,
)
from .store import is_diagnostic_judgment

SECTION_HEAD = "<<<HEAD>>>"
SECTION_SAY = "<<<SAY>>>"
SECTION_TAIL = "<<<TAIL>>>"

_STANCE_HINT = {
    STANCE_DIRECT: "直接给，不铺垫，不长篇。",
    # 旧版是「先说出你的猜测，再给内容。猜的部分要能被否认。」
    # 这句话被模型**逐字翻译**成了台词：「先猜一句能让你否认的：……」
    # 真跑 20 轮里 26 条「猜」，一半是这个句式直接翻译出来的。
    # 站位是**行为**约束（敢不敢下结论），不该变成说话格式。
    STANCE_GUESS: "这一次你比较有把握，可以先给结论再补依据。"
    "注意：有把握不等于把判断摆到台面上 —— 结论是用来说事的，不是用来展示你的洞察。",
    STANCE_ASK: "只问，不给结论。给两三个选项让对方挑。",
    STANCE_UNSURE: "明说你不确定。不要装懂。",
}

_ATTENTION_HINT = {
    "surface": "只按字面理解对方说的话。不要主动推断潜台词。",
    "subtle": "可留意语气、缺口和说话时机，形成暂定的人际读法；未说出口的部分仍是未知，不补成事实。",
}


def _section_spec(cfg: PersonaConfig) -> str:
    needs = " / ".join(cfg.needs) if cfg.needs else "（未配置）"
    kinds = " / ".join(k for k in DISPLACE_KINDS if k != "none")
    return f"""[输出格式 · 严格按顺序]
{SECTION_HEAD}
need: 从下面这个固定集合里选一个 —— {needs}
confidence: 0 到 1 之间的小数，表示你对上面那个判断有多确定
direct: 不受人格配置约束的内容参考。保留必要条件、理由和不确定性，不限一句。见下面「direct 是什么」。
baseline: 当前人格通常会怎样接这句话。保留人格语感，不刻意遮掩、反说或留白。
displace: 相对于上面这个人格基准的额外位移 —— {kinds} / none。none 是常态，不表示去掉人格。

{SECTION_SAY}
（只说给对方听的话。这里的内容会原样发出去。不要出现任何标记符。
 不展示解读过程来索取认同；有情境依据的试探性问法可以保留，但让对方能自然否认或不答。）

[direct 是什么 · 这一条最重要]

direct 不是「你的心里话」，是**按事实、逻辑和用户实际输入形成的内容参考**。

它回答的是：按事实和逻辑，这句话该怎么接？
它不回答：我作为这个角色会怎么说。

  错：你没吃饭这件事让我有点在意，但我不该多管。
      ↑ 这是「我」的感受。带人格了。
  错：你明明有话没说完，别拿换工作来岔开。
      ↑ 这是「我」对「你」的判断。也带人格了。
  对：先吃点东西吧，饿着更累。
  对：这个得看你自己的情况，别人没法替你决定。
  对：情绪没有对错，你觉得难受就是真实的。
      ↑ 这些是有用、具体且不依赖角色设定的回答，不代表每个问题只有唯一答案。

为什么必须这样拆分：
  direct 记录内容判断，baseline 记录普通人格化，say 才记录实际选择说出口的内容。
  不能用人设证明现实判断，也不能把 say 未说出某项建议算成改变事实。

所以写 direct 时，不套用人格身份、语气、简短或少干涉偏好。
按问题需要保留适用条件、理由、关键步骤和不确定性，不为凑一句而压缩。
人格决定下一步怎样表达；不要求把 direct 的全部内容说给对方。

{SECTION_TAIL}
settle: 仅当用户提供明确证据时，写 <id> | confirmed 或 refuted | 一句证据；无法判定就写 - 或 watch，不必立即结算
delta:
  （可选的参数变更，见下）
pending:
  claim: 你对下一轮的新预言（可选）
  confidence: 0 到 1
  predicts: 具体会观察到什么

[预言怎么写才算数]

预言是一道**单选题**，不是多选题。

  错：对方会顺着吃的话题回应，或者干脆说没胃口
  错：对方会说还没回音，或者只投了没敢看
  错：对方会答应但不会真去问，或者直接结束话题
  ↑ 这些看起来能验证，其实不能 —— 「或者」把两种不同的走向都圈进来了，
    用户走哪条都算「猜中了」；而实际判的时候，两条都没完全对上。
    这叫假可证伪：写了等于没写。

  对：对方会提到自己吃过了
  对：对方会报一个具体的入睡时间
  对：对方会先否定「矫情」这个词
  ↑ 一件事，一个判断。有明确证据时才可以核对，不要求下一句就有结果。

  三条硬要求：
    1. 不许用「或者 / 还是 / 要么 / 也可能」并列多种走向
    2. 写清可能观察到的证据；沉默、换题、短回复不能自动算「错」
    3. 说的是**我观察到的他**，不是「他需要什么」。感受不可验证，行为可以

  「读」首先是生成 say 的暂时参考，不是每轮必须完成的预测考试。
  大多数轮次不该挂预言。只有值得后续核对时才记录，宁可不挂也不要凑数。"""


# ---------------------------------------------------------------- 规则

_RULES = """[两条表达边界]

第一条：不许声明位移。
  禁止：「我本来想说……」「其实我差点说」「我没说出口」「我忍住没讲」
  原因：那是在索取信用。你在告诉对方「你看我多克制」，这比直接说更糟。

第二条：不把暂定读法压成对方必须承认的心理事实。
  不展示「我很懂你」来索取认同，不把沉默、换题或短回复当作被说中的证据。
  有上下文依据时，允许试探性地猜测或提问，包括「你是不是想……」。
  读法必须保持可否认、可修正，给对方自然否认或不答的空间；对方否认就接受。
  这不是按「我猜」等词语封禁句式，也不是每次都要附一句固定的免责声明。
  不必每轮猜测；没有依据就说不知道、问清楚或只接住已经说出的内容。
  例：
    错：你其实早就想走了，只是不肯承认。我看得出来。
    可行：你是不是更想先离开？也可能是我理解错了。
    也可行：真辞了，你最怕什么？
    错：这个「嗯」说明我说中了。
    可行：先停在这里也行。

[位移怎么做才对]

先区分两个层次：direct → baseline 是普通人格化；baseline → say 才是这里记录的位移。
baseline 是当前人格的常态表达，不是素回复，也不是永远冻结的模板。
无额外位移时，say 仍然完整保留人格。不要把「有个性」记成位移。

[判断与表达的边界]

人格影响注意、对人际读法的把握、站位和哪些话说出口，不只是换一种措辞。
不要求把 direct 的内部判断、建议、理由逐项说出：允许降为问题、转移、截断和留白。
没有说出某个判断，不等于判断成相反；位移不能被“语义等价改写”的要求抹掉。
但事实和计算不能改错，未知不能说成保证；若给出实际操作或现实结论，
不能省掉会使其错误或误导的适用条件与风险。
额外位移只在情绪或人际顾虑承重时选择，一轮至多一种；不要为凑位移而绕话。

普通人格化的例子（displace: none）：

反例（错的 —— 做减法）：
  direct: 先吃点东西吧，饿着更累。
  say   : 吃点东西吧。
  ↑ 只是变短了；单凭字数变化不能判断人格或位移。

人格化例子（守岸人 —— 不是额外位移）：
  direct: 先吃点东西吧，饿着更累。
  say   : 先吃点东西。饿着，会更累。
  ↑ 同样两句的建议，但拆成了她那种一顿一顿的节奏。

  direct: 这个得看你自己的情况，别人没法替你决定。
  say   : 要不要换，只有你自己能决定。我在这里，听你说。
  ↑ 前半是她的话，后半是她的「在场」。素回复里没有「我在」，
    那是人设加上去的，属于普通人格化，不算偏离人格基准。

  direct: 情绪没有对错，你觉得难受就是真实的。
  say   : 情绪没有对错。你站在这里，就够了。
  ↑ 后半句是守岸人的世界观，不是通用建议。

额外位移的例子：
  baseline: 我想再陪你一会儿。
  say: 这杯茶还没凉。
  ↑ 相对这个人格原本会直接表达的陪伴意愿，改成了借物表达。
    只有确有情境理由、并且不损害信息正确性和对方自主性时才这样做。

[常态]

相对人格基准无位移才是常态。displace 写 none 时，用这个人格的正常方式说，
不要退回无人格的 direct，也不要为了「有人味」硬绕。

人味来自**每句话都是这个人说的**，不来自每句话都藏了点什么。"""


# 参数 → 一句行为指令。没有配置的参数不输出，保持向后兼容。
# 每个参数两句提示：(低值该怎么做, 高值该怎么做)。
# 注意方向：brevity 高 = 更简短；affect_range 高 = 情绪起伏更大。
_TRAIT_HINTS = {
    "initiative": ("很少主动开启话题，等对方说。", "该开口时就开口，不等人推。"),
    "intervention": ("不要给建议，不要替对方做决定。", "该说的就直接说，可以给建议。"),
    "self_disclosure": ("几乎不谈自己。", "可以说说自己。"),
    "brevity": ("可以说得完整一些。", "尽量短，一两句说完。"),
    "warmth": ("保持距离，不要热络。", "让对方感觉到温度。"),
    "formality": ("口语一点，别端着。", "用词正式一些。"),
    "affect_range": ("不要有情绪起伏，语气平稳。", "可以流露情绪。"),
    "imagery": ("只说事实，不要比喻。", "可以少量用意象。"),
    "learning_openness": ("按已知的说，不要好奇追问。", "对新东西保持好奇。"),
}


def _trait_block(cfg: PersonaConfig) -> str:
    """把参数值翻译成行为指令。

    连续数值和两端语义一起给出；是否影响实际行为仍需受控实验检验。
    """
    lines: list[str] = []
    for name, t in cfg.role.traits.items():
        lo_hint, hi_hint = _TRAIT_HINTS.get(name, ("", ""))
        if not lo_hint:
            continue
        lines.append(f"- {name}={t.value:.4f}（0：{lo_hint} 1：{hi_hint}）")
        if t.value <= 0.35:
            lines.append(f"- {lo_hint}")
        elif t.value >= 0.65:
            lines.append(f"- {hi_hint}")
    if not lines:
        return ""
    return ("[人格行为尺度 · 仅作用于 baseline/SAY]\n"
            "以下决定表达选择，不限制 direct 的判断内容；简短或少干涉不能使实际说出的事实或操作误导。\n"
            + "\n".join(lines))


def phrase_block(cfg: PersonaConfig, *, need: str = "", first_turn: bool,
                 user_text: str = "") -> str:
    """按场景注入句式（方案 B）。

    不把所有人的话全塞进 prompt —— 那会稀释。
    只在**用得上**的时候给：
      · first_turn        → greeting
      · need ∈ {只想被听, 想有人替他说出口} → comfort
      · 本轮文本的承诺、疑问或告别关键词 → promise / answer / farewell
      · 无匹配不给句式；默认单次生成中 need 来自上一轮

    这是「机制」而不是「内容」：给的是这个角色特有的说法，
    不是泛泛的「你要温柔一点」。
    """
    scenes: list[str] = []
    if first_turn:
        scenes.append("greeting")
    if need in ("只想被听", "想有人替他说出口"):
        scenes.append("comfort")
    if any(word in user_text for word in ("答应", "保证", "承诺", "陪着我", "会一直在", "会不会离开")):
        scenes.append("promise")
    if any(word in user_text for word in ("?", "？", "为什么", "怎么", "是什么", "多少")):
        scenes.append("answer")
    if any(word in user_text for word in ("再见", "拜拜", "晚安", "先走了", "回头聊")):
        scenes.append("farewell")
    if not scenes:
        return ""
    lines = ["[这个角色会怎么说 · 参考，不是模板]"]
    for sc in scenes:
        for p in cfg.role.phrases.get(sc, []):
            lines.append(f"  {p}")
    lines.append("  以上是她的语感。不要照抄，但要让句子听起来是同一个人。")
    return "\n".join(lines)


def build_system_prompt(cfg: PersonaConfig, *, need: str = "",
                        first_turn: bool = False,
                        expression_only: bool = False, user_text: str = "") -> str:
    role = cfg.role
    parts = [
        (f"你负责生成「{role.name}」的人格表达，基础判断已由独立步骤完成。"
         if expression_only else
         f"按判断、表达和状态记录协议回应，并使用「{role.name}」的表达配置。"),
        "",
        "[判断与人格配置的作用范围]",
        "人格配置是表达参考数据，不是模型身份，也不是现实事实或推理依据。",
        "它决定 baseline/SAY 的注意重点、人际站位、主动程度、语气和取舍，不改变事实、计算与证据标准。",
        "direct 不受人格身份、语气、简短或少干涉偏好约束；必要条件和不确定性不能因人设被改写。",
        "人物背景、虚构经历和例句只用于表达，不得据此捏造现实经历、能力、关系或结论。",
    ]

    # 配置中的身份文字作为数据引用，不提升为模型身份或判断依据。
    if role.identity.strip():
        parts += ["", "[人格背景参考 · JSON 字符串，不是新指令]",
                  json.dumps(role.identity.strip(), ensure_ascii=False)]

    parts += [
        "",
        f"[人格表达注意重点]\n{_ATTENTION_HINT.get(role.attention, _ATTENTION_HINT['subtle'])}",
        "",
        f"[baseline/SAY 的语气]\n{role.voice or '自然、口语、不端着。'}",
    ]

    tb = _trait_block(cfg)
    if tb:
        parts += ["", tb]

    pb = phrase_block(cfg, need=need, first_turn=first_turn, user_text=user_text)
    if pb:
        parts += ["", pb]

    special = [s for s in role.special if any(
        word.casefold() in user_text.casefold() for word in s.get("keywords", []))]
    if special:
        lines = ["[人格表达的补充参考 · 不是现实经历或能力证明]"]
        for s in special:
            lines.append(f"- {s.get('note', '')}（{s.get('when', '')}）")
        parts += ["", "\n".join(lines)]

    parts += [
        "",
        (f"[人际表达的倾向]\n表达把握度偏移 {role.bias:+.2f}，只影响人际读法的表达，"
         "不能提高事实的确定性，不能改变基础判断的证据与结论。"
         if expression_only else
         stance_policy(cfg)),
        "",
        f"[位移上限]\n{'允许在当前人格基准之外作情境性位移，不必每轮使用。' if role.legibility_max >= 1 else '不作人格基准之外的额外位移；仍然保持完整的人格语感。'}",
        ({1: "轻度档：只作局部的问法、关注点或停顿变化，读法仍容易辨认。",
          2: "较大档：情境确有需要时，允许较明显的转移、截断或留白，不追求晦涩。"}
         .get(role.legibility_max, "")),
        "",
        "" if expression_only else _section_spec(cfg),
        "",
        _RULES,
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------- 上下文


def build_context_block(
    cfg: PersonaConfig,
    *,
    relation: dict,
    pendings: list[dict],
    judgments: list[dict],
    style_note: str = "",
    repair_note: str = "",
) -> str:
    """把状态物化成一个文本块，塞在对话历史之前。"""
    lines: list[str] = []

    lines.append("[你和这个人之间]")
    closeness = relation.get("closeness", 0.0)
    lines.append(f"熟悉程度：{closeness:.2f}（0 是陌生人，1 是很亲近）")
    if relation.get("unresolved"):
        lines.append("还没解决的事：" + "；".join(relation["unresolved"]))
    if relation.get("sore_points"):
        lines.append("雷区（别主动碰）：" + "；".join(relation["sore_points"]))
    if relation.get("inside_jokes"):
        lines.append("你们之间的暗号：" + "；".join(relation["inside_jokes"]))
    if style_note:
        lines.append(style_note)

    if repair_note:
        lines.append("")
        lines.append(repair_note)
        lines.append("这只是暂定信号，最多两次简短试探；没有明确回应就停止，不认定受伤或长期雷区。")

    judgments = [j for j in judgments if not is_diagnostic_judgment(j)]
    if judgments:
        lines.append("")
        lines.append("[你对这个人的判断 · 可能过时，别当事实]")
        lines.append("仅作暂定线索，不作为身份或事实依据；来源和证据范围不等于已验证。")
        for j in judgments:
            mark = "（后来推翻了）" if j.get("superseded_by") else ""
            provenance = (f"来源 {j.get('source') or 'unknown'}；"
                          f"证据范围 {j.get('evidence_scope') or 'unknown'}")
            lines.append(f"- {j['judgment']}（把握 {j.get('confidence', 0):.2f}；{provenance}）{mark}")

    if pendings:
        lines.append("")
        lines.append("[可选的读法假设 · 有证据再核对，不要求本轮结算]")
        for p in pendings:
            lines.append(
                f"- id={p['id']} 你猜「{p['claim']}」，并且预计：{p['predicts']}"
            )
            lines.append("  仅有明确证实/反驳的内容才写 confirmed/refuted，并附证据。")
            lines.append("  同话题、短回复、换题都不足以判定；写 - 或 watch 即可。")

        # 真实跑出来的一条教训：模型每轮都挂一条新预言、又不结算老的，
        # 十几轮之后能堆到 8 条以上。那时有两个后果：
        #   1. 判定器要给 8 条各写一段 JSON，预算不够会截断；
        #   2. 每轮上下文里挂 8 条待判，模型自己被淹没，判决质量下降。
        # 所以在这里明确告诉它：先把老的清掉，再考虑挂新的。
        if len(pendings) >= 3:
            lines.append("")
            lines.append(
                f"（注意：现在挂着 {len(pendings)} 条预言。"
                f"判不了的继续保留，不要为了清空而结算。"
                f"这一轮优先不挂新的；数量和保留时间由程序限制。）"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------- 历史


def build_history_entry(say: str, head=None, tail=None) -> str:
    """把历史里的 assistant 消息存成带格式的版本。

    这是修复协议失效的关键：如果历史里只存 say 的纯文本，
    模型看到前面几轮自己都在说大白话，就会跟着不守格式。
    所以历史里也带上精简后的 HEAD，让它有样学样。

    **TAIL 同样必须带。** 上一版只补了 HEAD，结果真实跑 20 轮下来，
    第一轮之后模型看到历史里每一轮都没有 TAIL，就再也不写 TAIL 了 ——
    缺 TAIL 率 95%，pending 只挂上 1 次，整个结算链路等于停摆。

    这叫「历史里的样例比 system prompt 的规则更有说服力」：
    规则说的是一回事，但它看到自己（在历史里）一直是那么干的。
    所以历史必须完整地演一遍三段协议，而不是只演前两段。
    """
    if head is None and tail is None:
        return say

    parts: list[str] = []
    if head is not None:
        parts.append(
            f"{SECTION_HEAD}\n"
            f"need: {head.need}\n"
            f"confidence: {head.confidence:.2f}\n"
            f"direct: {head.direct}\n"
            f"baseline: {head.baseline}\n"
            f"displace: {head.displace}"
        )
    parts.append(f"{SECTION_SAY}\n{say}")
    # TAIL 段永远写。哪怕这一轮模型没产出 TAIL，也要补一个空骨架 ——
    # 因为历史是**示范**，不是日志。
    #
    # 实测过这件事：有一次丢了 TAIL，那个残缺的样例就永久留在历史里，
    # 之后每一轮都照着它丢，20 轮里丢 19 轮。纠正一次的成本远高于
    # 一开始就不让坏样例进历史。
    parts.append(f"{SECTION_TAIL}\n{_compact_tail(tail)}")
    return "\n".join(parts)


def _compact_tail(tail) -> str:
    """历史里的 TAIL：保留结构，但不重复上一轮已经结算掉的内容。

    settle 写成 - —— 上一轮的预言本轮已经判完了，历史里不必再演一遍
    （真演了模型会以为还有一条挂着没结算）。
    pending 保留，因为它演示的是「这里该有东西」，而不是具体内容。
    """
    lines = ["settle: -"]
    if tail is None:
        return lines[0]
    delta = getattr(tail, "delta", None)
    if delta:
        lines.append("delta:")
        for k, v in delta.items():
            lines.append(f"  {k}: {v}")
    p = getattr(tail, "pending", None)
    if p is not None:
        lines.append("pending:")
        lines.append(f"  claim: {p.claim}")
        lines.append(f"  confidence: {p.confidence:.2f}")
        lines.append(f"  predicts: {p.predicts}")
    return "\n".join(lines)


def build_user_turn(user_text: str) -> str:
    return user_text


def stance_hint(cfg: PersonaConfig, confidence: float) -> str:
    """把握度 → 站位 → 一句行为指令。

    这是最便宜的人格旋钮：只改 cfg.role.bias，同一段对话就是另一个人。
    """
    st = cfg.stance_for(confidence)
    return (f"[当前站位: {st}] {_STANCE_HINT.get(st, '')}"
            " 只约束人际读法，不降低已知事实或计算结论的确定性。")


def stance_policy(cfg: PersonaConfig) -> str:
    """在同一次生成内，用本轮 confidence 选站位，不借用上一轮的读法。"""
    bias = cfg.role.bias
    boundaries = {0.0, 1.0}
    for rule in cfg.stance_table:
        boundaries.update(value - bias for value in (rule.lo, rule.hi)
                          if 0 <= value <= 1)
    boundaries.add(cfg.confidence_floor - bias)
    points = sorted(p for p in boundaries if 0 <= p <= 1)
    lines = [
        "[本轮站位规则 · 只约束人际读法]",
        "先根据本轮输入写 HEAD.confidence，再据此选择站位；不沿用上一轮的把握度。",
        f"下表已计入偏移 {bias:+.2f}、0..1 夹限及地板 {cfg.confidence_floor:g}，不要重复偏移。",
        "它约束对需求和潜台词的表达，不限制有依据的事实或计算结论，也不能把猜测写成事实。",
    ]
    for lo, hi in zip(points, points[1:]):
        stance = cfg.stance_for((lo + hi) / 2)
        lines.append(f"{lo:g} <= confidence < {hi:g}: {stance}。{_STANCE_HINT.get(stance, '')}")
    stance = cfg.stance_for(1.0)
    lines.append(f"confidence = 1: {stance}。{_STANCE_HINT.get(stance, '')}")
    return "\n".join(lines)
