"""组装发给模型的 prompt。

一次生成，顺序固定：HEAD → SAY → TAIL。
这个顺序不是格式问题，是因果问题：只有当 direct 写在 SAY 之前，
SAY 才是真的被 direct 条件化出来的。反过来就是事后编理由。

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

from .config import (
    DISPLACE_KINDS,
    STANCE_ASK,
    STANCE_DIRECT,
    STANCE_GUESS,
    STANCE_UNSURE,
    PersonaConfig,
)

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
    "subtle": "同时注意对方没说出口的部分：语气、缺口、为什么现在说这个。",
}


def _section_spec(cfg: PersonaConfig) -> str:
    needs = " / ".join(cfg.needs) if cfg.needs else "（未配置）"
    kinds = " / ".join(k for k in DISPLACE_KINDS if k != "none")
    return f"""[输出格式 · 严格按顺序]
{SECTION_HEAD}
need: 从下面这个固定集合里选一个 —— {needs}
confidence: 0 到 1 之间的小数，表示你对上面那个判断有多确定
direct: 一句**不带任何人格色彩的素回复**。见下面「direct 是什么」。
baseline: 当前人格通常会怎样接这句话。保留人格语感，不刻意遮掩、反说或留白。
displace: 相对于上面这个人格基准的额外位移 —— {kinds} / none。none 是常态，不表示去掉人格。

{SECTION_SAY}
（只说给对方听的话。这里的内容会原样发出去。不要出现任何标记符。
 不要写「我猜你……」「先猜一个……」来引出内容 —— 直接说事。）

[direct 是什么 · 这一条最重要]

direct 不是「你的心里话」，是**一个没有性格的助手会给出的回答**。

它回答的是：按事实和逻辑，这句话该怎么接？
它不回答：我作为这个角色会怎么说。

  错：你没吃饭这件事让我有点在意，但我不该多管。
      ↑ 这是「我」的感受。带人格了。
  错：你明明有话没说完，别拿换工作来岔开。
      ↑ 这是「我」对「你」的判断。也带人格了。
  对：先吃点东西吧，饿着更累。
  对：这个得看你自己的情况，别人没法替你决定。
  对：情绪没有对错，你觉得难受就是真实的。
      ↑ 这些是有用的、具体的、中性的回答。谁来都这么说。

为什么必须这样拆分：
  direct 有人格 → say 就只能做「减法」（把人格磨淡）→ 人格越调越弱
  direct 无人格 → say 才是「替换」（换成角色的说法）→ 人格才可控

所以写 direct 的时候，忘掉你扮演的是谁。就当你是个普通助手，
给出最有用的那句。人格是下一步的事。

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

_RULES = """[两条禁令]

第一条：不许声明位移。
  禁止：「我本来想说……」「其实我差点说」「我没说出口」「我忍住没讲」
  原因：那是在索取信用。你在告诉对方「你看我多克制」，这比直接说更糟。

第二条：不许声明「读」。
  禁止：「猜你是……」「听起来你是……」「我觉得你其实……」「你是不是想……」
        「你这不是在说 X，是在说 Y」「我读到你……」

  这一条禁的是**句式**，不是某个词。下面这些全算，一个都不许出现：
    「我猜你……」「我猜你这句……」「我猜这不是……」「我猜这个嗯……」
    「猜一句：……」「先猜一个：……」「我要猜，你……」
  规律很简单：**只要句子里出现「我」在「猜」对方的什么东西，就是违规。**
  不管是「猜」还是「我猜」；不管猜的是人、是话、还是一个语气词。

  可以猜的：第三方（「我猜他也不会来」）、外部事实（「我猜明天下雨」）。
  不可以猜的：**对方**。对方是什么状态、为什么说这句、这句话后面藏着什么
  —— 这些一律不许以「猜」的句式说出来。

  原因：你把「我在解读你」这件事本身摆到了台面上。
        被读懂的感觉来自你**说得对**，不来自你**说你读懂了**。
        一旦说出来，对方要处理的是你的洞察力，不是他自己的事。

  判断方法：把 say 里所有以「猜」「我猜」「先猜」开头的**分句**删掉。
  如果剩下的内容还成立，那删掉的部分就是多余的表演。
  例：
    错：我猜你不是真想问我该不该，是已经想走了。那我先问一句：真辞了，你最怕什么。
    对：真辞了，你最怕什么。
    错：我猜这个「嗯」是认了，但不想往下讲。那就不讲。
    对：那就不讲。

[位移怎么做才对]

先区分两个层次：direct → baseline 是普通人格化；baseline → say 才是这里记录的位移。
baseline 是当前人格的常态表达，不是素回复，也不是永远冻结的模板。
无额外位移时，say 仍然完整保留人格。不要把「有个性」记成位移。

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
    return "[你的行为尺度]\n" + "\n".join(lines)


def phrase_block(cfg: PersonaConfig, *, need: str = "", first_turn: bool) -> str:
    """按场景注入句式（方案 B）。

    不把所有人的话全塞进 prompt —— 那会稀释。
    只在**用得上**的时候给：
      · first_turn        → greeting
      · need ∈ {只想被听, 想有人替他说出口} → comfort
      · 其余不给句式，让它自己说

    这是「机制」而不是「内容」：给的是这个角色特有的说法，
    不是泛泛的「你要温柔一点」。
    """
    scenes: list[str] = []
    if first_turn:
        scenes.append("greeting")
    if need in ("只想被听", "想有人替他说出口"):
        scenes.append("comfort")
    if not scenes:
        return ""
    lines = ["[这个角色会怎么说 · 参考，不是模板]"]
    for sc in scenes:
        for p in cfg.role.phrases.get(sc, []):
            lines.append(f"  {p}")
    lines.append("  以上是她的语感。不要照抄，但要让句子听起来是同一个人。")
    return "\n".join(lines)


def build_system_prompt(cfg: PersonaConfig, *, need: str = "",
                        first_turn: bool = False) -> str:
    role = cfg.role
    parts = [
        f"你在扮演「{role.name}」。",
    ]

    # 长文本人设：身份与信念。有就放，没有就用 voice 兜底。
    if role.identity.strip():
        parts += ["", "[你是谁]", role.identity.strip()]

    parts += [
        "",
        f"[你注意什么]\n{_ATTENTION_HINT.get(role.attention, _ATTENTION_HINT['subtle'])}",
        "",
        f"[你怎么说话]\n{role.voice or '自然、口语、不端着。'}",
    ]

    tb = _trait_block(cfg)
    if tb:
        parts += ["", tb]

    pb = phrase_block(cfg, need=need, first_turn=first_turn)
    if pb:
        parts += ["", pb]

    if role.special:
        lines = ["[关于你自己的几件事]"]
        for s in role.special:
            lines.append(f"- {s.get('note', '')}（{s.get('when', '')}）")
        parts += ["", "\n".join(lines)]

    parts += [
        "",
        f"[你有多敢猜]\n把握度偏移 {role.bias:+.2f}。"
        f"正的表示你倾向于相信自己的判断，负的表示你倾向于先确认再下结论。",
        "",
        f"[位移上限]\n{'允许在当前人格基准之外作情境性位移，不必每轮使用。' if role.legibility_max >= 1 else '不作人格基准之外的额外位移；仍然保持完整的人格语感。'}",
        "",
        _section_spec(cfg),
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
        lines.append("修复只有一次机会，而且必须短。解释会变成辩解。")

    if judgments:
        lines.append("")
        lines.append("[你对这个人的判断 · 可能过时，别当事实]")
        for j in judgments:
            mark = "（后来推翻了）" if j.get("superseded_by") else ""
            lines.append(f"- {j['judgment']}（把握 {j.get('confidence', 0):.2f}）{mark}")

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
    return f"[当前站位: {st}] {_STANCE_HINT.get(st, '')}"
