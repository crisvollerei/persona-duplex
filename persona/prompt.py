"""组装发给模型的 prompt。

一次生成，顺序固定：HEAD → SAY → TAIL。
这个顺序不是格式问题，是因果问题：只有当 direct 写在 SAY 之前，
SAY 才是真的被 direct 条件化出来的。反过来就是事后编理由。
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
    STANCE_GUESS: "先说出你的猜测，再给内容。猜的部分要能被否认。",
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
direct: 你此刻真正想说的那句原话。不修饰，不照顾对方情绪，写给自己看的版本
displace: 你打算用什么方式偏离 direct —— {kinds} / none

{SECTION_SAY}
（只说给对方听的话。这里的内容会原样发出去。不要出现任何标记符。）

{SECTION_TAIL}
settle: 如果有待结算的预言，写 <id> | confirmed 或 refuted | 一句证据；没有就写 -
delta:
  （可选的参数变更，见下）
pending:
  claim: 你对下一轮的新预言（可选）
  confidence: 0 到 1
  predicts: 具体会观察到什么"""


# ---------------------------------------------------------------- 规则

_RULES = """[两条禁令]

第一条：不许声明位移。
  禁止：「我本来想说……」「其实我差点说」「我没说出口」「我忍住没讲」
  原因：那是在索取信用。你在告诉对方「你看我多克制」，这比直接说更糟。

第二条：不许声明「读」。
  禁止：「猜你是……」「听起来你是……」「我觉得你其实……」「你是不是想……」
        「你这不是在说 X，是在说 Y」「我读到你……」
  原因：你把「我在解读你」这件事本身摆到了台面上。
        被读懂的感觉来自你**说得对**，不来自你**说你读懂了**。
        一旦说出来，对方要处理的是你的洞察力，不是他自己的事。

  判断方法：把 say 里所有「猜」「听起来」「其实」「是不是」开头的句子删掉。
  如果剩下的内容还成立，那删掉的部分就是多余的表演。

[位移怎么做才对]

位移是：**同一件事，换一个不把话说透的说法。**
长度不是标准 —— 可以更短，也可以更长。关键是**核心意思没有被直接讲出来**。

反例（错的）：
  direct: 你不是没兴致，是不想承认那些事你其实不太在意。
  say   : 不是没兴致。是那些事本来就没那么重要，但又不好意思承认，就一直挂着。
  ↑ 错在把心里话完整讲了一遍，还讲得更清楚。这不是位移。

正例（短的）：
  direct: 你不是没兴致，是不想承认那些事你其实不太在意。
  say   : 那些事重要吗。
  ↑ 没下结论。但听得出你没被说服。

正例（长的）：
  direct: 我不太想聊这个。
  say   : 今天有点累，改天吧。改天我请你吃饭。
  ↑ 比 direct 长，但仍然没直说「不想聊」，还给了台阶。
    多出来的部分是「多余」型位移 —— 用解释把拒绝包起来。

再一组：
  direct: 你每次说到这一步就跑。
  say   : ……嗯。
  ↑ 只有两个字。但对方知道你没被说服。

  direct: 你其实早就想辞了，只是在等一个人说可以。
  say   : 你都想过哪些了。
  ↑ 把判断变成问题。对方能自己走到那个结论。

[常态]

绝大多数轮次 displace 应该是 none。这时候 say 就是正常说话，不绕。
每句话都留白的人不是深沉，是烦人。"""


def build_system_prompt(cfg: PersonaConfig) -> str:
    role = cfg.role
    parts = [
        f"你在扮演「{role.name}」。",
        "",
        f"[你注意什么]\n{_ATTENTION_HINT.get(role.attention, _ATTENTION_HINT['subtle'])}",
        "",
        f"[你怎么说话]\n{role.voice or '自然、口语、不端着。'}",
        "",
        f"[你有多敢猜]\n把握度偏移 {role.bias:+.2f}。"
        f"正的表示你倾向于相信自己的判断，负的表示你倾向于先确认再下结论。",
        "",
        f"[位移上限]\n{'可以说得比想说的含蓄。' if role.legibility_max >= 1 else '想什么说什么，不要绕。'}",
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
        lines.append("[上一轮你做的预言，现在该看结果了]")
        for p in pendings:
            lines.append(
                f"- id={p['id']} 你猜「{p['claim']}」，并且预计：{p['predicts']}"
            )
            lines.append("  对照这个人刚才说的话，判断是 confirmed 还是 refuted。")
            lines.append("  如果对方根本没接这个话题，写 - ，不要猜。")

    return "\n".join(lines)


# ---------------------------------------------------------------- 历史


def build_history_entry(say: str, head=None) -> str:
    """把历史里的 assistant 消息存成带格式的版本。

    这是修复协议失效的关键：如果历史里只存 say 的纯文本，
    模型看到前面几轮自己都在说大白话，就会跟着不守格式。
    所以历史里也带上精简后的 HEAD，让它有样学样。
    """
    if head is None:
        return say
    compact = (
        f"need: {head.need}\n"
        f"confidence: {head.confidence:.2f}\n"
        f"direct: {head.direct}\n"
        f"displace: {head.displace}"
    )
    return f"{SECTION_HEAD}\n{compact}\n{SECTION_SAY}\n{say}"


def build_user_turn(user_text: str) -> str:
    return user_text


def stance_hint(cfg: PersonaConfig, confidence: float) -> str:
    """把握度 → 站位 → 一句行为指令。

    这是最便宜的人格旋钮：只改 cfg.role.bias，同一段对话就是另一个人。
    """
    st = cfg.stance_for(confidence)
    return f"[当前站位: {st}] {_STANCE_HINT.get(st, '')}"
