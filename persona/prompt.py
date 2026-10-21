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
  predicts: 具体会观察到什么

[位移的规则]
- direct 是你的真心话，永远不会被发出去。displace 写的是一句「合格的话」怎么绕开了它。
- 绕过的方式是留痕，不是声明。绝对不要写「我本来想说……」「其实我差点……」这类句子，
  那是在索取信用，比直接说更糟。
- 位移只在情绪承重的时候发生。绝大多数轮次应该是 displace: none。
- 一轮最多一次位移。不要每句话都留白。"""


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
    ]
    return "\n".join(parts)


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


def build_user_turn(user_text: str) -> str:
    return user_text


def stance_hint(cfg: PersonaConfig, confidence: float) -> str:
    """把握度 → 站位 → 一句行为指令。

    这是最便宜的人格旋钮：只改 cfg.role.bias，同一段对话就是另一个人。
    """
    st = cfg.stance_for(confidence)
    return f"[当前站位: {st}] {_STANCE_HINT.get(st, '')}"
