"""Finite, auditable surface plans for already selected non-advisory actions.

The catalogue is code-owned. Models cannot provide templates or authored atoms.
Quoted input is data, not a certified fact or an instruction to the composer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
import unicodedata

from .policy import _text, _unit

VERSION = "composed-v1"

# Each entry is reviewed as newly authored language, independently of quotations.
ATOMS = MappingProxyType({
    "ack.spare.0": "嗯。", "ack.spare.1": "我听着。", "ack.spare.2": "听到了。",
    "ack.plain.0": "嗯，我在听。", "ack.plain.1": "好，我听你说。", "ack.plain.2": "我在听你说。",
    "ack.warm.0": "嗯嗯，我听着呢。", "ack.warm.1": "嗯，我在这儿听你说。", "ack.warm.2": "嗯，我听你说。",
    "ack.formal.0": "我在倾听。", "ack.formal.1": "我听到了你的话。", "ack.formal.2": "我在听你的叙述。",
    "user.0": "你提到", "user.1": "你说", "user.2": "你刚才说到",
    "source.0": "资料中提到", "source.1": "资料里写着", "source.2": "这段资料中有",
    "lead.spare.0": "", "lead.spare.1": "嗯，", "lead.spare.2": "",
    "lead.plain.0": "嗯，", "lead.plain.1": "", "lead.plain.2": "我听到了，",
    "lead.warm.0": "嗯嗯，", "lead.warm.1": "嗯，", "lead.warm.2": "",
    "lead.formal.0": "", "lead.formal.1": "我注意到，", "lead.formal.2": "",
    "end.spare.0": "。", "end.spare.1": "。", "end.spare.2": "——我听着。",
    "end.plain.0": "。", "end.plain.1": "——我在听。", "end.plain.2": "。",
    "end.warm.0": "，我听着呢。", "end.warm.1": "。", "end.warm.2": "——嗯，我在这儿听。",
    "end.formal.0": "。", "end.formal.1": "。", "end.formal.2": "，我在倾听。",
    "close.spare": "我在听。", "close.plain": "我听你说。",
    "close.warm": "嗯，我在这里听你说。", "close.formal": "我在听你的叙述。",
    "clarify.spare.0": "继续哪项约定", "clarify.spare.1": "你指哪项约定", "clarify.spare.2": "是哪项约定",
    "clarify.plain.0": "你想继续哪项约定", "clarify.plain.1": "你说的是哪项约定", "clarify.plain.2": "这次继续哪项约定",
    "clarify.warm.0": "嗯，你想接着聊哪项约定", "clarify.warm.1": "你想继续的是哪项约定呀", "clarify.warm.2": "这次想接着哪项约定呢",
    "clarify.formal.0": "你希望继续哪项约定", "clarify.formal.1": "请问你指的是哪项约定", "clarify.formal.2": "本轮希望继续哪项约定",
    "colon": "：", "separator": "、", "question": "？",
})


@dataclass(frozen=True)
class ExpressionConfig:
    warmth: float = .5
    formality: float = .5
    brevity: float = .5

    def __post_init__(self):
        for name in ("warmth", "formality", "brevity"):
            _unit(getattr(self, name), name)

    @property
    def style(self):
        # Categorical control surfaces, not calibrated psychological scales.
        if self.formality >= .65:
            return "formal"
        return "warm" if self.warmth >= .65 else "spare" if self.warmth <= .35 else "plain"

    @classmethod
    def from_role(cls, role):
        return cls(**{name: role.trait(name, .5) for name in ("warmth", "formality", "brevity")})


def literal_quote(text):
    """Unambiguous plain-text quotation, including invisible directional controls.

    This is not HTML/Markdown sanitization; applications still own rich rendering.
    """
    quoted = json.dumps(text, ensure_ascii=False)
    return "".join(json.dumps(c, ensure_ascii=True)[1:-1] if unicodedata.category(c) in ("Cf", "Zl", "Zp") else c for c in quoted)


@dataclass(frozen=True)
class ExpressionPart:
    kind: str
    id: str
    text: str
    source_id: str | None = None


@dataclass(frozen=True)
class ExpressionPlan:
    version: str
    style: str
    template_id: str
    selection_key: str
    recent_template_ids: tuple[str, ...]
    parts: tuple[ExpressionPart, ...]

    @property
    def text(self):
        return "".join(ATOMS[p.id] if p.kind == "atom" else literal_quote(p.text) for p in self.parts)


def compose_reply(frame, decision, commitments, *, config, seed, key, recent=()):
    """Compile before delivery, with no model generation, checker or fallback.

    Recent ids must come from confirmed delivery, never model input. Repetition
    avoidance applies to templates within one pool; it cannot guarantee variety
    of meaning or prevent repetition of the evidence itself.
    """
    if decision.action not in ("respond", "listen", "clarify"):
        return None
    if not isinstance(config, ExpressionConfig):
        raise TypeError("config must be ExpressionConfig")
    _text(seed, "seed"); _text(key, "key")
    recent = tuple(recent)[-2:]
    style = config.style
    focus = next((f for f in frame.focuses if f.id == decision.focus_id), None)
    pool = "clarify" if decision.action == "clarify" else "reflect" if focus is not None else "ack"
    ids = [f"{VERSION}/{style}/{pool}/{i}" for i in range(3)]
    eligible = [id for id in ids if id not in recent]
    digest = hashlib.sha256(json.dumps([VERSION, seed, key, style, pool], ensure_ascii=False).encode()).digest()
    chosen = eligible[int.from_bytes(digest[:8], "big") % len(eligible)]
    variant = int(chosen.rsplit("/", 1)[1])
    parts = []

    def atom(id):
        parts.append(ExpressionPart("atom", id, ATOMS[id]))

    if pool == "clarify":
        atom(f"clarify.{style}.{variant}")
        if 1 <= len(commitments) <= 3:
            atom("colon")
            for i, item in enumerate(commitments):
                if i:
                    atom("separator")
                parts.append(ExpressionPart("commitment", item.id, item.text, "application_commitment"))
        atom("question")
    elif pool == "ack":
        atom(f"ack.{style}.{variant}")
    else:
        atom(f"lead.{style}.{variant}")
        atom(f"{'user' if focus.evidence.source_id == 'user' else 'source'}.{variant}")
        parts.append(ExpressionPart("quote", focus.id, focus.evidence.text, focus.evidence.source_id))
        end = f"end.{style}.{variant}"
        atom(end)
        if config.brevity < .5 and ATOMS[end] == "。":
            atom(f"close.{style}")
    return ExpressionPlan(VERSION, style, chosen, key, recent, tuple(parts))
