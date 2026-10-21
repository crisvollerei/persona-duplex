"""受伤修复。

这是全系统最贵、最容易漏的一条链路。

用户在受伤之后不会说「你冒犯我了」，他会把句子变短、不再开玩笑、
不再反问你。所以检测靠语域，不靠内容。

但检测只是第一步。真正决定这段关系能不能撑住的，是**修复**：

    不修复       → 用户认为你根本没意识到，比冒犯本身更伤
    修得太重     → 把没发生的事说成发生了，反而更像猜错
    修得太轻     → 觉得在敷衍

而且修复是「位移不许声明」这条规则的**唯一例外**：

    声明位移 = 索取信用（关于我）
    承认损伤 = 请求修复（关于你）

「我刚才那句是不是有点重」不是在说「你看我多克制」，
是在问「你有没有被伤到」。前者关于我，后者关于你，所以它不违规。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 状态

OPEN = "open"            # 检测到了，还没处理
REPAIRING = "repairing"  # 已经发出修复，等反应
HEALED = "healed"        # 修复成功，销账
RESIDUAL = "residual"    # 修复失败或没回应，转为长期雷区

# 修复措辞的三种力度
SOFT = "soft"      # 「我刚才是不是说重了」
PLAIN = "plain"    # 「刚才那句话我不该那么说」
DIRECT = "direct"  # 「我道歉，那句话是我不对」

_HEAVY = re.compile(r"(对不起|抱歉|我错|我不该|是我不好)")
_SOFT = re.compile(r"(是不是|有点|可能|也许|是不是有点)")


@dataclass
class Repair:
    """一次修复动作。"""

    note: str                 # 对应 relation.unresolved 里的那条
    turn_seq: int
    level: str = PLAIN
    attempts: int = 0
    state: str = OPEN
    created: str = ""

    def as_dict(self) -> dict:
        return {
            "note": self.note, "turn_seq": self.turn_seq, "level": self.level,
            "attempts": self.attempts, "state": self.state, "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Repair":
        return cls(**{k: d.get(k) for k in
                      ("note", "turn_seq", "level", "attempts", "state", "created")})


# ---------------------------------------------------------------- 检测


def is_repair_attempt(say: str) -> bool:
    """这一句是不是在修复。

    注意：这里**允许**出现「我本来想」之类的自我指涉，因为修复本来
    就要承认自己说错了。跟平时的「位移不许声明」不冲突。
    """
    if not say:
        return False
    has_apology = bool(_HEAVY.search(say))
    has_softener = bool(_SOFT.search(say))
    # 「对不起」单用可能是别的语境，要求同时提到刚才/那句话/说
    refers_back = any(k in say for k in ("刚才", "那句", "之前", "刚才那句",
                                          "上一条", "刚说"))
    return (has_apology or has_softener) and refers_back


def repair_level(say: str) -> str:
    """判断修复的力度。太重和太轻都不好。"""
    if not say:
        return SOFT
    if _HEAVY.search(say) and not _SOFT.search(say):
        return DIRECT
    if _SOFT.search(say) and not _HEAVY.search(say):
        return SOFT
    return PLAIN


def pick_level(*, confidence: float, formality_rise: float) -> str:
    """根据线索强度选修复力度。

    线索越弱，越要轻 —— 把没发生的事说成发生了，比不修更糟。
    """
    if confidence >= 0.8 and formality_rise >= 0.45:
        return DIRECT
    if confidence >= 0.6 and formality_rise >= 0.3:
        return PLAIN
    return SOFT


def compose_repair(level: str, *, what: str = "") -> str:
    """生成修复用的那一句。

    刻意短。修复不是解释，解释会变成辩解。
    """
    target = f"「{what}」" if what else "刚才那句"
    if level == DIRECT:
        return f"{target}我说重了，对不起。"
    if level == PLAIN:
        return f"{target}……我不该那么说。"
    return f"{target}是不是有点重。"


# ---------------------------------------------------------------- 生命周期


@dataclass
class RepairBook:
    """关系账本里的修复记录。

    跟校准账本分开：这里记的不是「读得准不准」，是「关系有没有被碰伤」。
    """

    items: list[Repair] = field(default_factory=list)

    def open(self, note: str, turn_seq: int, created: str, level: str) -> Repair:
        for r in self.items:
            if r.note == note and r.state in (OPEN, REPAIRING):
                return r
        r = Repair(note=note, turn_seq=turn_seq, level=level,
                   state=OPEN, created=created)
        self.items.append(r)
        return r

    def active(self) -> list[Repair]:
        return [r for r in self.items if r.state in (OPEN, REPAIRING)]

    def mark_repairing(self, note: str) -> None:
        for r in self.items:
            if r.note == note and r.state == OPEN:
                r.state = REPAIRING
                r.attempts += 1

    def close(self, note: str, *, healed: bool) -> None:
        for r in self.items:
            if r.note == note and r.state in (OPEN, REPAIRING):
                r.state = HEALED if healed else RESIDUAL

    def to_list(self) -> list[dict]:
        return [r.as_dict() for r in self.items]

    @classmethod
    def from_list(cls, rows: list[dict]) -> "RepairBook":
        return cls(items=[Repair.from_dict(r) for r in rows or []])


# ---------------------------------------------------------------- 判读反应


def judge_repair_response(user_text: str, *, baseline_formality: float = 0.3,
                          similarity=None) -> str:
    """看用户对修复的反应。

    返回 healed / rejected / unclear。

    注意这里不能用「话题重合度」——用户不需要复述刚才那句话才算原谅你。
    真正该看的是语域有没有回落：重新用语气词、句子变长、反问回来。
    """
    if not user_text:
        return "unclear"

    from .settle import formality_score

    f = formality_score(user_text)
    relaxed = f <= baseline_formality + 0.1
    has_filler = any(e in user_text for e in "哈哈嘿嘿呃嗯嘛呀啦～~")
    longer = len([c for c in user_text if not c.isspace()]) >= 8

    # 明确接受：算了/没事/没关系
    forgiving = any(k in user_text for k in
                    ("没事", "没关系", "算了", "不要紧", "别放心上", "我也有问题"))
    # 明确拒绝：不用/不必/我说了
    refusing = any(k in user_text for k in
                   ("不用", "不必", "没必要", "我说了", "别说了", "不想聊"))

    if forgiving:
        return "healed"
    if refusing:
        return "rejected"

    # 「知道了。」「嗯。」这类极短的平静回应，是最典型的软拒绝。
    # 它不像吵架，所以最容易被当成没事 —— 而它恰恰意味着「别再往下说了」。
    n = len([c for c in user_text if not c.isspace()])
    if n <= 6 and not has_filler and "？" not in user_text and "?" not in user_text:
        return "rejected"

    if relaxed and (has_filler or longer):
        return "healed"
    if f > baseline_formality + 0.3:
        return "rejected"
    return "unclear"
