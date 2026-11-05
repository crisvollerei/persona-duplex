"""暂定关系信号的修复与停止追问。

语域变化只是观察线索，不能证明用户受伤。短回复、换题或语气放松均保持未知。
只有明确的接受/停止内容才用于对应收尾；至多两次尝试后暂停，不自动创建长期雷区。
这里管理的是尝试状态，不是用户心理真值。residual 保留用于旧账本与显式实验接口。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 状态

OPEN = "open"            # 检测到了，还没处理
REPAIRING = "repairing"  # 已经发出修复，等反应
HEALED = "healed"        # 修复成功，销账
RESIDUAL = "residual"    # 旧记录/显式实验的残留状态；默认不据沉默产生
PAUSED = "paused"        # 停止追问；不是已证实的受伤或长期雷区

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
        return cls(**{k: d[k] for k in
                      ("note", "turn_seq", "level", "attempts", "state", "created") if k in d})


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

    跟校准账本分开：这里记尝试及反馈状态，不把它当作已证实的心理结论。
    """

    items: list[Repair] = field(default_factory=list)

    def open(self, note: str, turn_seq: int, created: str, level: str) -> Repair:
        """开一条修复记录。同一条事还在处理中就复用，不重复开。

        同一条线索可能有多个 pending 指向它，复用记录才能约束总尝试次数。
        """
        for r in self.items:
            if r.note == note and r.state in (OPEN, REPAIRING):
                return r
        r = Repair(note=note, turn_seq=turn_seq, level=level,
                   state=OPEN, created=created)
        self.items.append(r)
        return r

    def active(self) -> list[Repair]:
        """还在处理中的修复（open + repairing）。

        open      还没发出修复语
        repairing 发出去了，等反应
        healed / residual / paused 不算 active —— 不再主动追问
        """
        return [r for r in self.items if r.state in (OPEN, REPAIRING)]

    def mark_repairing(self, note: str) -> None:
        """记一次「已经发出修复语」。

        **OPEN 和 REPAIRING 都要计数。**

        只在 OPEN 时 +1 的话，attempts 会封顶在 1 —— 第一次之后状态
        变成 REPAIRING，再也不满足条件。而 engine 里有一条
        「attempts >= 2 就暂停（不再反复道歉）」的规则，
        它依赖这个计数能到 2。计数卡住等于那条规则从未生效。

        实测：连续调用 5 次 mark_repairing，attempts 始终是 1。
        """
        for r in self.items:
            if r.note == note and r.state in (OPEN, REPAIRING):
                r.state = REPAIRING
                r.attempts += 1

    def close(self, note: str, *, healed: bool) -> None:
        """结案。healed=True 销账，False 转长期雷区（residual）。"""
        for r in self.items:
            if r.note == note and r.state in (OPEN, REPAIRING):
                r.state = HEALED if healed else RESIDUAL

    def pause(self, note: str) -> None:
        for r in self.active():
            if r.note == note:
                r.state = PAUSED

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

    只识别有限的明确接受/停止措辞，不以语域回落或话题重合判定原谅。
    这是局部词法信号，不是完整的语义或心理判读。
    """
    if not user_text:
        return "unclear"

    # Only explicit content can close an interpretation; register is not consent.
    forgiving = any(k in user_text for k in (
        "没事", "没关系", "不要紧", "别放心上", "不怪你", "不用道歉",
        "我没有生气", "我没生气", "我也有问题"))
    refusing = any(k in user_text for k in (
        "不用说了", "别说了", "别再道歉", "不要再道歉", "不想聊这个", "没必要再说"))
    negated = any(k in user_text for k in ("不是没事", "不是没关系", "并非没事", "别说没事"))
    if refusing:
        return "rejected"
    if forgiving and not negated and "?" not in user_text and "？" not in user_text:
        return "healed"
    return "unclear"
