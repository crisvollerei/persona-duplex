"""配置加载。

人格不是一段长 prompt，是几个数：
    bias            —— 把握度偏移，决定敢不敢猜
    attention       —— 只读表层 / 也读潜台词
    legibility_max  —— 位移上限（0=直说 1=有位移不声明 2=大位移）

这三个数必须稳定，因为它们是「可读性 = 位移 ÷ 基线」里的分母。
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 站位档位

STANCE_DIRECT = "direct"          # 不铺垫，直接给
STANCE_GUESS = "guess_first"      # 先猜后给
STANCE_ASK = "ask_only"           # 只问，不给结论
STANCE_UNSURE = "admit_unsure"    # 明说不知道

STANCES = (STANCE_DIRECT, STANCE_GUESS, STANCE_ASK, STANCE_UNSURE)

# 位移类型。none 是常态，不是失败。
DISPLACE_NONE = "none"
DISPLACE_KINDS = (
    DISPLACE_NONE,
    "降级",   # 判断 → 问题
    "转移",   # 主语从我 → 你
    "截断",   # 长度断掉
    "悬置",   # 结论停在半路
    "反差",   # 语域与内容不匹配
    "多余",   # 解释过量
)


@dataclass(frozen=True)
class StanceRule:
    lo: float
    hi: float
    stance: str


@dataclass
class Limits:
    """在线写入的额度。慢层字段不在此列，只能由后台写。"""

    closeness_step: float = 0.05    # relation.closeness 单轮最大变化
    unresolved_max: int = 3         # 未解决事项最多并存几条
    pending_max: int = 1            # 每轮最多挂起几条预言

    def as_dict(self) -> dict:
        return {
            "closeness_step": self.closeness_step,
            "unresolved_max": self.unresolved_max,
            "pending_max": self.pending_max,
        }


@dataclass
class Role:
    name: str = "默认"
    bias: float = 0.0
    attention: str = "subtle"
    legibility_max: int = 2
    voice: str = ""


@dataclass
class PersonaConfig:
    role: Role = field(default_factory=Role)
    needs: list[str] = field(default_factory=list)
    stance_table: list[StanceRule] = field(default_factory=list)
    limits: Limits = field(default_factory=Limits)
    watch_ttl_hours: float = 72.0
    confidence_floor: float = 0.25

    # 话题判定：接 LLM 做语义判断，异步跑，不开思考模式
    judge_enabled: bool = True
    judge_grace: float = 0.35

    # ---------------------------------------------------------- 判定

    def stance_for(self, confidence: float) -> str:
        """把握度 + 角色偏移 → 站位。

        这个函数是整个系统里最便宜的人格旋钮：只改 bias，
        同一段对话看起来就是另一个人。
        """
        eff = max(0.0, min(1.0, confidence + self.role.bias))
        for rule in self.stance_table:
            if rule.lo <= eff < rule.hi:
                return rule.stance
        return self.stance_table[-1].stance if self.stance_table else STANCE_UNSURE

    def effective_confidence(self, confidence: float) -> float:
        return max(0.0, min(1.0, confidence + self.role.bias))

    def can_displace(self, kind: str) -> bool:
        if kind in (DISPLACE_NONE, ""):
            return True
        if self.role.legibility_max <= 0:
            return False
        return kind in DISPLACE_KINDS

    # ---------------------------------------------------------- 加载

    @classmethod
    def load(cls, path: str | Path) -> "PersonaConfig":
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "PersonaConfig":
        r = raw.get("role", {})
        role = Role(
            name=r.get("name", "默认"),
            bias=float(r.get("bias", 0.0)),
            attention=r.get("attention", "subtle"),
            legibility_max=int(r.get("legibility_max", 2)),
            voice=r.get("voice", ""),
        )

        needs = list(raw.get("needs", {}).get("vocabulary", []))

        table: list[StanceRule] = []
        for row in raw.get("stance_table", {}).get("rows", []):
            lo, hi, stance = row[0], row[1], row[2]
            if stance not in STANCES:
                raise ValueError(f"未知站位: {stance}")
            table.append(StanceRule(float(lo), float(hi), stance))
        table.sort(key=lambda x: x.lo)
        if not table:
            table = [
                StanceRule(0.80, 1.01, STANCE_DIRECT),
                StanceRule(0.50, 0.80, STANCE_GUESS),
                StanceRule(0.25, 0.50, STANCE_ASK),
                StanceRule(0.00, 0.25, STANCE_UNSURE),
            ]

        lim = raw.get("limits", {})
        limits = Limits(
            closeness_step=float(lim.get("closeness_step", 0.05)),
            unresolved_max=int(lim.get("unresolved_max", 3)),
            pending_max=int(lim.get("pending_max", 1)),
        )

        watch = raw.get("watch", {})
        judge = raw.get("judge", {})
        return cls(
            role=role,
            needs=needs,
            stance_table=table,
            limits=limits,
            watch_ttl_hours=float(watch.get("ttl_hours", 72.0)),
            confidence_floor=float(watch.get("confidence_floor", 0.25)),
            judge_enabled=bool(judge.get("enabled", True)),
            judge_grace=float(judge.get("grace", 0.35)),
        )
