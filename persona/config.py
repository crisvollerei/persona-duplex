"""人格配置与显式参数状态。

bias / attention / legibility_max 控制站位、注意力和相对人格基准的位移上限。
traits 与身份、语感例句一起描述人格；anchor 是起点而非永远冻结的本质。
参数更新与恢复接口保留，但常规会话不自动把用户性格或预测命中当成长依据。
成长信号尚待讨论；参数是否有效需要受控实验，而非只看账本数值。
"""
from __future__ import annotations

import math
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


def _parse_traits(raw: dict) -> dict[str, "Trait"]:
    """解析 [params] 段。

    两种写法都支持：
        initiative = 0.20                      # 只要一个数，其余取默认
        [params.warmth]                        # 完整写法
        anchor = 0.70
        plasticity = 0.20
        drift = 0.12
        lo = 0.45
        hi = 0.95
    """
    out: dict[str, Trait] = {}
    for name, spec in (raw or {}).items():
        if isinstance(spec, dict):
            out[name] = Trait(
                anchor=float(spec.get("anchor", 0.0)),
                plasticity=float(spec.get("plasticity", 0.0)),
                drift=float(spec.get("drift", 0.0)),
                lo=float(spec.get("lo", 0.0)),
                hi=float(spec.get("hi", 1.0)),
            )
        elif isinstance(spec, (int, float)):
            out[name] = Trait(anchor=float(spec))
    return out


def _parse_phrases(raw: dict) -> dict[str, list[str]]:
    """解析 [phrases] 段。键是场景，值是句式列表。"""
    out: dict[str, list[str]] = {}
    for scene, items in (raw or {}).items():
        if isinstance(items, list):
            vals = [str(x).strip() for x in items if str(x).strip()]
        else:
            vals = [str(items).strip()]
        if vals:
            out[scene] = vals
    return out


@dataclass
class Limits:
    """在线写入的额度。慢层字段不在此列，只能由后台写。"""

    closeness_step: float = 0.05    # relation.closeness 单轮最大变化
    unresolved_max: int = 3         # 未解决事项最多并存几条
    pending_max: int = 1            # 每轮最多挂起几条预言
    pending_total_max: int = 8      # 同时挂着的预言总闸，防止无限堆积

    def as_dict(self) -> dict:
        return {
            "closeness_step": self.closeness_step,
            "unresolved_max": self.unresolved_max,
            "pending_max": self.pending_max,
            "pending_total_max": self.pending_total_max,
        }


@dataclass
class Trait:
    """一个人格参数。

    四件事分开：
        anchor      初始值。人格的「起点」。
        plasticity  可塑性。每次经历能推动它多少。0 = 永远不动。
        drift       允许漂移的范围（相对 anchor 的半径）。
                    plasticity 管「推得多快」，drift 管「能推多远」。
        min/max     参数本身的物理边界（0-1 之间的子区间）。

    为什么 plasticity 和 drift 必须分开：
      · plasticity 高 drift 小 = 这个人容易被触动，但骨子里不变
      · plasticity 低 drift 大 = 这个人平时很稳，但遇到大事会变很多
    两个数描述两件事，合成一个就分不出来了。
    """

    anchor: float
    plasticity: float = 0.0
    drift: float = 0.0
    lo: float = 0.0
    hi: float = 1.0
    value: float | None = None  # 当前值。None = 从 anchor 起步

    def __post_init__(self) -> None:
        # 用 is None 判断，不能用 falsy ——
        # value=0.0 是合法值（参数被推到物理下界），
        # 用 if not self.value 会把它当成「没设置」重置回 anchor。
        # 实测踩过：push_traits 报 value=0.425 而账本实际是 0.0。
        if self.value is None:
            self.value = self.anchor
        self.clamp()

    def clamp(self) -> None:
        """把 anchor 和 value 都夹回合法范围。

        **anchor 也必须夹** —— 它会被长期层推动，不夹就会漂出物理边界。
        实测踩过：affect_range（lo=0.02 hi=0.40 drift=0.10）连续 400 次
        anchor 推动后 anchor=2.15、value=2.05，全部越过 hi=0.40。
        然后 drift_room 的 lo=max(lo, anchor-drift) 反超 hi，
        effective_plasticity 双向返回 0 —— 这个参数从此对任何经历都不再响应，
        永久卡死。这是「人格不该撞墙停住」的反面。

        value 的约束是两段交集：物理边界 [lo,hi] ∩ drange [anchor±drift]。
        交集为空（配置矛盾）时退回 anchor —— 至少不会落在两个范围都不认的地方。
        """
        # 先夹 anchor 本身
        if self.lo <= self.hi:
            self.anchor = max(self.lo, min(self.hi, self.anchor))

        lo = max(self.lo, self.anchor - self.drift)
        hi = min(self.hi, self.anchor + self.drift)
        if lo > hi:
            self.value = self.anchor
            return
        self.value = max(lo, min(hi, self.value))

    def nudge(self, signal: float) -> float:
        """按 plasticity 推动一次，返回实际变化量。

        signal 是记忆系统给出的原始信号（-1..1）。人格只负责把它缩放。
        这就是「记忆不承载人格」的落点：同样的经历，不同人格推出的量不同。

        余弦退火：越靠近边界，同样的信号推得越少。

            plasticity_eff = plasticity × 0.5 × (1 + cos(π · saturation))

        saturation = 当前位置到边界还剩多少余量（0 = 在中心，1 = 贴着边界）。
        所以：
            在 anchor 附近 → 全额推动，人格容易被经历塑造
            逼近边界     → 推动趋近于 0，几乎推不动

        为什么需要它：没有退火时，plasticity 只决定「多快撞到墙」，
        撞到之后所有参数都僵在 drift 边界上 —— 那就变成了另一个固定人格，
        只是恰好落在边上。退火让人格在边界附近「变硬」，
        而不是「撞上去停住」。这也符合直觉：一个人越是走到自己的极端，
        再往前就越难。
        """
        delta = max(-1.0, min(1.0, float(signal))) * self.effective_plasticity(
            direction=1.0 if signal >= 0 else -1.0) * self._base_step()
        before = self.value
        self.value += delta
        self.clamp()
        return self.value - before


    # 单次推动的基础步长。plasticity=1 时，一次也只推动 drift 的这么多。
    # 为什么需要它：plasticity 是「可塑性程度」，不是「一次推多远」。
    # 直接把 plasticity 当步长用，drift 小于 plasticity 时一次就撞墙，
    # 余弦退火完全失效（实测：两个不同人格都 5 轮饱和到 1.00）。
    # 取 0.15 意味着 plasticity=1 的参数要走约 7 次才用掉一半余量 ——
    # 这给了退火曲线展开的空间，也符合「人格改变是渐进的」。
    STEP_FRACTION = 0.15

    def _base_step(self) -> float:
        """基础步长 = drift × STEP_FRACTION。没有 drift 就没有推动。"""
        if self.drift <= 0:
            return 0.0
        return self.drift * self.STEP_FRACTION

    def effective_plasticity(self, *, direction: float = 1.0) -> float:
        """当前这一侧的实际可塑性（已退火）。

        direction: +1 往上限推，-1 往下限推。
        两侧的余量不同 —— 比如 anchor 靠近上限时，往上几乎推不动，
        往下还很自由。所以必须按方向分别算。
        """
        room_lo, room_hi = self.drift_room
        if direction >= 0:
            span = room_hi - self.anchor
            used = self.value - self.anchor
        else:
            span = self.anchor - room_lo
            used = self.anchor - self.value
        if span <= 1e-9:
            return 0.0
        saturation = max(0.0, min(1.0, used / span))
        return self.plasticity * 0.5 * (1.0 + math.cos(math.pi * saturation))

    @property
    def saturation(self) -> float:
        """离边界有多近。0 = 在 anchor，1 = 贴着某一侧边界。"""
        room_lo, room_hi = self.drift_room
        if self.value >= self.anchor:
            span = room_hi - self.anchor
            used = self.value - self.anchor
        else:
            span = self.anchor - room_lo
            used = self.anchor - self.value
        if span <= 1e-9:
            return 1.0
        return max(0.0, min(1.0, used / span))

    @property
    def drift_room(self) -> tuple[float, float]:
        lo = max(self.lo, self.anchor - self.drift)
        hi = min(self.hi, self.anchor + self.drift)
        if lo > hi:
            # 配置矛盾（物理边界与漂移范围无交集）—— 退化成「不能动」
            return (self.anchor, self.anchor)
        return (lo, hi)

    def as_dict(self) -> dict:
        return {"anchor": round(self.anchor, 3), "value": round(self.value, 3),
                "plasticity": self.plasticity, "drift": self.drift,
                "range": [round(x, 3) for x in self.drift_room],
                "saturation": round(self.saturation, 3)}

    def to_state(self) -> dict:
        """写成 state/traits.json 用的完整快照。"""
        return {"anchor": self.anchor, "value": self.value,
                "plasticity": self.plasticity, "drift": self.drift,
                "lo": self.lo, "hi": self.hi}

    @classmethod
    def from_state(cls, d: dict) -> "Trait":
        return cls(
            anchor=float(d.get("anchor", 0.0)),
            plasticity=float(d.get("plasticity", 0.0)),
            drift=float(d.get("drift", 0.0)),
            lo=float(d.get("lo", 0.0)),
            hi=float(d.get("hi", 1.0)),
            value=float(d.get("value", d.get("anchor", 0.0))),
        )


@dataclass
class Role:
    name: str = "默认"
    bias: float = 0.0
    attention: str = "subtle"
    legibility_max: int = 2
    voice: str = ""
    # 人格参数表。键是参数名，值是 Trait。
    # 空表时行为跟以前完全一样（只用 bias / attention / legibility）。
    traits: dict[str, Trait] = field(default_factory=dict)
    # 长文本：身份、性格、信念。原样进 prompt。
    identity: str = ""
    # 按场景注入的句式。键是场景名（greeting / comfort / promise / answer / farewell）
    phrases: dict[str, list[str]] = field(default_factory=dict)
    # 特殊触发的行为提示：[{when, note}]，由 need / 关键词触发注入
    special: list[dict] = field(default_factory=list)

    def traits_state(self) -> dict:
        """人格参数的当前快照 —— 写进 state/traits.json。"""
        return {k: v.to_state() for k, v in self.traits.items()}

    def load_traits_state(self, state: dict) -> None:
        """用持久化的状态覆盖当前参数。

        覆盖 **value 和 anchor**，不覆盖 plasticity / drift / lo / hi ——
        后四个是「人格的材质」，只能由配置决定，不该因为读到一个
        旧快照就被改掉。

        为什么 anchor 也要恢复：长期层会推动它。不恢复的话，
        重启一次就把跑了上百轮的基线漂移丢了 ——
        实测账本 anchor=0.94 重启后回到 0.70，0.24 的漂移蒸发。
        """
        for name, d in (state or {}).items():
            t = self.traits.get(name)
            if t is None:
                continue
            try:
                if "anchor" in d:
                    t.anchor = float(d["anchor"])
                t.value = float(d.get("value", t.value))
            except (TypeError, ValueError):
                continue
            t.clamp()

    def trait(self, name: str, default: float = 0.0) -> float:
        t = self.traits.get(name)
        return t.value if t else default

    def trait_rng(self, name: str, default: float = 0.0) -> float:
        """把某个参数映射成 0-1 的强度，给 prompt 用。"""
        return self.trait(name, default)


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

        confidence_floor 是硬地板：把握度低于它的，一律不许下结论。
        TOML 里一直写着这条，但代码从来没读过它 —— 所以它是个假旋钮。
        """
        eff = max(0.0, min(1.0, confidence + self.role.bias))
        stance = None
        for rule in self.stance_table:
            if rule.lo <= eff < rule.hi:
                stance = rule.stance
                break
        if stance is None:
            # 落进表里的空隙：取**覆盖该值两侧中更保守的一档**。
            # 旧实现直接取 table[-1]，但排序后最后一档是置信度最高那档
            # （最敢说的），正好反了 —— 拿不准的时候应该更保守。
            # 实测 rows=[[0,0.3,admit_unsure],[0.6,1.01,direct]] 时，
            # eff=0.4 落到 direct，而 0.3 以下都是 admit_unsure。
            conservative = [STANCE_UNSURE, STANCE_ASK, STANCE_GUESS, STANCE_DIRECT]
            cands = [r.stance for r in self.stance_table if r.hi <= eff]
            cands += [r.stance for r in self.stance_table if r.lo > eff]
            if cands:
                stance = min(cands, key=lambda w: (conservative.index(w)
                                                   if w in conservative else 99))
            else:
                stance = self.stance_table[-1].stance if self.stance_table else STANCE_UNSURE

        # 地板：低于它的不许直接下结论，退回「只问」。
        # 这是对站位表的保护 —— 换一套更敢说的表时，地板仍然拦得住。
        if self.confidence_floor > 0 and eff < self.confidence_floor \
                and stance in (STANCE_DIRECT, STANCE_GUESS):
            return STANCE_ASK
        return stance

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
            identity=str(r.get("identity", "") or ""),
            traits=_parse_traits(raw.get("params", {})),
            phrases=_parse_phrases(raw.get("phrases", {})),
            special=[{"when": str(x.get("when", "")), "note": str(x.get("note", ""))}
                     for x in (raw.get("special", []) or []) if isinstance(x, dict)],
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
            pending_total_max=int(lim.get("pending_total_max", 8)),
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
