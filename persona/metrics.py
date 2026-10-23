"""只读账本的描述性指标。

位移率只计明确标记 persona_baseline 且有基准的轮次，旧口径和缺失基准另列未知。
无额外位移是常态，不用位移率推断有没有人格。字面相似检查不是语义证明。
校准只用有配对的预测样本；模型结算不是独立真值，更不等于用户认可度。
首见/重复的命中差异不能证明记忆的因果收益。人格变化只报告记录，不等同成长。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import _similar

HIT = {"CONFIRMED", "CONFIRMED'"}
MISS = {"REFUTED", "REFUTED'"}

_DECL = re.compile(
    r"(我本来想|我原本想|本来想说|我没说出口|我差点说|我差点儿说|"
    r"其实我刚才想|我克制住|我忍住没|我欲言又止)"
)


@dataclass
class Metrics:
    turns: int = 0
    displaced: int = 0
    declarations: int = 0
    inconsistent: int = 0

    hits: int = 0
    misses: int = 0
    watches: int = 0
    hurts: int = 0
    # 解析失败（没有 displace 字段）的轮次 —— 不进位移率分母
    unparsed: int = 0
    unknown_basis: int = 0  # 旧口径或缺人格基准，不能混入新位移率

    repair_opened: int = 0
    repair_attempted: int = 0
    repair_healed: int = 0
    repair_rejected: int = 0
    repair_residual: int = 0

    conf_pairs: list[tuple[float, bool]] = field(default_factory=list)
    displace_kinds: dict[str, int] = field(default_factory=dict)
    # 被挂起过的预言 id（去重）。同一条预言可能被晾好几轮，只算一次。
    watched_ids: set[str] = field(default_factory=set)
    # 过期的预言 id（去重）。同一条可能同时有 pending_expire 和
    # pending_settle(EXPIRED) 两条记录。
    expired_ids: set[str] = field(default_factory=set)
    # 没有 pending_id 的过期记录（老数据 / 手工改动）—— 无法去重，单独计数。
    expired_anon: int = 0

    # 人格：从 traits_init 起，累计每次 trait_update / anchor_update 的位移。
    # 两个 dict 分开存，不用前缀挤一个 —— 参数名可能含冒号，前缀法就分不开。
    trait_drift: dict[str, float] = field(default_factory=dict)
    anchor_drift: dict[str, float] = field(default_factory=dict)
    trait_updates: int = 0
    anchor_updates: int = 0
    # 判断：写了几条、推翻了几条
    judgments: int = 0
    judgments_superseded: int = 0

    # ---------------------------------------------------------- 派生

    @property
    def displacement_rate(self) -> float:
        """位移率。

        分母只算**成功解析出 displace 的轮次** —— 解析失败（displace 为空）
        的那些轮不是「没有位移」，是「不知道」，算进分母会拉低比率。
        """
        known = self.turns - self.unparsed - self.unknown_basis
        return self.displaced / known if known > 0 else 0.0

    @property
    def expired(self) -> int:
        """过期总数 = 去重后的 id 数 + 无法去重的匿名记录数。"""
        return len(self.expired_ids) + self.expired_anon

    @property
    def accuracy(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    @property
    def repair_success_rate(self) -> float | None:
        """修复成功率。没发起过就返回 None —— 0 和「没有」是两回事。

        分母包含「开了但没人理」的修复。只算已结案的话，
        10 次受伤里 9 次没回应、1 次治愈会报成「成功率 100%」——
        那 9 次恰恰是最该被看见的。
        """
        # 分母直接用「开了几条修复」。
        #
        # 用已经结案的数当分母会漏掉最该被看见的那批：
        # 10 次受伤、9 次对方完全没接、1 次治愈 ——
        # 旧算法报 100%，真实的修复覆盖率是 10%。
        total = self.repair_opened
        if not total:
            return None
        return self.repair_healed / total

    @property
    def brier(self) -> float:
        """Brier 分数：越低越好。0.25 是「永远说 0.5」的基准。"""
        if not self.conf_pairs:
            return 0.0
        return sum((c - (1.0 if h else 0.0)) ** 2 for c, h in self.conf_pairs) \
            / len(self.conf_pairs)

    def paired_accuracy(self) -> float:
        """**只在有配对样本上算**的命中率。

        为什么要单独的这一个：calibration_gap 拿 conf_pairs 的均值
        去比全局 accuracy，两边的分母不一样。
        实测构造：4 条有配对全中 + 6 条无配对全错
            -> 全局 accuracy = 40%
            -> conf_pairs 均值 = 0.90
            -> 报「平均把握 0.90 但命中 40%，在读空气」
        但这 4 条其实全中 —— 纯属分母错配，把好数据报成坏数据。
        """
        if not self.conf_pairs:
            return 0.0
        return sum(1 for _, h in self.conf_pairs if h) / len(self.conf_pairs)

    def calibration_gap(self) -> float:
        """预测的把握度和实际命中率差多少。

        两边都用 conf_pairs —— 有配对的那批才是校准样本。
        样本少于 4 条时返回 0：两条数据算出「偏差 0.35」是噪声。
        """
        if len(self.conf_pairs) < 4:
            return 0.0
        mean_conf = sum(c for c, _ in self.conf_pairs) / len(self.conf_pairs)
        return abs(mean_conf - self.paired_accuracy())

    def is_overconfident(self) -> bool:
        """是不是「嘴上很有把握，实际全错」。

        只有把握高、命中低才算。反过来（把握低命中高）是保守，
        不是毛病，不该报警。分母同样用 conf_pairs。
        """
        if len(self.conf_pairs) < 4:
            return False
        mean_conf = sum(c for c, _ in self.conf_pairs) / len(self.conf_pairs)
        return mean_conf - self.paired_accuracy() > 0.2

    def buckets(self) -> list[tuple[str, int, float, float]]:
        """按把握度分桶，看每个桶的实际命中率。"""
        edges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        out = []
        for lo, hi in edges:
            rows = [(c, h) for c, h in self.conf_pairs if lo <= c < hi]
            if not rows:
                continue
            avg_conf = sum(c for c, _ in rows) / len(rows)
            hit = sum(1 for _, h in rows if h) / len(rows)
            out.append((f"{lo:.1f}–{min(hi, 1.0):.1f}", len(rows), avg_conf, hit))
        return out


def compute(ledger: list[dict]) -> Metrics:
    m = Metrics()
    for rec in ledger:
        kind = rec.get("kind")

        if kind == "turn":
            m.turns += 1
            say = rec.get("say") or ""
            baseline = rec.get("baseline") or ""
            # 不能写 `or "none"` —— 那会把解析失败（空串）和
            # 「明确没位移」（none）混成一个，unparsed 永远是 0。
            disp = rec.get("displace")

            if disp is None or disp == "":
                # 解析失败：不知道有没有位移，不进分母
                m.unparsed += 1
            elif rec.get("displacement_basis") != "persona_baseline" or not baseline.strip():
                m.unknown_basis += 1
            elif disp != "none":
                m.displaced += 1
                m.displace_kinds[disp] = m.displace_kinds.get(disp, 0) + 1
                # 声称位移但 direct 和 say 几乎一样 = 装深沉。
                # direct 为空也要算 —— 没有原话就没有位移可言。
                if _similar(baseline, say):
                    m.inconsistent += 1

            if _DECL.search(say):
                m.declarations += 1

            # WATCH 按定义就是「这轮不记账」，所以它永远不会出现在
            # pending_settle 里。想数挂起，只能看每一轮留下的观测值 ——
            # 但同一条预言可能被晾好几轮，必须按 pid 去重，
            # 否则「挂起中 28」这种数字看着像有 28 条预言，实际只有 3 条。
            for row in (rec.get("probe") or {}).get("settle") or []:
                if row.get("settle"):
                    continue
                pid = row.get("pid")
                if pid:
                    m.watched_ids.add(str(pid))
                else:
                    m.watches += 1

        elif kind == "pending_expire":
            # 过期的预言走的是 pending_expire，不是 pending_settle。
            # 只认 pending_settle 的话，EXPIRED 永远是 0。
            #
            # 但**同一条预言可能两种记录都有**（老引擎两条都写），
            # 直接 ++ 会算成两次。按 pid 去重。
            pid = rec.get("pending_id")
            if pid:
                m.expired_ids.add(str(pid))
            else:
                m.expired_anon += 1

        elif kind == "pending_settle":
            outcome = rec.get("outcome", "")
            if outcome in HIT:
                m.hits += 1
            elif outcome in MISS:
                m.misses += 1
            elif outcome == "SUSPECTED_HURT":
                m.hurts += 1
            elif outcome == "EXPIRED":
                pid = rec.get("pending_id")
                if pid:
                    m.expired_ids.add(str(pid))
                else:
                    m.expired_anon += 1
            else:
                m.watches += 1

        elif kind == "judgment_add":
            m.judgments += 1
            if rec.get("supersedes"):
                m.judgments_superseded += 1

        elif kind == "trait_update":
            m.trait_updates += 1
            name = rec.get("name", "")
            m.trait_drift[name] = m.trait_drift.get(name, 0.0) + float(rec.get("delta", 0.0))

        elif kind == "anchor_update":
            m.anchor_updates += 1
            name = rec.get("name", "")
            # 基线漂移单独记 —— 它是「我变成了谁」，和参数浮动不是一回事
            m.anchor_drift[name] = m.anchor_drift.get(name, 0.0) \
                + float(rec.get("delta", 0.0))

        elif kind == "relation_observation" and rec.get("suspected_hurt"):
            m.hurts += 1
        elif kind == "repair_open":
            m.repair_opened += 1
        elif kind == "repair_attempt":
            m.repair_attempted += 1
        elif kind == "repair_healed":
            m.repair_healed += 1
        elif kind == "repair_rejected":
            m.repair_rejected += 1
        elif kind == "repair_residual":
            m.repair_residual += 1

    # 校准配对：把 pending_settle 和当初那条 pending_add 的 confidence 对上
    conf_by_id = {
        r["pending_id"]: float(r.get("confidence", 0.0))
        for r in ledger if r.get("kind") == "pending_add"
    }
    for rec in ledger:
        if rec.get("kind") != "pending_settle":
            continue
        c = conf_by_id.get(rec.get("pending_id"))
        if c is None:
            continue
        outcome = rec.get("outcome", "")
        if outcome in HIT:
            m.conf_pairs.append((c, True))
        elif outcome in MISS:
            m.conf_pairs.append((c, False))

    return m


def compare_with_without_history(full: Metrics, bare: Metrics) -> dict:
    """两组命中率的描述性差异；本函数不能判定记忆的因果效果。"""
    n_full = full.hits + full.misses
    n_bare = bare.hits + bare.misses
    return {
        "with_history": round(full.accuracy, 3),
        "without_history": round(bare.accuracy, 3),
        "lift": round(full.accuracy - bare.accuracy, 3),
        "n_with": n_full,
        "n_without": n_bare,
        "reliable": n_full >= 10 and n_bare >= 10,
        "causal": False,
    }


def final_bias(ledger: list[dict], initial: float = 0.0) -> float:
    """历史记录的 bias 末值，不按新初值重新计算旧决策。"""
    bias = initial
    for rec in ledger:
        if rec.get("kind") != "bias_update":
            continue
        try:
            bias = float(rec["after"])
        except (KeyError, TypeError, ValueError):
            continue
    return bias


def verdicts(m: Metrics) -> list[tuple[str, str, str]]:
    """把指标翻译成诊断。返回 (等级, 项目, 说明)。"""
    out: list[tuple[str, str, str]] = []

    if m.turns - m.unparsed - m.unknown_basis:
        rate = m.displacement_rate
        out.append(("info", "位移率", f"{rate:.0%}（相对人格基准，模型自报；低不等于无人格）"))
    if m.unknown_basis:
        out.append(("info", "口径未知", f"{m.unknown_basis} 轮旧口径/缺人格基准，不计新位移率"))

    if m.declarations:
        out.append(("fail", "声明违规",
                    f"{m.declarations} 次把位移说出来了 —— 出现即失败"))
    else:
        out.append(("ok", "声明违规", "无"))

    if m.inconsistent:
        out.append(("warn", "位移待核对",
                    f"{m.inconsistent} 次声称位移但 baseline≈say（字面诊断，不是语义定论）"))

    if m.conf_pairs:
        gap = m.calibration_gap()
        if len(m.conf_pairs) < 4:
            out.append(("info", "校准",
                        f"样本只有 {len(m.conf_pairs)} 条，还算不准"))
        elif m.is_overconfident():
            out.append(("warn", "校准",
                        f"平均把握 {sum(c for c, _ in m.conf_pairs) / len(m.conf_pairs):.2f}"
                        f" 但命中 {m.accuracy:.0%} —— 在读空气"))
        else:
            out.append(("ok", "校准",
                        f"偏差 {gap:.2f}，Brier {m.brier:.3f}"))

    if m.hits + m.misses:
        out.append(("info", "命中率", f"{m.accuracy:.0%}"
                                     f"（{m.hits}/{m.hits + m.misses}）"))
    total_watched = m.watches + len(m.watched_ids)
    if total_watched:
        note = "（按预言 id 去重）" if m.watched_ids else ""
        out.append(("info", "挂起过", f"{total_watched} 条{note}"))
    if m.expired:
        out.append(("info", "已过期", str(m.expired)))

    if m.repair_opened:
        rate = m.repair_success_rate
        detail = (f"{m.repair_opened} 次受伤，"
                  f"{m.repair_attempted} 次修复")
        if rate is not None:
            detail += f"，成功率 {rate:.0%}"
        level = "ok" if (rate is None or rate >= 0.5) else "warn"
        out.append((level, "受伤修复", detail))
        if m.repair_rejected:
            out.append(("info", "修复被拒",
                        f"{m.repair_rejected} 次 —— 已转雷区"))
        if m.repair_residual:
            out.append(("warn", "修复无回应",
                        f"{m.repair_residual} 次 —— 可能没被听见"))

    if m.judgments:
        detail = f"{m.judgments} 条"
        if m.judgments_superseded:
            detail += f"（{m.judgments_superseded} 条推翻了旧的）"
        out.append(("info", "判断记忆", detail))

    if m.trait_updates or m.anchor_updates:
        out.append(("info", "人格漂移",
                    f"参数推动 {m.trait_updates} 次，基线推动 {m.anchor_updates} 次"))

    return out


def format_report(m: Metrics) -> str:
    lines = ["", "指标", "─" * 54]
    icon = {"ok": "✓", "warn": "!", "fail": "×", "info": "·"}
    for level, name, detail in verdicts(m):
        lines.append(f"  {icon.get(level, '·')} {name:<8}{detail}")

    if m.displace_kinds:
        lines += ["", "位移类型分布", "─" * 54]
        total = sum(m.displace_kinds.values()) or 1
        for k, v in sorted(m.displace_kinds.items(), key=lambda x: -x[1]):
            lines.append(f"  {k:<6}{v:>4}  {v / total:>5.0%}")

    buckets = m.buckets()
    if buckets:
        lines += ["", "校准分桶", "─" * 54]
        lines.append(f"  {'把握度':<10}{'样本':>6}{'平均把握':>10}{'实际命中':>10}")
        for label, n, conf, hit in buckets:
            flag = "  ← 虚高" if conf - hit > 0.2 else ""
            lines.append(f"  {label:<10}{n:>6}{conf:>10.2f}{hit:>10.2f}{flag}")

    if m.trait_drift or m.anchor_drift:
        lines += ["", "人格漂移", "─" * 54]
        lines.append(f"  {'参数':<16}{'类型':>6}{'累计变化':>12}")
        rows = [(k, "参数", v) for k, v in m.trait_drift.items()]
        rows += [(k, "基线", v) for k, v in m.anchor_drift.items()]
        for name, kind, v in sorted(rows, key=lambda x: -abs(x[2])):
            lines.append(f"  {name:<14}{kind:>6}{v:>+12.4f}")

    return "\n".join(lines)
