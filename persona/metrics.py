"""从账本算指标。

README 里那张评测表如果不落到代码上，它就只是文档。
这里全部从 ledger.jsonl 重放计算 —— 不改账本，只读。

四个指标：
    位移率       direct ≠ say 的比例。太低客服，太高谜语人。
    校准         confidence 和实际命中率的偏差。差值 > 0.2 = 在读空气。
    声明违规     输出里出现「我本来想 / 没说出口」。出现即失败。
    关系增益     有无历史时的命中率差。差值为零 = 记忆是摆设。
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
    expired: int = 0

    repair_opened: int = 0
    repair_attempted: int = 0
    repair_healed: int = 0
    repair_rejected: int = 0
    repair_residual: int = 0

    conf_pairs: list[tuple[float, bool]] = field(default_factory=list)
    displace_kinds: dict[str, int] = field(default_factory=dict)
    # 被挂起过的预言 id（去重）。同一条预言可能被晾好几轮，只算一次。
    watched_ids: set[str] = field(default_factory=set)

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
        return self.displaced / self.turns if self.turns else 0.0

    @property
    def accuracy(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    @property
    def repair_success_rate(self) -> float | None:
        """修复成功率。没发起过就返回 None —— 0 和「没有」是两回事。"""
        done = self.repair_healed + self.repair_rejected + self.repair_residual
        if not done:
            return None
        return self.repair_healed / done

    @property
    def brier(self) -> float:
        """Brier 分数：越低越好。0.25 是「永远说 0.5」的基准。"""
        if not self.conf_pairs:
            return 0.0
        return sum((c - (1.0 if h else 0.0)) ** 2 for c, h in self.conf_pairs) \
            / len(self.conf_pairs)

    def calibration_gap(self) -> float:
        """预测的把握度和实际命中率差多少。

        只在样本够的时候才有意义。样本少于 4 条时返回 0 ——
        两条数据算出「偏差 0.35」是噪声，不是诊断。
        """
        if len(self.conf_pairs) < 4:
            return 0.0
        mean_conf = sum(c for c, _ in self.conf_pairs) / len(self.conf_pairs)
        return abs(mean_conf - self.accuracy)

    def is_overconfident(self) -> bool:
        """是不是「嘴上很有把握，实际全错」。

        只有把握高、命中低才算。反过来（把握低命中高）是保守，
        不是毛病，不该报警。
        """
        if len(self.conf_pairs) < 4:
            return False
        mean_conf = sum(c for c, _ in self.conf_pairs) / len(self.conf_pairs)
        return mean_conf - self.accuracy > 0.2

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
            direct = rec.get("direct") or ""
            disp = rec.get("displace") or "none"

            if disp not in ("", "none"):
                m.displaced += 1
                m.displace_kinds[disp] = m.displace_kinds.get(disp, 0) + 1
                # 声称位移但 direct 和 say 几乎一样 = 装深沉。
                # direct 为空也要算 —— 没有原话就没有位移可言。
                if not direct.strip() or _similar(direct, say):
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
            m.expired += 1

        elif kind == "pending_settle":
            outcome = rec.get("outcome", "")
            if outcome in HIT:
                m.hits += 1
            elif outcome in MISS:
                m.misses += 1
            elif outcome == "SUSPECTED_HURT":
                m.hurts += 1
            elif outcome == "EXPIRED":
                m.expired += 1
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
    """关系增益：清空历史前后，命中率差多少。

    差值为零说明记忆是摆设 —— 这是唯一诚实的「记忆有用吗」的度量。

    注意样本量：这个差值在两条样本上毫无意义。所以一起把两边的样本数
    带出来，让读的人自己判断能不能信。
    """
    n_full = full.hits + full.misses
    n_bare = bare.hits + bare.misses
    return {
        "with_history": round(full.accuracy, 3),
        "without_history": round(bare.accuracy, 3),
        "lift": round(full.accuracy - bare.accuracy, 3),
        "n_with": n_full,
        "n_without": n_bare,
        "reliable": n_full >= 10 and n_bare >= 10,
    }


def final_bias(ledger: list[dict], initial: float = 0.0) -> float:
    """账本里的 bias 轨迹末端。

    同一个账本配不同初值，可以对比「同一段经历、不同起点」会长成什么样 ——
    这正是账本优先想支持的实验。
    """
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

    if m.turns:
        rate = m.displacement_rate
        if rate < 0.05:
            out.append(("warn", "位移率", f"{rate:.0%} —— 几乎不位移，像客服"))
        elif rate > 0.6:
            out.append(("warn", "位移率", f"{rate:.0%} —— 每句都留白，是谜语人"))
        else:
            out.append(("ok", "位移率", f"{rate:.0%}"))

    if m.declarations:
        out.append(("fail", "声明违规",
                    f"{m.declarations} 次把位移说出来了 —— 出现即失败"))
    else:
        out.append(("ok", "声明违规", "无"))

    if m.inconsistent:
        out.append(("warn", "装深沉",
                    f"{m.inconsistent} 次声称位移但 direct≈say"))

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
