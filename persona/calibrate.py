"""从账本反推阈值该往哪调。

为什么需要它：`OVERLAP_KEEP=0.18`、受伤的语域阈值 `0.3`、`grace=0.35`
这些数字全是估的。聊三十轮之后，光看「发生了什么」不知道偏在哪 ——
必须看每次判定时那些原始数值分布在哪。

这个模块只读账本，不改任何东西。它输出的是**证据**，不是结论：
告诉你「被挂起的那些，字面重合度都落在哪」，然后由人决定阈值怎么动。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 当前的阈值，跟 settle.py / config 保持一致。
# 这里写死是为了让报告能直接标出「这条落在阈值哪一侧」。
OVERLAP_KEEP = 0.18
OVERLAP_DROP = 0.05
FORMALITY_RISE = 0.3
GRACE = 0.35


@dataclass
class SettleSample:
    """一次结算判定时的全部原始数值。"""

    pid: str
    overlap: float
    formality: float
    formality_base: float
    length: int
    topic: str | None
    verdict: str
    settle: bool

    @property
    def rise(self) -> float:
        return self.formality - self.formality_base

    @property
    def topic_decided(self) -> bool:
        return self.topic in ("returned", "left")


@dataclass
class Calibration:
    samples: list[SettleSample] = field(default_factory=list)
    judge_elapsed: list[float] = field(default_factory=list)
    judge_sources: dict[str, int] = field(default_factory=dict)

    # ---------------------------------------------------------- 统计

    def by_verdict(self, kind: str) -> list[SettleSample]:
        return [s for s in self.samples if s.verdict == kind]

    def overlap_dist(self, kind: str) -> list[float]:
        return sorted(s.overlap for s in self.by_verdict(kind))

    def pct(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        i = max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
        return values[i]

    def judge_success_rate(self) -> float | None:
        total = sum(self.judge_sources.values())
        if not total:
            return None
        return self.judge_sources.get("llm", 0) / total


def collect(ledger: list[dict]) -> Calibration:
    cal = Calibration()
    for rec in ledger:
        if rec.get("kind") != "turn":
            continue
        probe = rec.get("probe") or {}

        j = probe.get("judge") or {}
        if j:
            src = j.get("source")
            if src:
                cal.judge_sources[src] = cal.judge_sources.get(src, 0) + 1
            el = j.get("elapsed")
            if isinstance(el, (int, float)):
                cal.judge_elapsed.append(float(el))

        for row in probe.get("settle") or []:
            try:
                cal.samples.append(SettleSample(
                    pid=str(row.get("pid", "")),
                    overlap=float(row.get("overlap", 0.0)),
                    formality=float(row.get("formality", 0.0)),
                    formality_base=float(row.get("formality_base", 0.0)),
                    length=int(row.get("length", 0)),
                    topic=row.get("topic"),
                    verdict=str(row.get("verdict", "")),
                    settle=bool(row.get("settle", False)),
                ))
            except (TypeError, ValueError):
                continue
    return cal


# ---------------------------------------------------------------- 诊断


@dataclass
class Finding:
    level: str      # ok / warn / info
    area: str
    detail: str


def findings(cal: Calibration) -> list[Finding]:
    out: list[Finding] = []

    # ---- 1. 字面重合度阈值
    settled = [s for s in cal.samples if s.settle and s.verdict == "CONFIRMED"]
    watched = [s for s in cal.samples if s.verdict == "WATCH"]

    if len(cal.samples) >= 5:
        w = sorted(s.overlap for s in watched)
        if w:
            hi = cal.pct(w, 0.9)
            if hi >= OVERLAP_KEEP:
                out.append(Finding(
                    "warn", "OVERLAP_KEEP",
                    f"被挂起的样本里有 {sum(1 for x in w if x >= OVERLAP_KEEP)} 条"
                    f"字面重合 ≥ {OVERLAP_KEEP}（最高 {hi:.2f}）—— "
                    f"阈值可能偏高，把该结算的判成挂起了"
                ))
            elif hi < OVERLAP_DROP:
                out.append(Finding(
                    "ok", "OVERLAP_KEEP",
                    f"挂起的样本字面重合都在 {hi:.2f} 以下，阈值分得开"
                ))

        s_over = sorted(s.overlap for s in settled)
        if s_over and cal.pct(s_over, 0.1) < OVERLAP_DROP:
            out.append(Finding(
                "warn", "OVERLAP_DROP",
                f"已结算的样本里最低重合只有 {cal.pct(s_over, 0.1):.2f}，"
                f"低于 DROP 阈值 —— 说明这些是靠语义判定结的，字面阈值在拖后腿"
            ))

    # ---- 2. 受伤检测的语域阈值
    hurt = cal.by_verdict("SUSPECTED_HURT")
    normal = [s for s in cal.samples
              if s.verdict not in ("SUSPECTED_HURT",) and s.settle]
    if hurt and normal:
        h_rise = min(s.rise for s in hurt)
        n_rise = max(s.rise for s in normal)
        if n_rise >= h_rise:
            out.append(Finding(
                "warn", "语域阈值",
                f"判为受伤的最小升幅 {h_rise:.2f}，"
                f"但正常结算里最大升幅有 {n_rise:.2f} —— 两者重叠，会误判"
            ))
        else:
            gap = h_rise - n_rise
            out.append(Finding(
                "ok" if gap > 0.1 else "warn", "语域阈值",
                f"受伤最小升幅 {h_rise:.2f} vs 正常最大 {n_rise:.2f}，"
                f"间隔 {gap:.2f}"
                + ("（偏窄，样本再多可能就撞上了）" if gap <= 0.1 else "")
            ))
    elif hurt:
        out.append(Finding(
            "info", "语域阈值",
            f"有 {len(hurt)} 次受伤判定，但没有可对照的正常样本"
        ))

    # ---- 3. grace 预算
    if cal.judge_elapsed:
        el = sorted(cal.judge_elapsed)
        over = sum(1 for x in el if x > GRACE)
        rate = over / len(el)
        if rate > 0.3:
            out.append(Finding(
                "warn", "grace",
                f"{rate:.0%} 的轮次判定没在 {GRACE}s 内返回（中位 "
                f"{cal.pct(el, 0.5):.2f}s）—— 要么放宽 grace，要么换更快的模型"
            ))
        else:
            out.append(Finding(
                "ok", "grace",
                f"中位 {cal.pct(el, 0.5):.2f}s，超时率 {rate:.0%}"
            ))

    # ---- 4. 语义判定是否真的在起作用
    rate = cal.judge_success_rate()
    if rate is not None:
        if rate < 0.5:
            out.append(Finding(
                "warn", "语义判定",
                f"只有 {rate:.0%} 的轮次用上了语义判定，其余都是字面兜底 —— "
                f"接 LLM 的意义被抵消了大半"
            ))
        else:
            out.append(Finding(
                "ok", "语义判定", f"{rate:.0%} 的轮次用上了语义判定"
            ))

    if not cal.samples:
        out.append(Finding(
            "info", "样本", "还没有结算样本。先聊几轮，等预言被结算了再来看"
        ))

    return out


def format_report(cal: Calibration) -> str:
    lines = ["", "校准报告", "─" * 60]

    n_turn = len(cal.judge_elapsed) or sum(cal.judge_sources.values())
    lines.append(f"  样本：{len(cal.samples)} 次结算判定，{n_turn} 轮有判定记录")

    if cal.samples:
        lines += ["", "按判定结果分组", "─" * 60]
        lines.append(f"  {'判定':<18}{'数量':>6}{'字面重合中位':>14}{'语域升幅中位':>14}")
        kinds = sorted({s.verdict for s in cal.samples})
        for k in kinds:
            rows = cal.by_verdict(k)
            ov = sorted(s.overlap for s in rows)
            ri = sorted(s.rise for s in rows)
            lines.append(
                f"  {k:<18}{len(rows):>6}{cal.pct(ov, 0.5):>14.2f}{cal.pct(ri, 0.5):>14.2f}"
            )

    lines += ["", "诊断", "─" * 60]
    icon = {"ok": "✓", "warn": "!", "info": "·"}
    for f in findings(cal):
        lines.append(f"  {icon.get(f.level, '·')} {f.area:<14}{f.detail}")

    if cal.samples:
        lines += ["", "原始分布（字面重合度）", "─" * 60]
        for k in sorted({s.verdict for s in cal.samples}):
            vals = cal.overlap_dist(k)
            if not vals:
                continue
            bars = "".join(
                "█" if v >= OVERLAP_KEEP else ("▒" if v >= OVERLAP_DROP else "░")
                for v in vals
            )
            lines.append(f"  {k:<18}{bars}")
        lines.append(f"  {'':<18}░ <{OVERLAP_DROP}  ▒ {OVERLAP_DROP}-{OVERLAP_KEEP}  █ ≥{OVERLAP_KEEP}")

    return "\n".join(lines)
