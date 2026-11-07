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
    # 谁判的：model / heuristic。字面阈值诊断只能看 heuristic 的样本 ——
    # model 判的那些，字面 overlap 根本没参与决策。
    by: str = ""

    @property
    def rise(self) -> float:
        return self.formality - self.formality_base

    @property
    def topic_decided(self) -> bool:
        return self.topic in ("returned", "left")


@dataclass
class Calibration:
    samples: list[SettleSample] = field(default_factory=list)
    # 判定器自己的耗时。只有真的量到了才进来 —— 老账本里的 elapsed 是
    # 整轮耗时，拿它比 grace 会把「生成慢」误报成「判定超时」。
    judge_elapsed: list[float] = field(default_factory=list)
    judge_sources: dict[str, int] = field(default_factory=dict)
    judge_budgets: list[float] = field(default_factory=list)
    # 有多少轮账本只有旧的 elapsed 字段（无法区分判定耗时和生成耗时）
    legacy_elapsed: int = 0

    # 人格：参数从 anchor 漂了多远、基线动了几次
    trait_state: dict[str, dict] = field(default_factory=dict)
    anchor_updates: int = 0
    # 判断记忆的产出速率
    judgments: int = 0
    turns: int = 0
    diagnostic_judgments: int = 0

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
        """语义判定在**真的需要判定**的轮次里占多少。

        分母里不能有 source="none" —— 那是「这轮没有预言要判」，
        不是「判定失败」。算进去的话，覆盖率会被没有判定任务的轮次稀释。
        """
        active = {k: v for k, v in self.judge_sources.items() if k != "none"}
        total = sum(active.values())
        if not total:
            return None
        return active.get("llm", 0) / total


def collect(ledger: list[dict]) -> Calibration:
    """把账本摊平成一组可统计的样本。

    **样本的单位是「每条预言」，不是「每轮观测」。**
    同一轮可能对同一条预言判多次（挂起后每轮重判），只取最后一次 ——
    否则一条挂 17 轮的预言会贡献 17 个 WATCH 样本，把 OVERLAP 诊断
    带成「阈值偏高」的假警。

    三个人格相关的量顺带从账本重放出来：
    trait_state / anchor_updates / judgments / turns。
    它们回答的是另一个问题：「这个角色变了多少」。

    这也是为什么读这个模块时要注意：它的样本量和 turns 不是一回事。
    """
    from .store import is_diagnostic_judgment, replay_traits

    cal = Calibration()
    cal.trait_state = replay_traits(ledger)
    cal.anchor_updates = sum(1 for r in ledger if r.get("kind") == "anchor_update")
    records = [r for r in ledger if r.get("kind") == "judgment_add"]
    cal.diagnostic_judgments = sum(is_diagnostic_judgment(r) for r in records)
    cal.judgments = len(records) - cal.diagnostic_judgments
    cal.turns = sum(1 for r in ledger if r.get("kind") == "turn")

    # 同一轮可能对同一条预言判多次（挂起后每轮重判）。
    # samples 记「每条预言的最后一次判定」，不是「每轮的观测」——
    # 否则一条挂 17 轮的预言会贡献 17 个 WATCH 样本，
    # 把 OVERLAP_KEEP 诊断带成「阈值偏高」的假警。
    by_pid: dict[str, SettleSample] = {}

    for rec in ledger:
        if rec.get("kind") != "turn":
            continue
        probe = rec.get("probe") or {}

        j = probe.get("judge") or {}
        if j:
            src = j.get("source")
            if src:
                cal.judge_sources[src] = cal.judge_sources.get(src, 0) + 1
            # judge_elapsed 才是判定器自己的耗时。老账本只有 elapsed
            # （整轮耗时，绝大部分是主生成）—— 那种样本不能拿来比 grace，
            # 否则每条都「超时」，报告永远在报一个假警。
            if "judge_elapsed" in j and src != "none":
                # source="none" 表示这一轮没有判定任务，engine 写的是 0.0。
                # 把那些 0.0 算进分布会把真实的超时率稀释掉 ——
                # 实测 20 个 0.0 + 20 个 2.0 报成 50%，真实是 100%。
                el = j.get("judge_elapsed")
                if isinstance(el, (int, float)):
                    cal.judge_elapsed.append(float(el))
                    cal.judge_budgets.append(float(j.get("grace", GRACE)))
            elif isinstance(j.get("elapsed"), (int, float)):
                cal.legacy_elapsed += 1

        for row in probe.get("settle") or []:
            try:
                sample = SettleSample(
                    pid=str(row.get("pid", "")),
                    overlap=float(row.get("overlap", 0.0)),
                    formality=float(row.get("formality", 0.0)),
                    formality_base=float(row.get("formality_base", 0.0)),
                    length=int(row.get("length", 0)),
                    topic=row.get("topic"),
                    verdict=str(row.get("verdict", "")),
                    settle=bool(row.get("settle", False)),
                    by=str(row.get("by", "")),
                )
            except (TypeError, ValueError):
                continue
            key = sample.pid or f"_anon_{len(by_pid)}"
            # 后出现的覆盖先出现的 —— 拿最后一次判定
            by_pid[key] = sample

    cal.samples = list(by_pid.values())
    return cal


# ---------------------------------------------------------------- 诊断


@dataclass
class Finding:
    level: str      # ok / warn / info
    area: str
    detail: str


def findings(cal: Calibration) -> list[Finding]:
    """逐项诊断，每项给一个 level（ok / warn / info）。

    **level 的语义很重要**：
        ok    样本够、结论明确、没问题
        warn  样本够、结论明确、有问题
        info  样本不够，只能给现状，不能下结论

    为什么要有 info：2 条样本算出「100% 超时」是噪声不是诊断。
    所有分支都有最小样本量门槛，不够就降级成 info ——
    这是为了防止报告看起来很有信息量，实际全是噪声。

    诊断项（按顺序）：
        OVERLAP_KEEP / OVERLAP_DROP  字面阈值该往哪调
        语域阈值                      受伤检测的升幅门槛
        grace                         判定器的超时容差
        语义判定                       LLM 判定的覆盖率
        人格漂移                      参数离起点走了 drift 的百分之多少
        判断产出                      中期层有没有足够的证据
    """
    out: list[Finding] = []

    # 没有独立人工标签，不能用判定器自己的结果验证自己的阈值。
    if cal.samples:
        out.append(Finding(
            "info", "字面相关性",
            "重合度仅作观测，不用于证实/否证读法，也不据此推荐降低结算阈值"))
        out.append(Finding(
            "info", "语域阈值",
            "疑似受伤是启发式观测；需独立用户反馈/人工标注后才能验证阈值"))

    # ---- 3. grace 预算
    if len(cal.judge_elapsed) < 10:
        if cal.judge_elapsed:
            out.append(Finding(
                "info", "grace",
                f"只有 {len(cal.judge_elapsed)} 条判定耗时样本 —— 至少 10 条才下结论"))
    elif cal.judge_elapsed:
        el = sorted(cal.judge_elapsed)
        # Availability at collection, not total background duration, determines a timeout.
        active = sum(v for k, v in cal.judge_sources.items() if k != "none")
        over = cal.judge_sources.get("late", 0)
        if not active:
            budgets = cal.judge_budgets or [GRACE] * len(cal.judge_elapsed)
            over = sum(x > budget for x, budget in zip(cal.judge_elapsed, budgets))
        budget_label = "各轮记录的预算" if cal.judge_budgets else f"{GRACE}s"
        rate = over / (active or len(el))
        if rate > 0.3:
            out.append(Finding(
                "warn", "grace",
                f"{rate:.0%} 的轮次收集时判定仍未返回（{budget_label}；中位 "
                f"{cal.pct(el, 0.5):.2f}s）—— 要么放宽 grace，要么换更快的模型"
            ))
        else:
            out.append(Finding(
                "ok", "grace",
                f"中位 {cal.pct(el, 0.5):.2f}s，超时率 {rate:.0%}"
            ))
    elif cal.legacy_elapsed:
        out.append(Finding(
            "info", "grace",
            f"{cal.legacy_elapsed} 轮只有旧的 elapsed（整轮耗时，含生成），"
            f"量不出判定本身的花费 —— 再跑一轮就有准确数据了"
        ))

    # ---- 4. 语义判定是否真的在起作用
    rate = cal.judge_success_rate()
    if rate is not None:
        if rate < 0.5:
            out.append(Finding(
                "warn", "语义判定",
                f"只有 {rate:.0%} 的轮次用上了语义判定，其余无法取得语义判定 —— "
                f"未明确的假设继续保留；覆盖率不代表判定准确度"
            ))
        else:
            out.append(Finding(
                "ok", "语义判定", f"{rate:.0%} 的轮次用上了语义判定"
            ))

    # ---- 5. 人格漂移
    #
    # 「这个角色有没有变成另一个人」不能只看数值，要看它离自己的起点有多远。
    # 用 drift 当分母 —— 因为 drift 就是「这个人格允许自己走多远」。
    if cal.trait_state:
        moved = []
        for name, spec in cal.trait_state.items():
            drift = float(spec.get("drift", 0.0))
            if drift <= 0:
                continue
            offset = abs(float(spec.get("value", 0.0)) - float(spec.get("anchor", 0.0)))
            if offset > 1e-9:
                moved.append((name, offset / drift, offset))
        if moved:
            moved.sort(key=lambda x: -x[1])
            worst = moved[0]
            detail = (f"{len(moved)} 个参数偏离起点，最远的 {worst[0]} "
                      f"走了 {worst[1]:.0%}（{worst[2]:+.3f}）")
            level = "warn" if worst[1] > 0.8 else "ok"
            if worst[1] > 0.8:
                detail += " —— 快贴到漂移边界了"
            out.append(Finding(level, "人格漂移", detail))
        else:
            out.append(Finding("ok", "人格漂移", "所有参数都还在起点"))
        if cal.anchor_updates:
            out.append(Finding("info", "基线变动",
                               f"{cal.anchor_updates} 次 —— 人格的材质被改过"))

    # 判断数量不是陪伴质量，也不作为人格自动成长的依据。
    if cal.turns >= 10:
        rate = cal.judgments / cal.turns
        out.append(Finding("info", "判断产出",
                           f"{cal.judgments} 条 / {cal.turns} 轮（{rate:.2f}/轮）；不设产出配额"))
    if cal.diagnostic_judgments:
        out.append(Finding("info", "读法诊断", f"{cal.diagnostic_judgments} 条模型读法计数；不计用户判断产出"))

    if not cal.samples:
        out.append(Finding(
            "info", "样本", "还没有结算样本。先聊几轮，等预言被结算了再来看"
        ))

    return out


def format_report(cal: Calibration) -> str:
    """把诊断渲染成终端文本。

    分成四段：概况 / 诊断 / 原始分布 / 分布图。
    第一段是「有多少样本」，因为后面所有结论都受它限制 ——
    读者应该先看到样本量再看到结论，而不是反过来。
    """
    lines = ["", "校准报告", "─" * 60, "  模型判定不是独立真值；本报告不证明参数有效、人格成长或用户认可度提升。"]

    n_turn = cal.turns or sum(cal.judge_sources.values())
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
