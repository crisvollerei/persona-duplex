"""结算：把用户下一轮的实际反应，判成对上一轮预言的结果。

这里是「不需要裁判」的落点：ground truth 不在这一轮里，
而在用户下一句里。所以判不了就挂起，不硬猜。

「不接话题」有三种心理含义，当轮不可区分：
    ① 只是不关注        → 无信息
    ② 回避（碰到沉重记忆）→ 有信息，但现在不能说
    ③ 被冒犯但选择忍耐   → 关系已受损，且没有说出口

所以不能塞进 confirmed / refuted —— 那测的不是读得准不准，
是这段关系有没有被碰伤。

判定优先级（重要）：
    1. 模型自己在 TAIL 里给出的 settle（它读得懂语义，这是主路径）
    2. 语域突变 → SUSPECTED_HURT（关系信号，优先于校准）
    3. 话题延续度 → 能判就判
    4. 都不确定 → WATCH，不记账
"""
from __future__ import annotations

import re
from dataclasses import dataclass

WATCH = "WATCH"
CONFIRMED = "CONFIRMED"
REFUTED = "REFUTED"
CONFIRMED_DELAYED = "CONFIRMED'"
REFUTED_DELAYED = "REFUTED'"
SUSPECTED_HURT = "SUSPECTED_HURT"
EXPIRED = "EXPIRED"

# 话题延续度的判定阈值。偏保守：宁可挂起，也不要污染校准数据。
OVERLAP_KEEP = 0.18
OVERLAP_DROP = 0.05

_FILLERS = "哈哈嘿嘿呃嗯嘛呀啦～~"


# ---------------------------------------------------------------- 文本指标


def _norm_len(text: str) -> int:
    return len([c for c in text if not c.isspace()])


def _bigrams(text: str) -> set[str]:
    """中文零依赖切词：连续汉字取 2 字窗口。"""
    out: set[str] = set()
    for seg in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        for i in range(len(seg) - 1):
            out.add(seg[i:i + 2])
    return out


def topic_overlap(user_text: str, ref_text: str, similarity=None) -> float:
    """这一轮和参照文本的话题重合度。0 = 完全换题。"""
    sim = similarity
    if sim is None:
        from .similarity import DEFAULT
        sim = DEFAULT
    try:
        return max(0.0, min(1.0, float(sim(user_text, ref_text))))
    except Exception:
        return 0.0


def formality_score(text: str) -> float:
    """粗略的语域指标：越正式分越高。

    用户不说「你冒犯我了」，他会突然把句子变短、不再用表情、
    不再反问你。所以信号在语域上，不在话题上。
    """
    if not text:
        return 0.0
    n = _norm_len(text)
    score = 0.0
    if any(p in text for p in ("您", "请", "谢谢", "不好意思", "抱歉")):
        score += 0.4
    if "？" not in text and "?" not in text:
        score += 0.2                     # 不再反问
    punct = sum(text.count(c) for c in "，。！？、")
    if n and punct / n > 0.12:
        score += 0.2                     # 标点密集 = 书面化
    if any(e in text for e in _FILLERS):
        score -= 0.25                    # 语气词 = 放松
    return max(0.0, min(1.0, score))


@dataclass
class TurnFeatures:
    """从用户这一轮里能观察到的全部信号。"""

    text: str
    overlap: float                # 与上一轮 AI 说的 / 用户说的重合度
    length: int
    formality: float
    is_question: bool
    has_filler: bool              # 语气词 / 表情
    gap_hours: float = 0.0
    semantic: str | None = None   # 模型自己在 TAIL 里给的判定

    @property
    def mentions_topic(self) -> bool:
        return self.overlap >= OVERLAP_KEEP


def extract_features(
    user_text: str,
    *,
    pending_claim: str = "",
    predicts: str = "",
    prev_ai_say: str = "",
    prev_user_text: str = "",
    extra_refs: list[str] | None = None,
    gap_hours: float = 0.0,
    semantic: str | None = None,
    similarity=None,
) -> TurnFeatures:
    text = user_text or ""

    # 参照物不能只取一个。用户接的可能不是「上一轮 AI 那句话」，
    # 而是几轮前的话题、预言里描述的迹象、或者挂起时的话题窗口。
    # 取最大值才是「有没有回到那个话题」的正确问法。
    refs = [prev_ai_say, prev_user_text, pending_claim, predicts]
    refs += list(extra_refs or [])
    overlaps = [topic_overlap(text, r, similarity) for r in refs if r]
    overlap = max(overlaps) if overlaps else 0.0

    return TurnFeatures(
        text=text,
        overlap=overlap,
        length=_norm_len(text),
        formality=formality_score(text),
        is_question=("？" in text or "?" in text),
        has_filler=any(e in text for e in _FILLERS),
        gap_hours=gap_hours,
        semantic=semantic,
    )


# ---------------------------------------------------------------- 判定


@dataclass
class Verdict:
    kind: str
    reason: str
    settle: bool                      # 要不要当轮结账
    relation_signal: dict | None = None
    delayed: bool = False


def judge(
    feat: TurnFeatures,
    *,
    baseline_formality: float = 0.3,
    baseline_length: float | None = None,
) -> Verdict:
    """把一轮用户输入判成一个结果。"""

    # ① 模型自己判了 —— 主路径。它读得懂语义，启发式只是兜底。
    if feat.semantic in ("confirmed", "refuted"):
        # 但语域突变要压过它：受伤比读没读准重要。
        if _looks_hurt(feat, baseline_formality):
            return Verdict(
                kind=SUSPECTED_HURT,
                reason="模型认为读对了，但语域变了 —— 关系信号优先",
                settle=True,
                relation_signal={"suspected_hurt": True},
            )
        return Verdict(
            kind=CONFIRMED if feat.semantic == "confirmed" else REFUTED,
            reason="模型在 TAIL 里自己判的",
            settle=True,
        )

    # ② 语域突变。最贵、最容易漏的信号。
    if _looks_hurt(feat, baseline_formality):
        return Verdict(
            kind=SUSPECTED_HURT,
            reason="语域变正式、不再用语气词 —— 更可能是受伤，不是没兴趣",
            settle=True,
            relation_signal={"suspected_hurt": True},
        )

    # ③ 话题明显断了 —— 三种含义不可区分，挂起。
    if feat.overlap < OVERLAP_DROP:
        return Verdict(
            kind=WATCH,
            reason="没接话题。可能只是不关注，可能回避，可能被冒犯 —— 现在判不了",
            settle=False,
        )

    # ④ 话题接住了。再排除「接了但话突然变很短」。
    if feat.overlap >= OVERLAP_KEEP:
        if baseline_length is not None and baseline_length > 8 \
                and feat.length < baseline_length * 0.5:
            return Verdict(
                kind=WATCH,
                reason="接了话题但话变很短，可能读准了但说早了，先挂起",
                settle=False,
            )
        return Verdict(kind=CONFIRMED, reason="接住了话题", settle=True)

    # ⑤ 中间地带：不判。
    return Verdict(
        kind=WATCH,
        reason="话题重合度模棱两可，不猜",
        settle=False,
    )


def _looks_hurt(feat: TurnFeatures, baseline_formality: float) -> bool:
    rise = feat.formality - baseline_formality
    return rise > 0.3 and not feat.has_filler


def judge_delayed(feat: TurnFeatures, *, expired: bool) -> Verdict:
    """挂起的预言后来怎么收尾。"""
    if expired:
        return Verdict(kind=EXPIRED, reason="超时。这条信息对这个人本就不重要",
                       settle=True)
    if feat.mentions_topic:
        return Verdict(kind=CONFIRMED_DELAYED,
                       reason="话题后来回来了，说明读对了，只是当时说早了",
                       settle=True, delayed=True)
    return Verdict(kind=REFUTED_DELAYED,
                   reason="话题回来了但版本对不上", settle=True, delayed=True)


# 这张表要跟 pending_table.png 对得上
OUTCOME_LABEL = {
    CONFIRMED: "校准账本",
    REFUTED: "校准账本",
    CONFIRMED_DELAYED: "校准 + 关系",
    REFUTED_DELAYED: "校准",
    SUSPECTED_HURT: "关系账本",
    WATCH: "暂不记账",
    EXPIRED: "丢弃",
}
