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
    # 这条预言已经被「极短回应」晾过几轮了。连续两次就够判 refuted 了。
    minimal_streak: int = 0
    semantic: str | None = None   # 模型自己在 TAIL 里给的判定
    topic: str | None = None      # 异步判定的结果：returned / left / unclear
    topic_why: str = ""

    @property
    def mentions_topic(self) -> bool:
        """预言有没有被证实。

        语义判定优先于字面重合 —— 这就是接 LLM 的目的：
        「投了两家简历」和「会说出具体方案」字面几乎不重合，语义上完全对上。

        注意 unclear 的语义：**「还没说到」，不是「说反了」**。
        以前 unclear 会掉进字面兜底，而字面重合低就判 topic_left，
        等于把「用户还没提这件事」当成「预言错了」——
        这是命中率被压到 40% 上下最主要的原因。
        """
        if self.topic == "returned":
            return True
        if self.topic == "left":
            return False
        if self.topic == "unclear":
            # 明确判了「没涉及」—— 不能当证实，也不能当否证
            return False
        return self.overlap >= OVERLAP_KEEP

    @property
    def topic_left(self) -> bool:
        """预言有没有被**否证**（不是「有没有换题」）。

        只有语义判定明确说 left 才算。字面重合低一律不算 ——
        用户换个话题聊别的，不等于他把这条预言否掉了。
        """
        return self.topic == "left"

    @property
    def topic_undetermined(self) -> bool:
        """预言还没被验证 —— 该挂起，不该记账。"""
        return self.topic == "unclear" or self.topic is None


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
    topic: str | None = None,
    topic_why: str = "",
    minimal_streak: int = 0,
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
        topic=topic,
        topic_why=topic_why,
        minimal_streak=minimal_streak,
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

    # ②.5 「嗯」「算了」「唉」这种极短回应 —— 它不是换题，是不表态。
    #
    # 这一条是真实数据逼出来的：20 轮里有一半的挂起来自这类回复。
    # 它们和「换了话题」在字面重合度上看起来一样（都是 0），但语义完全不同：
    #   换题   = 用户主动把话题带走 → 三种含义不可区分 → 该挂起
    #   极短回应 = 用户还在这个对话里，只是没接住 → 这是信息，不是噪声
    #
    # 把「不表态」也算成 WATCH 的后果很严重：pending 永远关不掉，
    # 十几轮后堆到 8 条，判定器每轮要判 8 句，warning 爆炸，
    # 而且真正该学的（这条预言提得对不对）一条都没学到。
    #
    # 处理：一次「嗯/算了/唉」不构成反驳也不构成证实，但**连续两次**就是。
    # 单次仍挂起（保留原设计），但要知道它不是「换题」。
    if _is_minimal_reply(feat):
        # 一次可以再看看，连续两次就是明确的「这条预言没说到点上」——
        # 用户连着两轮只回一个「嗯」，说明他既不认同也不反驳，是没被说中。
        # 再挂下去只会让 pending 越堆越多，而一条也学不到。
        if feat.minimal_streak >= 1:
            return Verdict(
                kind=REFUTED,
                reason="连续两轮极短回应 —— 这条预言没说到点上",
                settle=True,
            )
        return Verdict(
            kind=WATCH,
            reason="极短回应，不构成证实或反驳，但也不是换题 —— 再挂一轮看看",
            settle=False,
        )

    # ③ 话题明显断了 —— 三种含义不可区分，挂起。
    if feat.topic_left:
        why = feat.topic_why or "没接话题"
        return Verdict(
            kind=WATCH,
            reason=f"{why}。可能只是不关注，可能回避，可能被冒犯 —— 现在判不了",
            settle=False,
        )

    # ④ 预言被证实了。再排除「接了但话突然变很短」。
    if feat.mentions_topic:
        if baseline_length is not None and baseline_length > 8 \
                and feat.length < baseline_length * 0.5:
            return Verdict(
                kind=WATCH,
                reason="接了话题但话变很短，可能读准了但说早了，先挂起",
                settle=False,
            )
        why = feat.topic_why or "接住了话题"
        return Verdict(kind=CONFIRMED, reason=why, settle=True)

    # ⑤ 还没验证到 —— 挂起，不记账。
    if feat.topic_undetermined:
        return Verdict(
            kind=WATCH,
            reason=feat.topic_why or "这一句还没涉及这条预言，等它出现",
            settle=False,
        )
    return Verdict(
        kind=WATCH,
        reason=feat.topic_why or "信息不足，不猜",
        settle=False,
    )


def _looks_hurt(feat: TurnFeatures, baseline_formality: float) -> bool:
    rise = feat.formality - baseline_formality
    return rise > 0.3 and not feat.has_filler


# 「嗯」「算了」「唉」「……」「可能吧」——没有内容，但也没有离开对话。
_MINIMAL = re.compile(
    r"^[\s，。、,.!?！？~～…]*(?:嗯+|哦+|噢+|啊+|唉+|诶+|呃+|额+|"
    r"算了|好吧|行吧|可能吧|也许吧|是吧|随便|不知道|没什么|"
    r"嗯嗯|对对|是是|哦哦|好好|行行)[\s，。、,.!?！？~～…]*$"
)


def _is_minimal_reply(feat: TurnFeatures) -> bool:
    """这句话是不是「极短、不表态、但也没换题」。

    注意只认**纯**极短回应。一旦带内容（「算了不说了。你觉得我该不该换个工作」）
    就不是这一档了 —— 那里面有真问题，得按正常流程判。
    """
    if feat.length > 6:
        return False
    return bool(_MINIMAL.match((feat.text or "").strip()))


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
