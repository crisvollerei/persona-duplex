"""可选读法的证据结算，与关系观测分离。

读法主要服务于 direct -> say，不要求本轮或下一轮验证。
明确语义证据可结算；相关性、连续短回复、换题和时间经过都不构成真值。
语域变化仅给出暂定关系信号，不关闭预测。过期表示停止保留而非读错。
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
    # 极短回应次数只供观测，不构成证实或反驳。
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
        return False  # 话题重合不等于预测被证实

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
    """把这一轮的用户输入和上下文，摊平成判定需要的那些原始数值。

    **只算数，不判定。** 判定在 judge() 里 —— 分开是为了让 calibrate
    能直接看这些原始分布（「被挂起的样本 overlap 都落在哪」），
    而不是只看结论。

    算出来的特征：
        overlap          用户这句和「参照文本」的最大字面/语义重合度。
                         参照物取多个（上一轮 AI 说的 / 用户说的 /
                         预言 claim / predicts / 挂起时的窗口）取最大值 ——
                         用户接的可能不是上一轮那句话，而是几轮前的话题。
        formality        语域（句长、语气词、标点），受伤检测用
        length           本句长度
        topic            异步语义判定的结果，可能还是 None（没跑完）
        semantic         模型自己在 TAIL 里给的判定
        minimal_streak   这条预言被「极短回应」晾了几轮

    注意 topic 可能是 None：异步判定还没回来时就是 None，
    此时判定会退回字面规则。这是有意的 —— 不能让判定卡在关键路径上。
    """
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
    """读法可一直未知。只有明确语义证据才能结算，相关性不是真值。

    语域变化另作暂定关系观测，既不覆盖证据，也不关闭预言。
    """
    signal = ({"suspected_hurt": True}
              if _looks_hurt(feat, baseline_formality) else None)
    if feat.semantic in ("confirmed", "refuted"):
        return Verdict(
            kind=CONFIRMED if feat.semantic == "confirmed" else REFUTED,
            reason="模型依据用户内容给出的暂定语义判定",
            settle=True, relation_signal=signal,
        )
    if feat.topic in ("returned", "left"):
        return Verdict(
            kind=CONFIRMED if feat.topic == "returned" else REFUTED,
            reason=feat.topic_why or "语义判定认为存在明确证据",
            settle=True, relation_signal=signal,
        )
    return Verdict(
        kind=WATCH,
        reason=feat.topic_why or "没有明确证实或反驳；同话题、短回复和时间经过都不足以结算",
        settle=False, relation_signal=signal,
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
    """延迟不改变证据标准；过期只代表停止保留，不代表读错。"""
    if expired:
        return Verdict(kind=EXPIRED, reason="保留期结束，仍无结论", settle=True)
    verdict = judge(feat)
    if verdict.kind in (CONFIRMED, REFUTED):
        verdict.kind = CONFIRMED_DELAYED if verdict.kind == CONFIRMED else REFUTED_DELAYED
        verdict.delayed = True
    return verdict


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
