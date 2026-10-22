"""解析 HEAD / SAY / TAIL。

最重要的原则：best-effort。
解析失败、字段缺失、格式跑偏 —— 一律跳过，永不阻塞对话。
状态坏了可以重建，对话卡住就什么都不剩了。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import DISPLACE_KINDS, DISPLACE_NONE
from .prompt import SECTION_HEAD, SECTION_SAY, SECTION_TAIL

# 一旦输出里出现这些，说明它在「声明位移」—— 这是硬失败，比不做还差。
_DECLARATION_PATTERNS = [
    r"我本来想",
    r"我原本想",
    r"本来想说",
    r"我没说出口",
    r"我差点说",
    r"我差点儿说",
    r"其实我刚才想",
    r"我克制住",
    r"我忍住没",
    r"我欲言又止",
]

# 「声明读」——把「我在解读你」摆到台面上。
# 被读懂的感觉来自你说得对，不来自你说你读懂了。
# 一旦说出来，对方要处理的是你的洞察力，而不是他自己的事。
_READ_ALOUD_PATTERNS = [
    r"^猜",                 # 以「猜」开头 —— 真实数据里全中的是这一条
    r"^听起来你",
    r"^感觉你(?:其实|可能|好像)",
    r"^我觉得你(?:其实|可能)",
    r"^我读到你",
    r"^你这不是在说",
    r"^你不是在说.{1,20}，是",
    r"^你其实(?:不是|并不)",
    r"^(?:这|那)不像是在说",
]


def detect_read_aloud(text: str) -> list[str]:
    """找出「猜你是……」这类把解读过程说出口的句子。

    只在句首匹配：句中出现「猜」可能只是正常用词，
    但一句话以「猜你」开头，几乎一定是在展示洞察。
    """
    hits: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for pat in _READ_ALOUD_PATTERNS:
            if re.search(pat, line):
                hits.append(line[:40])
                break
    return hits


@dataclass
class Head:
    need: str = ""
    confidence: float = 0.0
    direct: str = ""
    displace: str = DISPLACE_NONE


@dataclass
class SettleItem:
    pending_id: str
    outcome: str          # confirmed | refuted
    evidence: str = ""


@dataclass
class PendingDraft:
    claim: str = ""
    confidence: float = 0.0
    predicts: str = ""


@dataclass
class Tail:
    settle: list[SettleItem] = field(default_factory=list)
    delta: dict = field(default_factory=dict)
    pending: PendingDraft | None = None


@dataclass
class ParsedTurn:
    head: Head | None
    say: str
    tail: Tail | None
    declared: list[str] = field(default_factory=list)   # 命中的声明句式
    read_aloud: list[str] = field(default_factory=list)  # 命中的「猜你是……」句式
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.say)

    @property
    def has_displacement(self) -> bool:
        return (
            self.head is not None
            and self.head.displace not in ("", DISPLACE_NONE)
        )

    @property
    def is_pure_say(self) -> bool:
        return self.head is None and self.tail is None

    def is_consistent(self) -> bool:
        """HEAD 里说有位移，但 direct 和 say 几乎一样 → 在装。

        反过来 direct 为空也算不一致：没写真心话就没有位移可谈。
        """
        if self.head is None:
            return True
        if self.head.displace in ("", DISPLACE_NONE):
            return True
        if not self.head.direct.strip():
            return False
        return not _similar(self.head.direct, self.say)


def _norm(s: str) -> str:
    return re.sub(r"[\s，。、,.!?！？~～…\"'“”‘’]", "", s)


def _similar(a: str, b: str, threshold: float = 0.9) -> bool:
    """相似度用包含关系近似。

    只查完全相同是不够的：模型很容易把 direct 复制一遍再加个「啊」，
    那仍然是没有位移的装深沉。
    """
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) < 4:
        return False
    common = sum(1 for c in set(shorter) if c in longer)
    return common / max(1, len(set(shorter))) >= threshold and \
        len(shorter) / len(longer) > 0.6


def _split_sections(raw: str) -> tuple[str, str, str]:
    """返回 (head, say, tail) 原文。缺段就是空串。"""
    text = raw

    i_head = text.find(SECTION_HEAD)
    i_say = text.find(SECTION_SAY)
    i_tail = text.find(SECTION_TAIL)

    head_raw = ""
    say_raw = ""
    tail_raw = ""

    if i_say == -1 and i_head == -1 and i_tail == -1:
        return "", text.strip(), ""

    if i_head != -1:
        end = min([x for x in (i_say, i_tail) if x != -1] + [len(text)])
        head_raw = text[i_head + len(SECTION_HEAD):end]

    if i_say != -1:
        end = i_tail if i_tail != -1 else len(text)
        say_raw = text[i_say + len(SECTION_SAY):end]

    if i_tail != -1:
        tail_raw = text[i_tail + len(SECTION_TAIL):]

    # 兜底：如果 SAY 段没找到，但 HEAD 找到了，就把 HEAD 之后剩余部分当 SAY
    if i_say == -1 and say_raw == "":
        say_raw = text[i_head + len(SECTION_HEAD):] if i_head != -1 else text
        # 但头段之后如果直接跟着 TAIL，说明模型漏了 SAY —— 不能把 TAIL 当回复
        if i_tail != -1 and i_head != -1:
            say_raw = text[i_head + len(SECTION_HEAD):i_tail]

    # SAY 里如果混进了 TAIL 标记（模型忘记分段），也要切掉
    if say_raw:
        for marker in (SECTION_HEAD, SECTION_TAIL):
            if marker in say_raw:
                say_raw = say_raw.split(marker)[0]

    return head_raw.strip(), say_raw.strip(), tail_raw.strip()


# 注意：这里一律用 [ \t]* 而不是 \s*。
# \s 会把换行也吃掉，导致「direct:（空）」直接吞掉下一行的内容，
# 把一个空字段判成有值。
_LEAD = r"^[ \t]*"
_SEP = r"[ \t]*[:：][ \t]*"


def _get_field(block: str, name: str) -> str:
    m = re.search(rf"{_LEAD}{name}{_SEP}(.*)$", block, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _to_float(s: str, default: float = 0.0) -> float:
    raw = s or ""
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        return default
    try:
        v = float(m.group(0))
    except ValueError:
        return default
    # 只有明确写了百分号才当百分比。裸数字 5 就是 5，然后被夹到 1.0。
    if "%" in raw or "％" in raw:
        v /= 100.0
    return max(0.0, min(1.0, v))


def parse_head(block: str) -> Head:
    displace = _get_field(block, "displace").lower()
    if displace in ("", "无", "-", "null"):
        displace = DISPLACE_NONE
    # 允许写成 "降级（判断→问题）"
    if displace not in DISPLACE_KINDS:
        for k in DISPLACE_KINDS:
            if k != DISPLACE_NONE and k in displace:
                displace = k
                break
        else:
            displace = DISPLACE_NONE
    return Head(
        need=_get_field(block, "need"),
        confidence=_to_float(_get_field(block, "confidence")),
        direct=_get_field(block, "direct"),
        displace=displace,
    )


def parse_tail(block: str) -> Tail:
    tail = Tail()
    if not block.strip():
        return tail

    # settle: id | outcome | evidence
    m = re.search(rf"{_LEAD}settle{_SEP}(.+)$", block, re.MULTILINE | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        if raw not in ("-", "无", "none", ""):
            for piece in re.split(r"[;；][ \t]*(?=p_)", raw):
                parts = [p.strip() for p in piece.split("|")]
                if len(parts) < 2:
                    continue
                pid = parts[0].strip(",， ")
                outcome = parts[1].strip().lower()
                if outcome not in ("confirmed", "refuted"):
                    continue
                tail.settle.append(
                    SettleItem(pid, outcome, parts[2] if len(parts) > 2 else "")
                )

    # delta
    m = re.search(rf"{_LEAD}delta{_SEP}(.*)$", block, re.MULTILINE | re.IGNORECASE)
    if m:
        rest = block[m.end():]
        for line in rest.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^(pending|settle)[ \t]*[:：]", line, re.IGNORECASE):
                break
            dm = re.match(r"^[-*]?[ \t]*([a-zA-Z_.]+)[ \t]*[:：][ \t]*(.+)$", line)
            if dm:
                tail.delta[dm.group(1).strip()] = dm.group(2).strip()

    # pending
    m = re.search(rf"{_LEAD}pending{_SEP}(.*)$", block, re.MULTILINE | re.IGNORECASE)
    if m:
        rest = block[m.end():]
        p = PendingDraft()
        for line in rest.splitlines():
            dm = re.match(r"^[ \t]*[-*]?[ \t]*(claim|confidence|predicts)[ \t]*[:：][ \t]*(.+)$",
                          line, re.IGNORECASE)
            if dm:
                k = dm.group(1).lower()
                v = dm.group(2).strip()
                if k == "claim":
                    p.claim = v
                elif k == "confidence":
                    p.confidence = _to_float(v)
                else:
                    p.predicts = v
        if p.claim and p.predicts:
            tail.pending = p

    return tail


def detect_declarations(text: str) -> list[str]:
    """找出「把位移说出来」的句子。这是硬失败。"""
    hits = []
    for pat in _DECLARATION_PATTERNS:
        for m in re.finditer(pat, text):
            start = max(0, m.start() - 12)
            end = min(len(text), m.end() + 24)
            hits.append(text[start:end].replace("\n", " "))
    return hits


def parse(raw: str) -> ParsedTurn:
    head_raw, say_raw, tail_raw = _split_sections(raw)

    head = parse_head(head_raw) if head_raw else None
    tail = parse_tail(tail_raw) if tail_raw else None

    errors: list[str] = []
    if head is None:
        errors.append("缺少 HEAD")
    if tail is None:
        errors.append("缺少 TAIL")
    if not say_raw:
        errors.append("缺少 SAY")

    # 声明检查只看 SAY，因为那才是用户会看到的
    declared = detect_declarations(say_raw)
    read_aloud = detect_read_aloud(say_raw)

    return ParsedTurn(head=head, say=say_raw, tail=tail, declared=declared,
                      read_aloud=read_aloud, errors=errors)
