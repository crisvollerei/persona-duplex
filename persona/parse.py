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
#
# 判定标准不是「猜」这个字，是**猜的对象**：
#   猜你 / 我猜你 / 我猜这（指对方的话）  → 在读人，违规
#   我猜他也不会来                        → 在猜第三方的行为，正常
# 所以不能简单禁「猜」，要禁「把猜测指向对方的那个句式」。
# 真实跑出来的 20 轮里有 7 条是「我猜你……」，旧规则一条都没抓到 ——
# 因为旧规则只认行首的「猜」，而这些句子以「我猜」开头。
_READ_ALOUD_PATTERNS = [
    # —— 指向对方的猜测（真正要禁的）
    r"(?:^|[，。！？；：\s])我?猜(?:你|您)",
    r"(?:^|[，。！？；：\s])我?猜(?:这|那)(?:句|话|个|些|种)?",
    r"(?:^|[，。！？；：\s])我?猜[“\"「]",
    # 「猜一句：」「先猜一个：」「先猜：」—— 同一个模式的换皮写法。
    # 实测：禁掉「我猜你」之后，模型立刻改用它，因为它们意思一样。
    r"(?:^|[，。！？；：\s])猜(?:一句|一下|一个)?\s*[:：]",
    r"^先?猜",
    r"我要猜",
    # —— 行首的判断式开场（原来的那批）
    r"^猜(?!他|她|它)",
    r"^听起来你",
    r"^感觉你(?:其实|可能|好像)",
    r"^我觉得你(?:其实|可能)",
    r"^我读到你",
    r"^你这不是在说",
    r"^你不是在说.{1,20}，是",
    r"^你其实(?:不是|并不)",
    r"^(?:这|那)不像是在说",
]

# 一句话里如果出现这些，说明「猜」的对象是第三方或外部事实，不算读人。
_THIRD_PARTY = re.compile(r"猜(?:他|她|它|他们|她们|别人|对方)")


def _is_read_aloud(line: str) -> bool:
    if _THIRD_PARTY.search(line):
        return False
    for pat in _READ_ALOUD_PATTERNS:
        if re.search(pat, line):
            return True
    return False


def detect_read_aloud(text: str) -> list[str]:
    """找出 say 里「把解读过程说出口」的句子。**只用于统计，不用于拦截。**

    这一条的历史反复值得记下来，因为它是一条被我修歪又修回来的规则：

    v1 只认行首的「猜」 → 漏掉「我猜你……」，真跑 20 轮里 7 条没抓到。
    v2 按对象判（猜对方 = 违规）→ 抓到了，但把「猜错你就说」这种
       带退路的台阶也算成违规。数据上一看：26 条里 4 条有退路、
       22 条没退路。也就是说 v2 罚对了 22 条，但错罚了 4 条。
    v3 我在输出侧硬删 → **修歪了**。删「猜一句：失眠跟工作有关」
       会把紧跟着的「不对就当我没说」一起删掉 —— 恰好删掉了设计里
       最该保留的那半句（退路）。用户看到的回复因此变短、变硬。

    现在回到设计文档 §7 的原意：

        假读错（明知故问）是位移的**最高明形式**，自带免责空间。
        硬线是「问完之后留不留退路」。留 = 台阶；逼 = 操控。

    也就是说「猜」本身不是问题，**不给退路的断言**才是。而这一层
    用正则分不干净（「我猜你不是真想问，是已经想走了」没有退路，
    「我猜你难受那会儿一个人扛的，猜错你就说」有退路 —— 结构几乎一样）。
    所以这里只做**观测**：数出来，给 metrics 和 trace 看，
    不修改模型说的话。

    真要让模型少写断言式解读，得从源头治：stance_hint 里那句
    「先说出你的猜测，再给内容」就是在教它这么写（见 prompt.py）。
    """
    hits: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_read_aloud(line):
            hits.append(line[:40])
    return hits


def has_exit_offer(text: str) -> bool:
    """这句话有没有给对方留退路（「猜错你就说」「不对就当我没说」）。

    留了退路 = 台阶；没留 = 审判。这是设计文档 §7 的硬线。
    只用于统计和 trace，不参与拦截。
    """
    return bool(_EXIT_OFFER.search(text or ""))


# 「留了退路」的信号。设计文档 §7：留 = 台阶；逼 = 操控。
_EXIT_OFFER = re.compile(
    r"(否认|否掉|能被你|能让你|猜错|不对就|错了就|错了你|你可以反驳|"
    r"随时否|不认|当我没说|如果我说错|要是我猜错|直接说)"
)


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


_HEAD_FIELDS = ("need", "confidence", "direct", "displace")
_HEAD_FIELD_RE = re.compile(
    r"^[ \t]*(" + "|".join(_HEAD_FIELDS) + r")[ \t]*[:：]", re.IGNORECASE
)


def _after_head_fields(block: str) -> str:
    """HEAD 块里、最后一个 HEAD 字段之后的内容。

    这里存在一个安全边界：direct 就在 HEAD 里，而 direct 永远不能出现在
    用户面前。所以只有在**确认出现过 displace 字段**之后，才把后面的文本
    当作正文 —— 否则宁可返回空串，也不能把 direct 的续行当成回复。

    返回空串是安全的失败：用户看到空回复，但不会看到内心独白。
    """
    lines = block.splitlines(keepends=True)
    last_field = -1
    saw_displace = False
    for i, line in enumerate(lines):
        m = _HEAD_FIELD_RE.match(line)
        if not m:
            continue
        last_field = i
        if m.group(1).lower() == "displace":
            saw_displace = True
    if not saw_displace or last_field == -1:
        return ""
    return "".join(lines[last_field + 1:])


def plain_text(raw: str) -> str:
    """没有任何协议标记时，整段就是回复。

    一旦出现标记，说明这轮解析失败 —— 此时**不能**拿原文兜底，
    否则 HEAD 里的 direct 会被当成回复发给用户。宁可空着。
    """
    text = raw or ""
    if any(m in text for m in (SECTION_HEAD, SECTION_SAY, SECTION_TAIL)):
        return ""
    return text.strip()


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

    # 第一个标记之前先开的口，那是真正的 SAY，不能丢
    first = min(x for x in (i_head, i_say, i_tail) if x != -1)
    if first > 0:
        say_raw += text[:first]

    if i_head != -1:
        end = min([x for x in (i_say, i_tail) if x != -1] + [len(text)])
        head_raw = text[i_head + len(SECTION_HEAD):end]

    if i_say != -1:
        end = i_tail if i_tail != -1 else len(text)
        say_raw += text[i_say + len(SECTION_SAY):end]
    elif i_head != -1:
        # SAY 缺失。绝不能把 HEAD 当回复 —— direct 就在里面。
        say_raw += _after_head_fields(head_raw)

    if i_tail != -1:
        tail_raw = text[i_tail + len(SECTION_TAIL):]

    # SAY 里如果混进了标记（模型忘记分段），也要切掉
    if say_raw:
        for marker in (SECTION_HEAD, SECTION_SAY, SECTION_TAIL):
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
