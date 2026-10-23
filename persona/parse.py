"""解析 HEAD / SAY / TAIL。

这里有两层规则，看起来矛盾，其实分工不同：

**第一层：结构解析，best-effort**
    字段缺失、格式跑偏、标记乱序 —— 一律跳过，永不阻塞对话。
    状态坏了可以重建，对话卡住就什么都不剩了。

**第二层：内容安全，零容忍**
    只要输出里出现 HEAD 字段行、近似标记、脚手架回显，
    **整段清空**，不给用户看（safe_say）。

为什么两层规则不同：第一层失败只是「这轮少记了点状态」，
第二层失败是「把内心独白发给了用户」—— 后者不可恢复。
空回复用户会再问一句；direct 泄露他可能就再也不说了。

实测的四种漏法（都写过回归测试）：
  1. 模型完全不写标记，只把四个字段列出来
  2. 标记写错一个字（少一个 > / 小写 / 内部空格）
  3. 流式路径走了另一套判定，绕过了整段解析的 guard
  4. 多阶段的清洗层把「标记之前」当正文
所以安全判定要宽松（宁可误杀），且必须覆盖所有出口。
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
    baseline: str = ""  # 当前人格在本情境下的常态表达，不是 neutral direct


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
    # 模型明确说「再等等」的预言 id。必须保留 ——
    # 丢掉的话引擎会退回字面判定，可能把「只是附和」判成 CONFIRMED。
    watch: list[str] = field(default_factory=list)


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
        """只检查相对人格基准的位移；旧记录缺基准时不作推断。

        字面相似只是格式诊断，不证明语义位移真的成立。
        """
        if self.head is None:
            return True
        if self.head.displace in ("", DISPLACE_NONE):
            return True
        if not self.head.baseline.strip():
            return True
        return not _similar(self.head.baseline, self.say)


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
        # 短句不能直接判「非复述」。
        # 实测 direct='别走' say='别走啊' —— 明显是复述，但旧逻辑
        # 因为 shorter 只有 2 字就 return False，装深沉完全抓不到。
        # 短句改用「包含」判断：短的整个出现在长的里面就是复述。
        if len(shorter) >= 2 and shorter in longer:
            return len(shorter) / len(longer) > 0.5
        return False
    common = sum(1 for c in set(shorter) if c in longer)
    return common / max(1, len(set(shorter))) >= threshold and \
        len(shorter) / len(longer) > 0.6


_HEAD_FIELDS = ("need", "confidence", "direct", "baseline", "displace")
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


# 宽松的标记识别：容忍大小写、空格、少写一个 ">"
_SECTION_LOOSE = re.compile(
    r"<<\s*\^?\s*(HEAD|SAY|TAIL)\s*>>+", re.IGNORECASE
)

# HEAD 字段行：允许前后空白、全角冒号、大小写
_FIELD_LINE = re.compile(
    r"^[ \t]*(need|confidence|direct|baseline|displace)[ \t]*[:：]",
    re.IGNORECASE | re.MULTILINE,
)

# 模型有时不写字段名，而是用脚手架式的括号回显：
#   （内心：……／位移方式：降级）
_SCAFFOLD = re.compile(
    r"[（(]\s*(内心|心里话|素回复|原始回复|人格基准|位移方式|判断)[^）)]*[）)]"
)


def _looks_like_protocol(text: str) -> bool:
    """这段文本像不像「协议残留」而不是正常回复。

    判据取最宽的那种 —— 宁可误杀也不要漏放：
      · 任何形式的段落标记（容忍写错大小写/空格/少个 >）
      · 出现 HEAD 字段行（need:/confidence:/direct:/displace:）
      · 出现脚手架式回显（（内心：…））

    为什么要这么宽：DIRECT 永不进入输出是硬约束，
    而模型漏写标记是很常见的漂移。实测四种漏法：
        1. 完全不写标记，只写四个字段
        2. 标记写错（<<<SAY>> 少一个 >）
        3. 流式解析走另一条路，没经过分段的 guard
        4. tool/persona 的 clean_phase2 把标记之前当正文
    逐处打补丁总会漏一处，所以统一在这里判。
    """
    t = text or ""
    if not t.strip():
        return False
    return bool(_SECTION_LOOSE.search(t) or _FIELD_LINE.search(t)
                or _SCAFFOLD.search(t))


# 模型把内部特殊 token 当正文写出来了。
# 实测 DeepSeek 会在回复末尾带上 "<｜end▁of▁sentence｜>" ——
# 这个给用户看是明显的 bug。用正则通吃各家写法：
#   <|endoftext|>  <｜end▁of▁sentence｜>  <end_of_turn>  [INST] ...
_SPECIAL_TOKEN = re.compile(
    r"<[|｜]?[\w▁\s]{1,40}[|｜]?>\s*$"      # 结尾的特殊 token
    r"|<[|｜][^>]{1,40}[|｜]>"                  # 任何位置的 <|...|> 形式
)


def strip_special_tokens(text: str) -> str:
    """去掉模型漏出来的特殊 token。

    这些不该出现在用户可见文本里。注意只在**结尾**或**明确的
    <|...|> 形式**上动手 —— 不能把正常的尖括号内容（比如
    「<3」或者代码片段）也删了。
    """
    t = text or ""
    prev = None
    while prev != t:
        prev = t
        t = _SPECIAL_TOKEN.sub("", t).rstrip()
    return t


def safe_say(text: str) -> str:
    """最后一道闸：用户可见的文本必须过这里。

    这是硬约束的落点，**所有出口**都要过：
        parse 的 SAY 段、stream 的最终文本、clean_phase2 的结果。

    做两件事：
      1. 去掉漏出来的特殊 token（<｜end▁of▁sentence｜> 之类）
      2. 看起来像协议残留就整段清空 —— 宁可这轮没回复，
         也不能把内心独白发出去。空回复用户会再问一句，
         direct 泄露他可能就再也不说了。
    """
    t = strip_special_tokens((text or "").strip())
    if not t:
        return ""
    if _looks_like_protocol(t):
        return ""
    return t


def plain_text(raw: str) -> str:
    """没有任何协议标记时，整段就是回复。

    一旦出现标记 —— 或者看起来像协议残留 —— 就不能拿原文兜底，
    否则 HEAD 里的 direct 会被当成回复发给用户。宁可空着。
    """
    if _looks_like_protocol(raw or ""):
        return ""
    return (raw or "").strip()


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

    # 第一个标记之前先开的口。
    #
    # 但这里有个**安全边界**：那段「开头」可能是模型漏写 <<<HEAD>>>
    # 但把四个字段裸着写出来的内容，而 direct 就在里面。
    #
    # 实测踩过（这是最严重的一次）：
    #   raw = "need: 只想被听\nconfidence: 0.82\ndirect: 你根本不是在问我\n
    #          displace: 降级\n\n<<<SAY>>>\n我没事。\n"
    #   旧逻辑把 SAY 之前全当正文 -> direct 原样发给了用户，
    #   而且写进 history，之后每一轮 prompt 都带着这段内心话。
    #
    # 判据：这段里出现 HEAD 字段行 = 它是 HEAD，不是正文。整段丢弃。
    first = min(x for x in (i_head, i_say, i_tail) if x != -1)
    if first > 0:
        prefix = text[:first]
        if not _HEAD_FIELD_RE.search(prefix):
            say_raw += prefix
        # 否则：它是没写标记的 HEAD —— 直接丢掉，一个字都不能发

    if i_head != -1:
        # HEAD 段的结束位置。
        #
        # 注意不能只取「下一个标记」—— 如果模型把顺序写乱了
        # （SAY 在 HEAD 之前，比如 <<<SAY>>>正文<<<HEAD>>>字段<<<TAIL>>>），
        # 那 i_say < i_head，min() 会算出负数长度的区间，head 直接变空。
        #
        # 实测踩过：同一份字节
        #   非流式 head=False / displace=None
        #   流式   head=True  / displace='降级'
        # 两份账本记下不同的状态，重放可比性直接没了。
        #
        # 正确做法：只取 HEAD 之后、且**位置大于 i_head** 的标记。
        after = [x for x in (i_say, i_tail) if x != -1 and x > i_head]
        end = min(after) if after else len(text)
        head_raw = text[i_head + len(SECTION_HEAD):end]  # noqa: E501

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
    # 百分比只在「首个数字本身不在 0-1 区间」时才除。
    #
    # 模型常把两种写法一起写：`0.85（85%）`、`0.6（60%）`。
    # 旧逻辑只要串里出现 % 就整体除 100 —— 0.85 变成 0.0085，
    # 于是这一轮被判成「完全没把握」，还把脏样本写进校准账本。
    #
    # 判据：看**首个数字**。已经在 [0,1] 就是小数写法，不再除。
    if ("%" in raw or "％" in raw) and not (0.0 <= v <= 1.0):
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
        baseline=_get_field(block, "baseline"),
    )


def parse_tail(block: str) -> Tail:
    tail = Tail()
    if not block.strip():
        return tail

    # settle: id | outcome | evidence
    #
    # 用 finditer 而不是 search —— 模型常写成一行一条：
    #     settle: p_0001 | confirmed | a
    #     settle: p_0002 | refuted | b
    # 只取第一条的话，第二条的判定被静默丢掉，它继续挂着，
    # 下一轮可能被字面判成 CONFIRMED（污染校准账本）。
    for m in re.finditer(rf"{_LEAD}settle{_SEP}(.+)$", block,
                         re.MULTILINE | re.IGNORECASE):
        raw = m.group(1).strip()
        if raw in ("-", "无", "none", ""):
            continue
        for piece in re.split(r"[;；][ \t]*(?=p_)", raw):
            parts = [p.strip() for p in piece.split("|")]
            if len(parts) < 2:
                continue
            pid = parts[0].strip(",， ")
            outcome = parts[1].strip().lower()
            if outcome == "watch":
                # 模型明确说「再等等」——必须保留这个意图。
                # 旧实现直接 continue 丢掉，引擎拿不到 model_item，
                # 就退回字面判定，可能把「只是附和」判成 CONFIRMED。
                # 那不接话题记成 refuted 更糟（凭空提高命中率）。
                tail.watch.append(pid)
                continue
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

    # 最后一道闸：像协议残留就清空。
    # 这一步必须在这里，不能只在上游各处判 —— 实测有四种漏法
    # （无标记只写字段块 / 标记写错 / 流式走另一条路 / clean_phase2 兜底），
    # 逐处打补丁总会漏。
    say = safe_say(say_raw)
    if not say and say_raw:
        errors.append("SAY 被安全闸清空（疑似协议残留）")

    # 声明检查只看 SAY，因为那才是用户会看到的
    declared = detect_declarations(say)
    read_aloud = detect_read_aloud(say)

    return ParsedTurn(head=head, say=say, tail=tail, declared=declared,
                      read_aloud=read_aloud, errors=errors)
