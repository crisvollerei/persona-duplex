"""增量解析 HEAD / SAY / TAIL。

为什么需要它：SAY 夹在中间。等整段生成完再吐，用户要盯着空白，
而首字延迟本来就是体验里最贵的一段。所以必须边生成边识别标记，
在 SAY 段一到就开始往外吐。

关键难点是**不能把标记的前缀当正文吐出去**。
比如缓冲区末尾是 `<<<TA`，那是 `<<<TAIL>>>` 的前半截，
这时候得扣住不吐 —— 否则用户会看到「……好<<<TA」。

`_holdback` 就是干这个的。
"""
from __future__ import annotations

import re

from .parse import (
    ParsedTurn,
    detect_declarations,
    detect_read_aloud,
    parse_head,
    parse_tail,
    plain_text,
    safe_say,
)

# 宽松识别协议残留（和 parse 里那份一致）。
# 单独 import 一个名字，避免 stream 依赖 parse 的私有函数。
from .parse import _looks_like_protocol as _looks_like_protocol_full
from .prompt import SECTION_HEAD, SECTION_SAY, SECTION_TAIL

MARKERS = (SECTION_HEAD, SECTION_SAY, SECTION_TAIL)
# SAY 段里出现任何标记都算异常，全都得扣留：
#   · <<<TAIL>>>  —— 正常结束
#   · <<<HEAD>>>  —— 模型把协议重开了一次，后面那段里有 direct
#   · <<<SAY>>>   —— 同样要切，否则标记本身会原样吐给用户
_SAY_END = (SECTION_TAIL, SECTION_HEAD, SECTION_SAY)

# 逐字流式时，缓冲区里至少要攒这么多字符才判「是不是裸文本」——
# 否则只看到 "n" 就判，既判不出字段行也判不出正文。
_FIELD_PEEK = 12

# 第一行是不是 HEAD 的字段行
_HEAD_LINE = re.compile(
    r"^[ \t]*(need|confidence|direct|baseline|displace)[ \t]*[:：]", re.IGNORECASE
)

_HEAD_WORDS = ("need", "confidence", "direct", "baseline", "displace")


def _any_field_prefix(text: str) -> bool:
    """这段文本是不是某个 HEAD 字段名的前缀（还在长）。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(w.startswith(t) for w in _HEAD_WORDS)

SEEK = "seek"
IN_HEAD = "head"
IN_SAY = "say"
IN_TAIL = "tail"


def _holdback(buf: str, markers: tuple[str, ...]) -> int:
    """buf 末尾有多少字符可能是某个标记的开头，必须扣住不吐。"""
    best = 0
    for m in markers:
        for k in range(1, len(m)):
            if buf.endswith(m[:k]):
                best = max(best, k)
    return best


def _is_marker_prefix(text: str) -> bool:
    """这段文本有没有可能正在拼一个标记。"""
    t = text.lstrip()
    if not t:
        return True
    return any(m.startswith(t) for m in MARKERS)


class StreamParser:
    """把流式片段喂进来，吐出可以立刻显示的 SAY 增量。

    有一个容易忽略的细节：标记后面紧跟的换行不该吐出去。
    用户看到回复以空行开头是很怪的，而且逐字流式时会先闪一行空白。
    所以开头和结尾的纯空白都要裁掉，中间的不动。
    """

    def __init__(self) -> None:
        self.buf = ""
        self.state = SEEK
        self.head_text = ""
        self.say_text = ""
        # 整段被判为协议残留 —— 什么都不输出
        self._dead = False
        self.tail_text = ""
        self._bare = False          # 模型完全没按协议来，整段都是回复
        self._emitted = False       # 是否已经吐过非空白内容
        self._pending_ws = ""       # 已收下但还没吐的尾部空白

        # 流式只负责标记边界，不审查内容。
        # 「猜你」那类问题属于措辞，归 prompt 和统计管 —— 在流式层删句子
        # 会把紧跟着的退路（「不对就当我没说」）一起删掉，反而削平了位移。
        self._suppressed: list[str] = []

    # ------------------------------------------------------------ 主接口

    def feed(self, chunk: str) -> list[str]:
        """返回这一片里可以立刻发出去的文本。"""
        if not chunk:
            return []
        if self._dead:
            # 已经判定整段是协议残留 —— 后续内容一律不吐
            return []
        self.buf += chunk
        raw: list[str] = []

        while True:
            if self.state == SEEK:
                if not self._seek(raw):
                    break
                if self._dead:
                    self.buf = ""
                    break
                continue
            if self.state == IN_HEAD:
                if not self._head():
                    break
                continue
            if self.state == IN_SAY:
                if not self._say(raw):
                    break
                continue
            if self.state == IN_TAIL:
                self.tail_text += self.buf
                self.buf = ""
                break
            break

        return self._trim(raw)

    def finish(self) -> ParsedTurn:
        """收尾。把还扣在缓冲区里的东西处理掉。"""
        self._flush_tail_ws()
        if self.buf:
            if self.state == SEEK:
                # 还停在 SEEK，说明缓冲区里是半个标记（比如 "\n<<<HEAD"）。
                # 那不是正文，吞掉。
                if not _is_marker_prefix(self.buf.lstrip()):
                    self.say_text += self.buf
            elif self.state == IN_SAY:
                # 缓冲区里可能是半截标记（"你好<<<HE"）。
                # 无条件并进 say_text 的话，用户看到「你好」，
                # 而账本里记的是「你好<<<HE」。实测踩过。
                hold = 0
                for marker in _SAY_END:
                    for k in range(1, len(marker)):
                        if self.buf.endswith(marker[:k]):
                            hold = max(hold, k)
                keep = self.buf[:-hold] if hold else self.buf
                if keep:
                    self.say_text += keep
            elif self.state == IN_HEAD:
                self.head_text += self.buf
            elif self.state == IN_TAIL:
                self.tail_text += self.buf
            self.buf = ""
        head = parse_head(self.head_text) if self.head_text.strip() else None
        tail = parse_tail(self.tail_text) if self.tail_text.strip() else None

        if self._dead:
            # 整段是协议残留（比如模型只写字段不写标记）—— 一个字都不出
            self.say_text = ""

        say = self.say_text.strip()
        if not say and self._bare and not self._dead:
            # 完全没按协议写的裸文本，整段就是回复
            say = plain_text(self.say_text)

        # 流式这条出口也必须过安全闸。
        #
        # 实测踩过：同一份字节（前四行是字段块 + <<<SAY>>>）
        #   非流式 parse -> '你都想过哪些了。'
        #   流式 StreamParser -> 'need: ...\ndirect: 你其实早就想辞职了\n...'
        # 而 chat 默认走流式 —— 也就是线上走的正是会泄漏的那条。
        say = safe_say(say)

        errors: list[str] = []
        if head is None:
            errors.append("缺少 HEAD")
        if tail is None:
            errors.append("缺少 TAIL")
        if not say:
            errors.append("缺少 SAY")

        return ParsedTurn(
            head=head,
            say=say,
            tail=tail,
            declared=detect_declarations(say),
            # 已经删干净了，所以 say 里当然扫不到 —— 但要把「删过什么」报上去，
            # 否则上游完全不知道这轮发生过违规。
            read_aloud=detect_read_aloud(say) + list(self._suppressed),
            errors=errors,
        )

    # ------------------------------------------------------------ 空白处理

    def _trim(self, pieces: list[str]) -> list[str]:
        """裁掉 SAY 开头和结尾的纯空白，保留中间的。

        两头都要处理：
          · 开头 —— 标记后的换行会被切成单独一片，吐出去先闪一行空白
          · 结尾 —— SAY 末的换行不该出现在回复里，但要等到确认后面
            没有正文才能丢，所以先扣在 _pending_ws 里
        """
        out: list[str] = []
        for piece in pieces:
            if not self._emitted:
                stripped = piece.lstrip()
                if not stripped:
                    continue                # 还没开始，整片空白，丢掉
                self._emitted = True
                piece = stripped

            if piece.strip() == "":
                # 可能是结尾的空白，先扣住
                self._pending_ws += piece
                continue

            # 出现了正文 —— 之前扣的空白是中间的，要补回去
            if self._pending_ws:
                out.append(self._pending_ws)
                self._pending_ws = ""
            body = piece.rstrip()
            out.append(body)
            self._pending_ws = piece[len(body):]

        return out

    def _flush_tail_ws(self) -> None:
        """确认没有更多正文了，把扣住的尾部空白永久丢弃。"""
        self._pending_ws = ""

    # ------------------------------------------------------------ 各状态

    def _seek(self, out: list[str]) -> bool:
        """找第一个标记。找不到就当整段都是回复。

        注意：判「裸文本」之前必须先排除协议残留 —— 否则模型漏写标记时，
        这里会绕过 parse 的安全闸，把 direct 直接吐给用户。
        """
        hits = [(self.buf.find(m), m) for m in MARKERS]
        hits = [(i, m) for i, m in hits if i >= 0]
        if hits:
            i, m = min(hits)
            prefix = self.buf[:i]
            self.buf = self.buf[i + len(m):]
            # 标记之前如果已经有正文，说明模型提前开口了，直接当 SAY
            if prefix.strip() and not _looks_like_protocol_full(prefix):
                self.say_text += prefix
                out.append(prefix)
            self.state = {SECTION_HEAD: IN_HEAD,
                          SECTION_SAY: IN_SAY,
                          SECTION_TAIL: IN_TAIL}[m]
            return True

        # 还没出现标记。判断它到底是在拼标记，还是压根没按协议写。
        t = self.buf.lstrip()
        if t and not _is_marker_prefix(t):
            # 判「裸文本」之前先等等 —— 逐字流式时缓冲区里可能只有一个
            # "n"，那时候既判不出「是字段行」也判不出「是正文」。
            #
            # 判据：等到出现换行，或者攒够一行字段的长度。
            # 一到换行就知道第一行是什么了：是 need:/direct: 这种
            # 字段行就整段作废；否则就是正常裸文本。
            # 逐字流式：还没换行时不能立刻判，否则看到 "n" 就判成裸文本，
            # 后面补上 "eed: ..." 才发现在写字段。
            # 但也**不能无限等** —— 正常回复同样没换行。
            # 判据：只有「当前内容仍是某个字段名的前缀」才值得等。
            if chr(10) not in t and len(t) < _FIELD_PEEK and _any_field_prefix(t):
                return False
            hold = _holdback(t, MARKERS)
            candidate = t[:-hold] if hold else t
            line1 = candidate.split(chr(10), 1)[0]
            if _HEAD_LINE.match(line1) or _looks_like_protocol_full(candidate):
                # 模型只写了字段没写标记 —— 整段是协议残留，一个字都不出。
                # 实测：非流式 parse 已经被 guard 拦下，
                # 流式这里却把整段当 SAY 吐出去，而 chat 默认走流式。
                self._dead = True
                self.state = IN_SAY
                return True
            self._bare = True
            self.state = IN_SAY
            return True

        # 可能是标记的前半截，等下一片
        return False

    def _head(self) -> bool:
        hits = [(self.buf.find(m), m) for m in (SECTION_SAY, SECTION_TAIL)]
        hits = [(i, m) for i, m in hits if i >= 0]
        if not hits:
            return False
        i, m = min(hits)
        self.head_text += self.buf[:i]
        self.buf = self.buf[i + len(m):]
        self.state = IN_SAY if m == SECTION_SAY else IN_TAIL
        return True

    def _say(self, out: list[str]) -> bool:
        # 裸字段也可能出现在 SAY 中。整行出现前扣住字段前缀；一旦确认，
        # 按内部段收取其余内容，不能等 finish() 才发现已经泄露。
        previous_line = self.say_text.rsplit("\n", 1)[-1]
        field = re.search(
            r"^[ \t]*(need|confidence|direct|baseline|displace)[ \t]*[:：]",
            self.buf, re.MULTILINE | re.IGNORECASE)
        hits = [(self.buf.find(m), m) for m in _SAY_END]
        hits = [(i, m) for i, m in hits if i >= 0]
        if field and (field.start() > 0 or not previous_line.strip()) \
                and (not hits or field.start() < min(i for i, _ in hits)):
            piece = self.buf[:field.start()]
            self.say_text += piece
            if piece:
                out.append(piece)
            self.buf = self.buf[field.start():]
            self.state = IN_HEAD
            return True

        if hits:
            i, m = min(hits)
            piece = self.buf[:i]
            if piece:
                self.say_text += piece
                out.append(piece)
            self.buf = self.buf[i + len(m):]
            if m == SECTION_TAIL:
                self.state = IN_TAIL
            elif m == SECTION_HEAD:
                # 模型把协议重开了一次。剩下的按 HEAD 收着，
                # 不能当正文吐出去 —— 那里面就是 direct。
                self.state = IN_HEAD
            else:
                # 重复的 SAY 标记：继续当前段，但标记本身已经吞掉了
                self.state = IN_SAY
            return True

        # 没到结尾。扣住可能是标记前缀的尾巴。
        hold = _holdback(self.buf, _SAY_END)
        n = len(self.buf) - hold
        start = self.buf.rfind("\n", 0, n) + 1
        last_line = self.buf[start:n]
        if (start > 0 or not previous_line.strip()) and _any_field_prefix(last_line):
            n = start
        if n > 0:
            piece = self.buf[:n]
            self.say_text += piece
            out.append(piece)
            self.buf = self.buf[n:]
        return False


def parse_stream(chunks) -> tuple[StreamParser, list[str]]:
    """便利函数：把一整串片段跑完，返回解析器和全部 SAY 增量。"""
    parser = StreamParser()
    deltas: list[str] = []
    for c in chunks:
        deltas.extend(parser.feed(c))
    return parser, deltas
