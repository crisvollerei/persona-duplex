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

from .parse import ParsedTurn, parse_head, parse_tail, detect_declarations
from .prompt import SECTION_HEAD, SECTION_SAY, SECTION_TAIL

MARKERS = (SECTION_HEAD, SECTION_SAY, SECTION_TAIL)
_SAY_END = (SECTION_TAIL,)

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
        self.tail_text = ""
        self._bare = False          # 模型完全没按协议来，整段都是回复
        self._emitted = False       # 是否已经吐过非空白内容
        self._pending_ws = ""       # 已收下但还没吐的尾部空白

    # ------------------------------------------------------------ 主接口

    def feed(self, chunk: str) -> list[str]:
        """返回这一片里可以立刻发出去的文本。"""
        if not chunk:
            return []
        self.buf += chunk
        raw: list[str] = []

        while True:
            if self.state == SEEK:
                if not self._seek(raw):
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
            if self.state in (SEEK, IN_SAY):
                self.say_text += self.buf
            elif self.state == IN_HEAD:
                self.head_text += self.buf
            elif self.state == IN_TAIL:
                self.tail_text += self.buf
            self.buf = ""
        head = parse_head(self.head_text) if self.head_text.strip() else None
        tail = parse_tail(self.tail_text) if self.tail_text.strip() else None
        say = self.say_text.strip()

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
            out.append(piece)

        return out

    def _flush_tail_ws(self) -> None:
        """确认没有更多正文了，把扣住的尾部空白永久丢弃。"""
        self._pending_ws = ""

    # ------------------------------------------------------------ 各状态

    def _seek(self, out: list[str]) -> bool:
        """找第一个标记。找不到就当整段都是回复。"""
        hits = [(self.buf.find(m), m) for m in MARKERS]
        hits = [(i, m) for i, m in hits if i >= 0]
        if hits:
            i, m = min(hits)
            prefix = self.buf[:i]
            self.buf = self.buf[i + len(m):]
            # 标记之前如果已经有正文，说明模型提前开口了，直接当 SAY
            if prefix.strip():
                self.say_text += prefix
                out.append(prefix)
            self.state = {SECTION_HEAD: IN_HEAD,
                          SECTION_SAY: IN_SAY,
                          SECTION_TAIL: IN_TAIL}[m]
            return True

        # 还没出现标记。判断它到底是在拼标记，还是压根没按协议写。
        t = self.buf.lstrip()
        if t and not _is_marker_prefix(t):
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
        i = self.buf.find(SECTION_TAIL)
        if i >= 0:
            piece = self.buf[:i]
            if piece:
                self.say_text += piece
                out.append(piece)
            self.buf = self.buf[i + len(SECTION_TAIL):]
            self.state = IN_TAIL
            return True

        # 没到结尾。扣住可能是标记前缀的尾巴。
        hold = _holdback(self.buf, _SAY_END)
        n = len(self.buf) - hold
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
