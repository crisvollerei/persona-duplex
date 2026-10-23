"""多阶段协议：内部素回答与人格表达分工。

text 一次生成 direct/baseline/displace、SAY、TAIL。
tool 分三次调用：内部参考工具、带人格/关系上下文的纯文本 SAY、可选观察工具。
persona 分三次调用：不直接注入人设的 raw、人格层 HEAD+SAY、可选观察工具。
raw 不决定位移，历史只保留用户原话和可解析的 direct；这不是完全中性保证。

位移以当前人格的常态表达 baseline 为参照，普通人格化不算额外位移。
内部字段不直接展示，已经发出的内容不得在失败回退时重复生成。
结构与输入边界可测试，但不因此宣称模型内部心理因果或参数效果已获验证。
"""
from __future__ import annotations

import re

from .config import DISPLACE_KINDS, DISPLACE_NONE
from .parse import (Head, ParsedTurn, PendingDraft, SettleItem, Tail,
                    _similar, detect_declarations, detect_read_aloud, safe_say)

# ---------------------------------------------------------------- schema

# need 的枚举来自人格配置，这里留一个 getter 生成 —— 不同人格的词表不同，
# 写死会导致 guarded / service 的 need 被 schema 拒掉。
def inner_tool(cfg) -> dict:
    needs = list(cfg.needs) or ["只是想说说话"]
    return {
        "type": "function",
        "function": {
            "name": "inner",
            "strict": True,
            "description": "记录素回答、当前人格的常态表达和相对该基准的额外位移。内部字段不直接展示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "need": {"type": "string", "enum": needs},
                    "confidence": {
                        "type": "number",
                        "description": "0 到 1，你对上面那个判断有多确定",
                    },
                    "direct": {
                        "type": "string",
                        "description": "一个没有性格的助手会给出的素回复。有用的、具体的、中性的。没有人格色彩。",
                    },
                    "baseline": {
                        "type": "string",
                        "description": "当前人格在本情境下的常态表达；保留人格，不刻意遮掩或反说。",
                    },
                    "displace": {
                        "type": "string",
                        "enum": list(DISPLACE_KINDS),
                        "description": "相对于人格 baseline 的额外位移，不是相对于 direct。none 是常态。",
                    },
                },
                "required": ["need", "confidence", "direct", "baseline", "displace"],
                "additionalProperties": False,
            },
        },
    }


PHASE1_SYSTEM_SUFFIX = """

[这一轮分两步走]
现在只做第一步，使用 inner 工具记录内部参考，不输出正文。
direct 是中性的素回答，不是隐秘真心话；baseline 是当前人格通常会说的话。
displace 仅标记相对 baseline 的额外变化。正常人格化不算位移。"""

PHASE2_SYSTEM = """现在只输出这个人格实际说出口的 say，覆盖前文的三段输出格式。

规则：
- direct 是中性的内容参考，不是隐秘真心话，不要求隐藏它的核心意思。
- baseline 是当前人格的常态表达；完整遵循所给人格与关系上下文。
- displace 仅指偏离 baseline 的情境性位移。none 时正常人格化，不退回无人格口吻。
- 不许出现「我本来想说」「其实我差点说」「我没说出口」这类声明 ——
  那是在索取信用，比直接说更糟。
- 不许出现「我猜你……」「先猜一个……」来引出内容 —— 那是在展示洞察。
  读法只作为内容生成参考，允许不确定，不必展示解读过程。
- 位移方式是 none 时，就正常说，别绕。
- 只输出要发出去的那句话本身。不要加引号、不要加解释、不要写「好，我说：」。
- **不要写任何标记符**（no <<<HEAD>>> / <<<SAY>>> / <<<TAIL>>>，no need/confidence/
  direct/baseline/displace 字段）。你不是在填表，你就是在说话。"""

PERSONA_PHASE2_SUFFIX = """
[本阶段输出格式 · 覆盖之前的格式]
素回答已经由独立步骤给出，不要重写 direct。
先记录当前人格的常态表达及额外位移，再输出实际说的话：
<<<HEAD>>>
baseline: 当前人格通常会怎样回应，保留人格但不刻意遮掩或反说
displace: 相对于 baseline 的额外位移类型；none 是常态，普通人格化不算位移
<<<SAY>>>
给对方的实际回复
本阶段不要输出 TAIL。内部字段不要出现在 SAY 中。"""


# ---------------------------------------------------------------- 结果


def build_phase1_messages(system_prompt: str, context: str,
                          history: list[dict], user_text: str,
                          stance: str) -> list[dict]:
    """phase 1 的输入。

    注意 system prompt 里那套三段协议说明对这一步是多余的 ——
    但保留它有价值：人格、禁令、位移类型定义都在里面，
    只是把「怎么输出」换成「填 inner 工具」。
    """
    return [
        {"role": "system", "content": system_prompt + "\n\n" + context
                                  + PHASE1_SYSTEM_SUFFIX},
        *history[-12:],
        {"role": "user", "content": f"[当前站位: {stance}]\n\n{user_text}"},
    ]




# ---------------------------------------------------------------- 素回复

# 这一层**故意看不到任何人设**。
# 它的职责只有一个：按事实和逻辑，这句话该怎么接。
#
# 为什么必须隔离：实跑发现，只要人设在同一个 prompt 里，
# 哪怕明确写了「direct 不要带人格」，模型产出的 direct 仍然是
# 被人格染色过的（两个人格对同一句话的「素回复」相似度 0.00）。
# 顺序上人格在前、direct 说明在后，先入为主。
#
# 隔离之后 direct 才真的是中立的。这是「可控」的前提：
# 输入固定了，输出的人格差异才能归因到人设参数上。

RAW_SYSTEM = """你是一个没有性格的助手。只回答一件事：按事实和逻辑，这句话该怎么接。

规则：
- 给最有用的那句。具体、可执行、不绕。
- 不表达自己的感受、立场、态度。
- 不评判对方是什么样的人。
- 不要用比喻、意象、修辞。
- 不要模拟任何角色的口吻。你没有角色。
- 一两句就够。

你回答的是「事情本身该怎么办」，不是「我作为某某会怎么说」。"""


def raw_tool(cfg=None) -> dict:
    """素回复 + 可选读法；无人格的层不能判断人格基准或位移。

    关于 need 的词表：它来自人格配置，不同人格给的词表不一样
    （守岸人 6 个、阿澈 5 个）。这看起来像污染，但**不是**：

        need 是「对方此刻需要什么」的分类标签，不是「我是什么样的人」。
        不同人格关心的事本来就不同 —— 一个冷淡的角色根本不会去分辨
        「要被逗」，那对她是无效选项。

    所以词表随人格变是**对的**：它描述的是这个人格的注意力范围。
    raw 的目标仍是中性回答，但中性程度需另行检验。
    """
    needs = list(cfg.needs) if (cfg is not None and cfg.needs) else ["只是想说说话"]
    return {
        "type": "function",
        "function": {
            "name": "raw_reply",
            "strict": True,
            "description": "给出不含人格的素回复",
            "parameters": {
                "type": "object",
                "properties": {
                    "need": {"type": "string", "enum": needs,
                             "description": "对方此刻需要什么"},
                    "confidence": {"type": "number"},
                    "raw": {"type": "string",
                            "description": "素回复：最有用的那句，不带任何人格色彩"},
                },
                "required": ["need", "confidence", "raw"],
                "additionalProperties": False,
            },
        },
    }


def build_raw_messages(history: list[dict], user_text: str) -> list[dict]:
    """去掉历史中的人格化 SAY 和协议字段；无法中性化的助手消息不转发。

    用户原话仍可能提到角色，因此这不是形式化的完全隔离保证。
    """
    from .parse import parse

    neutral = []
    for msg in history[-12:]:
        if msg.get("role") == "user":
            neutral.append({"role": "user", "content": msg.get("content", "")})
        elif msg.get("role") == "assistant":
            head = parse(msg.get("content", "")).head
            if head and head.direct.strip():
                neutral.append({"role": "assistant", "content": head.direct})
    return [{"role": "system", "content": RAW_SYSTEM},
            *neutral,
            {"role": "user", "content": user_text}]


def to_head(data: dict) -> Head:
    """把素回复调用的产物转成 Head（direct 就是 raw）。"""
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Head(
        need=str(data.get("need", "") or "").strip(),
        confidence=max(0.0, min(1.0, conf)),
        direct=str(data.get("raw", "") or "").strip(),
        displace=DISPLACE_NONE,
    )


def build_phase2_messages(inner: Head, history: list[dict],
                          user_text: str, *, system_prompt: str = "",
                          context: str = "") -> list[dict]:
    """phase 2 的输入：把 direct 和 displace 明确交给它。

    这里**不**再给三段协议说明 —— 它只需要输出一句话。
    给了反而会让它想写标记。
    """
    return [
        {"role": "system", "content": system_prompt + "\n\n" + context
                                     + "\n\n" + PHASE2_SYSTEM},
        *history[-12:],
        {"role": "user", "content": user_text},
        {"role": "assistant",
         "content": f"（素回复：{inner.direct}／人格基准：{inner.baseline}／位移方式：{inner.displace}）"},
        {"role": "user", "content": "说出口。"},
    ]


_ANY_MARKER = re.compile(r"<<<(?:HEAD|SAY|TAIL)>>>")


class StreamCleanPhase2:
    """phase 2 的流式增量过滤。

    两个需求互相打架：
      · 用户不该看到任何标记（<<<HEAD>>> / settle: -）
      · 已经吐出去的字收不回来

    所以只能「扣住可疑片段、确认干净再放行」。可疑的是：
      · 任何可能长成 <<<XXX>>> 的前缀（以 < 开头）
      · 行首的 need:/confidence:/direct:/displace:/settle:/delta:/pending:

    其余内容立刻放行 —— phase 2 是正常说话，绝大多数片段都不需要扣。
    """

    _FIELD = re.compile(
        r"^[ \t]*(need|confidence|direct|baseline|displace|settle|delta|pending)[ \t]*[:：]",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.buf = ""
        self.text = ""          # 放行过的文本
        self._emitted_any = False
        self._in_head = False   # 正处在 HEAD 段（见过 <<<HEAD>>>，还没到 <<<SAY>>>）
        self._done = False      # 见过 TAIL，之后全丢

    def feed(self, chunk: str) -> list[str]:
        self.buf += chunk
        return self._drain(final=False)

    def finish(self) -> list[str]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        out: list[str] = []
        while True:
            cut = self._safe_cut(final=final)
            if cut is None:
                break
            if cut == 0:
                continue        # 有内容被丢弃，缓冲区已更新，继续看
            piece, self.buf = self.buf[:cut], self.buf[cut:]
            out.append(piece)
            self.text += piece
            if piece.strip():
                self._emitted_any = True
        return out

    _FIELD_INLINE = re.compile(
        r"(?:^|[。！？；，\s])(?:settle|delta|pending|need|confidence|"
        r"direct|baseline|displace)[ \t]*[:：]", re.IGNORECASE
    )

    # 字段名（不带前导条件）—— 用来判断「这一行会不会变成字段行」
    _FIELD_START = re.compile(
        r"^[ \t]*(settle|delta|pending|need|confidence|direct|baseline|displace)[ \t]*[:：]",
        re.IGNORECASE,
    )

    def _safe_cut(self, *, final: bool) -> int | None:
        """缓冲区里可以安全放行到哪里。

        用一个最小的状态机消化标记，避免「丢弃」和「放行」互相打架：

            TEXT   —— 正常说话，逐行放行（字段行丢掉）
            HEAD   —— 见过 <<<HEAD>>>，里面的 direct 一律不要，等 SAY
            DONE   —— 见过 <<<TAIL>>>，之后全丢

        为什么要处理 HEAD / DONE：模型偶尔会完整照抄文本协议
        （历史里有旧样例，它会跟着学）。那时 SAY 段才是给用户看的，
        见到 <<<HEAD>>> 直接关掉的话，整句回复就丢了 —— 实测 3/20。
        """
        out_len = 0

        while True:
            if self._done:
                self.buf = ""
                return out_len or None

            m = _ANY_MARKER.search(self.buf)
            if m:
                before = self.buf[:m.start()]
                marker = m.group(0)
                self.buf = self.buf[m.end():]
                kept = self._keep_lines(before) if not self._in_head else ""
                if marker == "<<<HEAD>>>":
                    self._in_head = True
                elif marker == "<<<SAY>>>":
                    self._in_head = False
                else:
                    self._done = True
                if kept:
                    # 先切换标记状态，再单独放行前缀；否则同一片里的 HEAD
                    # 会被跳过，后面的 baseline/direct 可能一起被放行。
                    self.buf = kept + self.buf
                    return len(kept)
                continue

            if self._in_head:
                # HEAD 段里的内容全部丢弃，等下一行
                nl = self.buf.find("\n")
                if nl >= 0:
                    self.buf = self.buf[nl + 1:]
                    continue
                if final:
                    self.buf = ""
                return out_len or None

            # 正常状态：逐行放行
            nl = self.buf.find("\n")
            if nl >= 0:
                line = self.buf[:nl]
                if self._FIELD_START.match(line) or self._FIELD_INLINE.search(line):
                    self.buf = self.buf[nl + 1:]
                    self._in_head = True
                    continue
                return out_len + nl + 1

            # 没有换行符
            cur = self.buf
            if self._FIELD_INLINE.search(cur):
                if final:
                    self.buf = ""
                return out_len or None
            if self._could_be_field(cur, final=final) or self._FIELD_INLINE.search(cur):
                if final:
                    # 收尾时还被扣着的内容必须放行 —— 此时不可能再长出字段行。
                    # 旧实现直接 self.buf = "" 丢掉，实测 "yes" 只输出 "ye"。
                    self.buf = ""
                    return out_len + len(cur) if cur else (out_len or None)
                return out_len or None
            if final:
                return out_len + len(cur) if cur else (out_len or None)

            # 末尾可能正在拼标记 -> 扣住
            stripped = cur.lstrip()
            if stripped.startswith("<") and any(
                    mk.startswith(stripped) for mk in
                    ("<<<HEAD>>>", "<<<SAY>>>", "<<<TAIL>>>")):
                return out_len or None
            hold = 0
            for marker in ("<<<HEAD>>>", "<<<SAY>>>", "<<<TAIL>>>"):
                for k in range(1, len(marker)):
                    if cur.endswith(marker[:k]):
                        hold = max(hold, k)
            cut = len(cur) - hold
            return out_len + cut if cut > 0 else (out_len or None)

    def _keep_lines(self, text: str) -> str:
        """过滤掉字段行，返回能放行的那部分。"""
        keep = []
        for ln in text.splitlines(keepends=True):
            if self._FIELD_START.match(ln) or self._FIELD_INLINE.search(ln):
                break  # 字段的续行同样属于内部内容，直到下一个明确标记
            keep.append(ln)
        return "".join(keep)

    @staticmethod
    def _could_be_field(line: str, *, final: bool = False) -> bool:
        """这一行有没有可能正在长成「settle: ...」这样的字段行。

        **必须要求冒号存在或行还没写完** —— 只做前缀匹配会吞掉正常句子：
        "Sounds good" 逐字喂进来时 "Sounds goo" 的前缀 "s" 以 settle/..."
        这些词开头的话就会被扣住不放。实测英文被吞字、中文正常
        （中文前缀不会撞上字段名），所以中文测试抓不到。

        判据：
          · 已经出现冒号     -> 可能是字段行，扣住
          · 有冒号但字段名不在前面 -> 是正文，放行
          · 完全还没冒号且前缀像字段名 -> 再等等（final 时无条件放行）
        """
        s = line.lstrip()
        if not s:
            return True
        # 已有冒号 = 这一行已经成形，看字段名是否在冒号前
        if ":" in s or "：" in s:
            head = re.split(r"[:：]", s, maxsplit=1)[0].strip().lower()
            return head in ("settle", "delta", "pending", "need",
                            "confidence", "direct", "baseline", "displace")
        if final:
            # 收尾了还没冒号 —— 不可能是字段行
            return False
        # 还没冒号：只有「整行都还是字段名的前缀」才值得等
        low = s.lower()
        for word in ("settle", "delta", "pending", "need", "confidence",
                     "direct", "baseline", "displace"):
            if word.startswith(low) and low:
                return True
        return False


def _looks_like_head_fields(text: str) -> bool:
    """这段文本像不像 HEAD 的字段内容（而不是正常回复）。

    用于 clean_phase2 切分之后 —— 即使标记切对了，
    切出来的「最后一段」也可能是 HEAD 的残片：
        "正文。<<<HEAD>>>秘密想法"  ->  '秘密想法'
    这时候只能靠内容判断：极短、无标点、不像人说的话。
    """
    t = (text or "").strip()
    if not t:
        return False
    # 含字段名直接判
    if re.search(r"(need|confidence|direct|baseline|displace)\s*[:：]", t, re.IGNORECASE):
        return True
    # 极短且不含任何句末标点 —— 正常的 SAY 不会这么写
    return len(t) <= 20 and not re.search(r"[。！？.!?，,~…]", t)


def _split_head_fields(text: str) -> list[str]:
    """把一段疑似 HEAD 的内容按字段名切开，返回各字段的值。

    用于判断「标记之后切出来的这段」是不是某个 HEAD 字段的残片：
        "正文。<<<HEAD>>>秘密想法"  ->  切出的 '秘密想法'
    单独看它像正常句子，只有和 HEAD 字段值比对才认得出。
    """
    parts = re.split(r"(?:need|confidence|direct|baseline|displace)\s*[:：]",
                     text or "", flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def clean_phase2(say: str) -> str:  # noqa: D401
    """phase 2 应该只回一句话。它偶尔会不听话，把协议标记也写出来 ——
    那是文本协议的残影，必须切掉，否则标记会原样发给用户。

    切法是「标记之前的部分保留」：如果它先写了标记再写正文，
    正文在标记之后，取最后一段；如果标记在前面（比如写了 <<<HEAD>>> 清单），
    那就只有第一段之前的东西是干净的。
    """
    text = (say or "").strip()
    # 有些端点不写标记、但会补一行 settle:/delta: —— 同样不是用户该看的。
    text = re.sub(r"^[ \t]*(settle|delta|pending)[ \t]*[:：].*$", "",
                  text, flags=re.MULTILINE).strip()
    if not _ANY_MARKER.search(text):
        # 没有标记也要过安全闸 —— 模型可能只写字段块不写标记，
        # 那种「整段就是回复」的兜底会把 direct 送出去。
        return safe_say(text)
    # 优先保留最后一个标记之后的内容（那才是真正的 SAY）
    parts = _ANY_MARKER.split(text)
    tail = parts[-1].strip()

    # 标记之前如果有 HEAD 字段，说明这是「正文 + 重新写了一遍 HEAD」。
    # 那标记之后的 tail 就是 HEAD 的残片，不是回复 —— 整段作废。
    before = "".join(parts[:-1])
    if _split_head_fields(before):
        return ""
    if tail:
        # 去掉可能残留的字段行（need: / confidence: / direct: / displace:）
        lines = [ln for ln in tail.splitlines()
                 if not re.match(
                     r"^[ \t]*(need|confidence|direct|baseline|displace|settle|delta|pending)"
                     r"[ \t]*[:：]", ln, re.IGNORECASE)]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            # 切出来的这段仍要过安全闸。
            # 实测 "正文。<<<HEAD>>>秘密想法" 会切出 '秘密想法' ——
            # 标记之后的「最后一段」也可能是 HEAD 的残片。
            guarded = safe_say(cleaned)
            if guarded and not _looks_like_head_fields(guarded):
                return guarded

    # **不要**退回「标记之前的部分」。
    #
    # 那段通常正是 HEAD 的四个字段（含 direct）。实测：
    #   clean_phase2("direct: 秘密想法\n<<<SAY>>>") -> 'direct: 秘密想法'
    # tool 协议下这个值直接变成 res.say 发给用户，warnings 还是空的。
    #
    # 空回复是可接受的失败 —— 用户会再问一句；
    # direct 泄露他可能就再也不说了。
    return ""




# ---------------------------------------------------------------- 结算阶段

def tail_tool(pendings: list[dict]) -> dict:
    """第三阶段的工具：结算旧预言 + 挂一条新预言。

    schema 里 settle 的 id 用 enum 限死已有的预言 —— 模型编一个不存在的
    id 是最常见的失败，schema 层直接挡住。
    """
    ids = [p["id"] for p in pendings] or ["（无）"]
    return {
        "type": "function",
        "function": {
            "name": "settle_turn",
            "strict": True,
            "description": "结算上一轮的预言，并（可选地）挂一条新预言。",
            "parameters": {
                "type": "object",
                "properties": {
                    "settle": {
                        "type": "array",
                        "description": "对每条待结算预言的判定。没有就空数组。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "enum": ids},
                                "outcome": {"type": "string",
                                            "enum": ["confirmed", "refuted",
                                                     "watch"]},
                                "evidence": {"type": "string"},
                            },
                            "required": ["id", "outcome", "evidence"],
                            "additionalProperties": False,
                        },
                    },
                    "pending": {
                        "type": ["object", "null"],
                        "description": "你对下一轮的新预言。没有就给 null。",
                        "properties": {
                            "claim": {"type": "string"},
                            "confidence": {"type": "number"},
                            "predicts": {"type": "string"},
                        },
                        "required": ["claim", "confidence", "predicts"],
                        "additionalProperties": False,
                    },
                    # delta：关系增量。
                    #
                    # 少了这个通道，tool / persona 协议下的关系账本永远
                    # 停在初始值 —— 实测 3 轮后 relation 仍是默认的
                    # {closeness: 0.3, unresolved: [], ...}，
                    # ledger 里 relation_delta 全是 {}。
                    # 而 text 协议是有的，三个协议行为不一致。
                    "delta": {
                        "type": ["object", "null"],
                        "description": "这一轮的关系增量。没有就给 null。",
                        "properties": {
                            "closeness": {"type": "number"},
                            "unresolved": {"type": "array",
                                           "items": {"type": "string"}},
                            "sore_points": {"type": "array",
                                            "items": {"type": "string"}},
                            "inside_jokes": {"type": "array",
                                             "items": {"type": "string"}},
                        },
                        "required": ["closeness", "unresolved",
                                     "sore_points", "inside_jokes"],
                        "additionalProperties": False,
                    },
                },
                "required": ["settle", "pending"],
                "additionalProperties": False,
            },
        },
    }


TAIL_SYSTEM = """你在记录可选读法的后续证据，并决定是否值得新增一个假设。

规则：
- settle：只在用户内容提供明确证据时核对，不要求立即完成。
    confirmed = 用户明确提供所预测的具体内容，必须附证据
    refuted   = 用户明确反驳所预测的内容，必须附证据
    watch     = 无法判断。换题、同话题、短回复和时间经过都不足以结算。
  没有待结算的预言就给空数组 []。
- pending：可选的后续观察假设。必须是可观察的具体行为，不是感受。
  好：「下一句会继续描述今天发生的事」 坏：「他需要被理解」
  读法主要用于生成 say，不是预测考试或人格成长分数。
  大多数轮次给 null；不要为凑学习信号而挂假设，也不要为了清空而结算。"""


def build_tail_messages(*, user_text: str, say: str, pendings: list[dict],
                        history: list[dict]) -> list[dict]:
    lines = []
    for p in pendings:
        lines.append(f"- id={p['id']} 你猜「{p.get('claim', '')}」，"
                     f"预计：{p.get('predicts', '')}")
    block = "\n".join(lines) if lines else "（这一轮没有待结算的预言）"
    return [
        {"role": "system", "content": TAIL_SYSTEM},
        {"role": "user",
         "content": f"[用户这一轮说的]\n{user_text}\n\n"
                    f"[角色本轮回复]\n{say}\n\n"
                    f"[待结算的预言]\n{block}"},
    ]


def parse_tail(data: dict) -> Tail:
    """把第三阶段的工具参数转成 Tail。坏数据一律降级成空，不抛。"""
    tail = Tail()
    for item in data.get("settle") or []:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id", "") or "").strip()
        outcome = str(item.get("outcome", "") or "").strip().lower()
        if not pid or outcome not in ("confirmed", "refuted", "watch"):
            continue
        if outcome == "watch":
            # watch 不结算，但**要记住模型说过 watch** ——
            # 引擎靠它避免退回字面判定。
            tail.watch.append(pid)
            continue
        tail.settle.append(SettleItem(pid, outcome, str(item.get("evidence", "") or "")))
    # delta：关系增量。engine 会再过一遍白名单 + 限额。
    d = data.get("delta")
    if isinstance(d, dict):
        got = {}
        for key in ("closeness", "unresolved", "sore_points", "inside_jokes"):
            if key in d and d[key] not in (None, ""):
                got[key] = d[key]
        if got:
            tail.delta = got

    p = data.get("pending")
    if isinstance(p, dict):
        claim = str(p.get("claim", "") or "").strip()
        predicts = str(p.get("predicts", "") or "").strip()
        if claim and predicts:
            try:
                conf = float(p.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            tail.pending = PendingDraft(claim, max(0.0, min(1.0, conf)), predicts)
    return tail


def to_parsed(inner: Head, say: str) -> ParsedTurn:
    """把两阶段的产物拼成下游认识的 ParsedTurn。

    除了 head 和 say，还要**真的做硬失败检测** ——
    declared / read_aloud 不能写死成空列表（下面有详细原因）。

    tail 给一个空骨架 —— 这一步不产出 settle / pending，
    那是引擎在 phase 2 之后单独补的（见 engine._tool_turn）。
    """
    # declared / read_aloud 必须真的检测 —— 写死空列表会让「硬失败」
    # 在 tool / persona 协议下完全失效。
    # 实测同一句话 "我猜你早就想辞职了。我本来想说你该走了。"：
    #   text   协议 -> declared=2, warnings=[硬失败…]
    #   tool   协议 -> declared=[], read_aloud=[], warnings=[]
    # 也就是 README 的硬约束在结构化协议里等于不存在。
    final_say = safe_say(say)
    declared_hits = detect_declarations(final_say)
    read_hits = detect_read_aloud(final_say)

    errors: list[str] = []
    if not final_say and (say or "").strip():
        errors.append("SAY 被安全闸清空（疑似协议残留）")

    return ParsedTurn(
        head=inner,
        say=final_say,
        tail=Tail(),
        declared=declared_hits,
        read_aloud=read_hits,
        errors=errors,
    )


# ---------------------------------------------------------------- 判定


def displacement_ok(inner: Head, say: str) -> tuple[bool, str]:
    """诊断声称的额外位移是否只是人格基准的字面复述。

    缺基准时返回未发现冲突及“未知”说明，不等于已经验证。
    不判断人格质量，不负责重新生成，语义位移仍需独立检验。
    """
    baseline = (inner.baseline or "").strip()
    say = (say or "").strip()
    if not say:
        return False, "phase 2 没产出任何文本"
    if not baseline:
        return True, "缺少人格基准，不能检查位移（未知，不算已验证）"

    if inner.displace in ("", DISPLACE_NONE):
        return True, ""          # 说了不位移，怎么写都算合规

    # 声称位移：不能是复述。
    # 用 parse._similar（字符集覆盖率 + 长度比），不用 bigram ——
    # bigram 抓不到同义改写：「是不是矫情」→「怀疑自己矫情」重合度很低，
    # 但意思一模一样。这正是第一轮就记录在案的已知盲区。
    if _similar(baseline, say):
        return False, f"声称 {inner.displace} 但 say 基本是 baseline 的复述"
    return True, ""


def parse_inner(data: dict) -> Head:
    """把工具参数转成 Head，顺手做一次防御性清洗。

    有 strict schema 兜底，但不同端点的实现质量参差 ——
    缺字段、类型错、枚举值不认识，都要能活着回来。
    """
    displace = str(data.get("displace", "") or "").strip().lower()
    if displace not in DISPLACE_KINDS:
        displace = DISPLACE_NONE
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Head(
        need=str(data.get("need", "") or "").strip(),
        confidence=max(0.0, min(1.0, conf)),
        direct=str(data.get("direct", "") or "").strip(),
        displace=displace,
        baseline=str(data.get("baseline", "") or "").strip(),
    )
