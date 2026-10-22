"""语义话题判定。

为什么需要它：bigram 只认字面。用户说「投了两家简历」而预言写的是
「会说出某个具体方案」，字面重合接近零，但语义上完全对上了。

为什么异步：这个判定卡在每一轮的关键路径上。同步调等于每轮多一次网络
往返。但主生成本来就要 1-3 秒 —— 只要在生成的同时并行跑，它就永远
不会成为瓶颈。

再加一条 deadline 兜底：预算按「从这一轮开始算」计。如果生成已经花掉
了全部预算，collect 会立刻返回，改用字面判定，绝不额外加延迟。
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

RETURNED = "returned"
LEFT = "left"
UNCLEAR = "unclear"

# 字面判定的阈值。跟 settle.py 里的一致。
OVERLAP_KEEP = 0.18
OVERLAP_DROP = 0.05


@dataclass
class TopicVerdict:
    verdict: str = UNCLEAR
    why: str = ""

    @property
    def returned(self) -> bool:
        return self.verdict == RETURNED

    @property
    def left(self) -> bool:
        return self.verdict == LEFT


@dataclass
class JudgeRequest:
    """一次判定所需的全部材料。"""

    user_text: str
    prev_ai_say: str = ""
    prev_user_text: str = ""
    pendings: list[dict] = field(default_factory=list)
    extra_refs: dict[str, list[str]] = field(default_factory=dict)


class TopicJudge(Protocol):
    name: str

    def judge(self, req: JudgeRequest) -> dict[str, TopicVerdict]:
        ...


# ---------------------------------------------------------------- 字面兜底


class BigramJudge:
    """零依赖兜底。LLM 挂了、超时了、没配，都退回这里。"""

    name = "bigram"

    def __init__(self, similarity=None):
        self.similarity = similarity

    def judge(self, req: JudgeRequest) -> dict[str, TopicVerdict]:
        from .settle import topic_overlap

        out: dict[str, TopicVerdict] = {}
        for p in req.pendings:
            refs = [req.prev_ai_say, req.prev_user_text,
                    p.get("claim", ""), p.get("predicts", "")]
            refs += list(req.extra_refs.get(p["id"], []))
            scores = [topic_overlap(req.user_text, r, self.similarity)
                      for r in refs if r]
            overlap = max(scores) if scores else 0.0

            if overlap >= OVERLAP_KEEP:
                out[p["id"]] = TopicVerdict(RETURNED, f"字面重合 {overlap:.2f}")
            elif overlap < OVERLAP_DROP:
                # 字面重合低 ≠ 预言被否证。它也可能只是「还没说到」。
                # 退回 UNCLEAR 而不是 LEFT —— 让启发式判 WATCH 而不是 REFUTED。
                # 这是修命中率的关键：以前字面兜底把「没提」当成「说反了」。
                out[p["id"]] = TopicVerdict(UNCLEAR, f"字面重合 {overlap:.2f}")
            else:
                out[p["id"]] = TopicVerdict(UNCLEAR, f"字面重合 {overlap:.2f}")
        return out


# ---------------------------------------------------------------- LLM 判定


_PROMPT = """判断：用户这一句，是否**证实或否证**了下面这几条预言。只输出 JSON。

[AI 上一轮说的] {prev_ai}
[用户这一轮说的] {user}

待判断的预言：
{pending_lines}

对每一条，问自己一个问题：**用户这句话说出来的事，和这条预言描述的一致吗？**

verdict 三选一：
  returned —— 一致。用户说的正是预言预计会出现的那个东西。
  left     —— 不一致。用户说的事和预言预计的明显是两回事。
  unclear  —— 用户这一句根本没涉及这条预言说的内容（换题、太短、
              只回应了别的部分）。这种不是「不一致」，是「还没发生」。

关键区别 —— 这一条最容易判错：
  · 用户换了个话题 → unclear，不是 left。
    「话题变了」不等于「预言错了」；预言可能只是还没被验证。
  · 用户说了别的、但和预言不矛盾 → unclear。
  · 只有用户说的事**明确指向相反**，才是 left。

例：
  预言：「对方会报一个具体的入睡时间」
  用户：「我最近老失眠」        -> unclear（还没说时间，不代表不说）
  用户：「两三点才睡着」        -> returned（就是这么个时间）
  用户：「我不困，开始干活了」  -> left（明确顶回来）

格式：[{{"id":"p_0001","verdict":"returned","why":"十五字以内"}}]"""


def _clip(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


class LLMJudge:
    """让模型判断「有没有回到话题」。

    刻意用不思考模式、极短输出、temperature=0 —— 这是个分类任务，
    不需要推理链，要的是稳定和快。
    """

    name = "llm"

    def __init__(self, llm: Callable[..., str]):
        self.llm = llm

    def _call(self, messages: list[dict], *, max_tokens: int) -> str:
        """尽量把 max_tokens 传下去。

        openai_compatible 产出的闭包接受 (messages, *, max_tokens=...)；
        但测试和自定义判定器常常只接受 messages 一个参数。两种都要能跑。
        """
        try:
            return self.llm(messages, max_tokens=max_tokens)
        except TypeError:
            return self.llm(messages)

    # 每条预言大约要 40 个 token 的 JSON（id + verdict + why）。
    # 固定 120 的预算只够判 2-3 条；挂到 8 条时输出会被截断，
    # JSON 尾部断掉 → 整批解析失败 → 全部退回「模型输出无法解析」。
    # 实测过：挂 8 条时 max_tokens=120 的输出停在 '"id":"p' 半截。
    _TOKENS_PER_PENDING = 48
    _TOKENS_BASE = 80

    def _budget(self, n: int) -> int:
        return self._TOKENS_BASE + self._TOKENS_PER_PENDING * max(1, n)

    def judge(self, req: JudgeRequest) -> dict[str, TopicVerdict]:
        if not req.pendings:
            return {}
        lines = "\n".join(
            f"- {p['id']}：猜「{_clip(p.get('claim', ''), 60)}」，"
            f"预计「{_clip(p.get('predicts', ''), 60)}」"
            for p in req.pendings
        )
        prompt = _PROMPT.format(
            prev_ai=_clip(req.prev_ai_say) or "（无）",
            user=_clip(req.user_text) or "（无）",
            pending_lines=lines,
        )
        raw = self._call([{"role": "user", "content": prompt}],
                         max_tokens=self._budget(len(req.pendings)))
        return self._parse(raw, req)

    def _parse(self, raw: str, req: JudgeRequest) -> dict[str, TopicVerdict]:
        valid = {p["id"] for p in req.pendings}
        out: dict[str, TopicVerdict] = {}

        text = raw or ""
        i, j = text.find("["), text.rfind("]")
        if i == -1 or j <= i:
            # 模型没给数组 —— 当它判不了，而不是当它说「没回来」
            return {pid: TopicVerdict(UNCLEAR, "模型输出无法解析") for pid in valid}

        try:
            items = json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return {pid: TopicVerdict(UNCLEAR, "JSON 解析失败") for pid in valid}

        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            pid = str(it.get("id", "")).strip()
            verdict = str(it.get("verdict", "")).strip().lower()
            if pid not in valid or verdict not in (RETURNED, LEFT, UNCLEAR):
                continue
            out[pid] = TopicVerdict(verdict, _clip(str(it.get("why", "")), 40))

        for pid in valid - out.keys():
            out[pid] = TopicVerdict(UNCLEAR, "模型漏判")
        return out


# ---------------------------------------------------------------- 异步包装


@dataclass
class JudgeHandle:
    """一次在飞的判定。

    关键设计：**collect 不会等到底**。
    它只给一个很短的宽限期（grace），拿不到就立刻用字面兜底。
    因为判定慢于主生成时，等待等于把判定变成瓶颈 —— 那就违背了
    「异步不拖慢对话」的初衷。

    没拿到的判定不会浪费：预言还挂着，下一轮会被重新判定。
    """

    req: JudgeRequest
    grace: float
    started: float
    fallback: TopicJudge
    finished_at: float | None = None
    _thread: threading.Thread | None = None
    _result: dict[str, TopicVerdict] | None = None
    _error: BaseException | None = None

    def collect(self) -> tuple[dict[str, TopicVerdict], str]:
        """返回 (verdicts, source)。source: llm / late / bigram。

        late = 判定还在飞，这轮先用字面兜底，预言继续挂着。
        """
        remaining = self.grace - (time.monotonic() - self.started)
        if remaining > 0 and self._thread is not None:
            self._thread.join(timeout=remaining)

        if self._error is None and self._result is not None:
            return self._result, "llm"
        return self.fallback.judge(self.req), "late"


class AsyncJudge:
    """把判定丢到后台线程，主链路只负责「发射」和「有就拿」。

    grace 默认 0.35s：比一次网络往返短，比人感知延迟也短。
    主生成通常 1-3 秒，所以判定基本都能在生成期间跑完 —— 那时
    collect 是零等待的。
    """

    def __init__(self, inner: TopicJudge, *, grace: float = 0.35,
                 fallback: TopicJudge | None = None):
        self.inner = inner
        self.grace = grace
        self.fallback = fallback or BigramJudge()

    def submit(self, req: JudgeRequest) -> JudgeHandle:
        handle = JudgeHandle(
            req=req, grace=self.grace,
            started=time.monotonic(), fallback=self.fallback,
        )
        inner = self.inner

        def work() -> None:
            try:
                # 允许传普通函数，也允许传带 .judge 的对象。
                # 前者写测试和临时替换时方便得多。
                fn = getattr(inner, "judge", None)
                handle._result = fn(req) if callable(fn) else inner(req)
            except BaseException as e:      # 网络错误、解析错误，一律兜底
                handle._error = e
            finally:
                # 记录判定器真正完成的时刻。整轮耗时不能拿来比 grace：
                # 主生成慢 ≠ 判定慢，两者必须分开量。
                handle.finished_at = time.monotonic()

        t = threading.Thread(target=work, daemon=True, name="topic-judge")
        handle._thread = t
        t.start()
        return handle


# ---------------------------------------------------------------- 组装


def build_judge(cfg, fast_llm: Callable[[list[dict]], str] | None,
                *, similarity=None) -> AsyncJudge | None:
    """按配置组装一个判定器。没配就返回 None（引擎会退回纯字面）。"""
    if not getattr(cfg, "judge_enabled", True):
        return None
    fallback = BigramJudge(similarity=similarity)
    if fast_llm is None:
        return None
    return AsyncJudge(LLMJudge(fast_llm), grace=cfg.judge_grace,
                      fallback=fallback)
