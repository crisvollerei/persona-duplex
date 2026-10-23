"""会话：把引擎、后台线程、账本串成一个东西。

这是唯一需要关心「生命周期」的地方：
    with Session(cfg, store, llm) as s:
        s.say("今天好累啊")

退出 with 时后台会跑最后一次固化 —— 不然积压永远留在 buffer 里，
慢层就一直是空的，而慢层才是「成长」所在。
"""
from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass
from typing import Callable, Iterator

from .config import PersonaConfig
from .consolidate import Consolidator
from .engine import TurnEngine, TurnResult
from .store import Store
from .topicjudge import build_judge


@dataclass
class SessionConfig:
    consolidate_interval: float = 300.0   # 后台多久跑一次
    consolidate_on_exit: bool = True      # 退出时补一次
    stream: bool = False                  # 是否用流式


class Session:
    """一次会话。关闭时保证后台跑完。"""

    def __init__(
        self,
        cfg: PersonaConfig,
        store: Store,
        llm: Callable[[list[dict]], str],
        *,
        stream_llm: Callable[[list[dict]], Iterator[str]] | None = None,
        similarity=None,
        judge=None,
        fast_llm: Callable[[list[dict]], str] | None = None,
        consolidate_llm: Callable[[str], str] | None = None,
        tool_llm=None,
        protocol: str = "text",   # text | tool | auto | persona
        options: SessionConfig | None = None,
    ):
        self.opts = options or SessionConfig()
        self.store = store

        # 人格参数：账本里有快照就用快照，没有就用人格配置初始化。
        #
        # 为什么不能直接用 cfg.role.traits 的 anchor 当活状态：
        # 那样重启一次就把跑了几百轮的人格漂移丢了 ——
        # 参数是记忆，不是配置。
        _state = store.traits()
        _same = store.persona_name() == (cfg.role.name or "")
        if _state and _same:
            cfg.role.load_traits_state(_state)
        else:
            # 新人格，或账本里记的人格和当前配置不一致 —— 重新初始化。
            # 不这么做的话，换人格会拿到旧人格的参数：实测守岸人跑过之后
            # 用同一目录载入 ache，warmth 拿到 0.45（被 ache 的 drift 夹住的
            # 守岸人值 0.70），而不是 ache 自己的 0.30。
            store.init_traits(cfg.role.traits_state(), persona=cfg.role.name or "")

        self.engine = TurnEngine(
            cfg, store, llm,
            similarity=similarity if similarity is not None else _default_sim(),
            stream_llm=stream_llm,
            judge=judge if judge is not None else build_judge(
                cfg, fast_llm, similarity=similarity
            ),
            protocol=protocol,
            tool_llm=tool_llm,
        )
        # 慢层固化也走模型：没有 llm 时它只会数 need 出现几次，
        # 那不是「判断」，只是计数。给了模型才可能提炼出有洞察的判断。
        self.consolidator = Consolidator(store, llm=consolidate_llm)
        self._closed = False
        self._lock = threading.Lock()

        self.consolidator.start(self.opts.consolidate_interval)
        atexit.register(self.close)

    # ------------------------------------------------------------ 对话

    def say(self, text: str, on_delta: Callable[[str], None] | None = None) -> TurnResult:
        """说一句，拿回一轮结果。"""
        if self._closed:
            raise RuntimeError("会话已关闭")
        res = self.engine.turn(text, on_delta=on_delta)
        self.engine.update_style_baseline(text)
        return res

    def say_stream(self, text: str) -> "_StreamedTurn":
        """流式说一句，逐片吐。返回的对象迭代耗尽后，本轮结果在 .result 上。

        返回的不是 generator 而是实现了迭代协议的对象 —— generator 上
        挂不了属性，而调用方需要拿到这一轮的 TurnResult（结算了什么、
        有没有受伤）才能做后续决策。
        """
        return _StreamedTurn(self, text)

    def chat(self, text: str, on_delta: Callable[[str], None] | None = None) -> str:
        return self.say(text, on_delta=on_delta).say

    # ------------------------------------------------------------ 后台

    def consolidate_now(self) -> dict:
        """手动催一次后台固化。测试和 CLI 用。"""
        return self.consolidator.run_once()

    def close(self) -> None:
        """关掉会话。幂等。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.consolidator.stop()
        finally:
            if self.opts.consolidate_on_exit:
                try:
                    self.consolidator.run_once()
                except Exception:
                    pass

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _StreamedTurn:
    """一次流式轮次。迭代时逐片吐 SAY，结束后 .result 可用。"""

    def __init__(self, session: "Session", text: str):
        self._session = session
        self._text = text
        self._pieces: list[str] = []
        self._started = False
        self.result: TurnResult | None = None

    def __iter__(self) -> "_StreamedTurn":
        return self

    def __next__(self) -> str:
        if not self._started:
            self._started = True
            self.result = self._session.say(self._text, on_delta=self._pieces.append)
        if self._pieces:
            return self._pieces.pop(0)
        raise StopIteration


def _default_sim():
    from .similarity import DEFAULT
    return DEFAULT
