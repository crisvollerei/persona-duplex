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
        options: SessionConfig | None = None,
    ):
        self.opts = options or SessionConfig()
        self.store = store
        self.engine = TurnEngine(
            cfg, store, llm,
            similarity=similarity if similarity is not None else _default_sim(),
            stream_llm=stream_llm,
        )
        self.consolidator = Consolidator(store)
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

    def say_stream(self, text: str) -> Iterator[str]:
        """流式说一句，逐字吐。返回的迭代器耗尽后，结果在 .result 上。"""
        chunks: list[str] = []
        result_box: list[TurnResult] = []

        def sink(piece: str) -> None:
            chunks.append(piece)

        res = self.say(text, on_delta=sink)
        result_box.append(res)
        yield from chunks

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


def _default_sim():
    from .similarity import DEFAULT
    return DEFAULT
