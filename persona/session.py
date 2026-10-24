"""会话生命周期：引擎、后台记忆提炼、账本与退出收尾。

默认前台一次生成三段；记忆在后台异步提炼，不是回复必经的串行调用。
默认后台只记录暂定认识和陪伴经历，不自动改变人格参数。成长策略尚待验证。
"""
from __future__ import annotations

import atexit
import threading
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable, Iterator

from .config import PersonaConfig
from .consolidate import Consolidator
from .engine import TurnEngine, TurnResult
from .store import Store
from .topicjudge import build_judge
from .audit import model_info


@dataclass
class SessionConfig:
    consolidate_interval: float = 300.0   # 后台多久跑一次
    consolidate_on_exit: bool = True      # 退出时补一次
    stream: bool = False                  # 是否用流式
    fidelity_check: bool = False          # persona 实验：额外模型检查
    json_fallback: bool = False           # persona 实验：工具失败后改用 JSON
    memory_mode: str = "live"             # live | frozen (fixed long-term judgments)
    condition: str = ""


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
        cfg.validate()
        if self.opts.memory_mode not in ("live", "frozen"):
            raise ValueError("memory_mode must be live or frozen")
        if not math.isfinite(self.opts.consolidate_interval) or self.opts.consolidate_interval <= 0:
            raise ValueError("consolidate_interval must be finite and positive")
        self.store = store

        # 人格参数：账本里有快照就用快照，没有就用人格配置初始化。
        #
        # 为什么不能直接用 cfg.role.traits 的 anchor 当活状态：
        # 那样重启一次就把跑了几百轮的人格漂移丢了 ——
        # 参数是记忆，不是配置。
        _state = store.traits()
        _prev = store.persona_name()
        _same = _prev == (cfg.role.name or "")
        if _state and (_same or _prev == ""):
            # 同一个人格，或者老账本（没有 persona 字段）。
            # 后者按「同一个人格」处理 —— 否则升级一次就把历史漂移清零。
            store.init_traits(cfg.role.traits_state(), persona=cfg.role.name or "")
            cfg.role.load_traits_state(_state)
        else:
            # 真的换了人格 —— 之前的参数对新人格没有意义，重新初始化。
            # 不做这一步的话，实测守岸人跑过之后用同一目录载入 ache，
            # ache 的 warmth 会拿到 0.45（被它的 drift 夹住的守岸人 0.70），
            # 而不是自己的 0.30。
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
            fidelity_check=self.opts.fidelity_check,
            json_fallback=self.opts.json_fallback,
            audit_options={"session": asdict(self.opts), "memory_model": model_info(consolidate_llm),
                           "judge_model": model_info(fast_llm)},
        )
        if self.opts.memory_mode == "frozen":
            self.engine.memory_snapshot = deepcopy(store.user_judgments())
        # 慢层固化也走模型：没有 llm 时它只会数 need 出现几次，
        # 那不是「判断」，只是计数。给了模型才可能提炼出有洞察的判断。
        self.consolidator = Consolidator(store, llm=consolidate_llm,
                                        audit_context={"run_id": self.engine.run["run_id"]})
        self._closed = False
        self._active_turn = False
        self._lock = threading.RLock()

        if self.opts.memory_mode == "live":
            self.consolidator.start(self.opts.consolidate_interval)
        atexit.register(self.close)

    # ------------------------------------------------------------ 对话

    def say(self, text: str, on_delta: Callable[[str], None] | None = None) -> TurnResult:
        """说一句，拿回一轮结果。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("会话已关闭")
            if self._active_turn:
                raise RuntimeError("不能在回复回调内重入会话")
            self._active_turn = True
            try:
                return self.engine.turn(text, on_delta=on_delta)
            finally:
                self._active_turn = False

    def say_stream(self, text: str) -> "_StreamedTurn":
        """迭代 SAY；此迭代包装器先完成一轮，实时输出请使用 say 的 on_delta。

        返回的对象迭代耗尽后，本轮结果在 .result 上。

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
        if self.opts.memory_mode == "frozen":
            raise RuntimeError("冻结记忆条件下不能固化；请在独立的 live 会话中处理积压")
        return self.consolidator.run_once()

    def close(self) -> None:
        """关掉会话。幂等。"""
        with self._lock:
            if self._closed:
                return
            if self._active_turn:
                raise RuntimeError("不能在回复回调内关闭会话；请在本轮结束后关闭")
            self._closed = True
            try:
                self.consolidator.stop()
            finally:
                atexit.unregister(self.close)
            if self.opts.consolidate_on_exit and self.opts.memory_mode == "live":
                try:
                    self.consolidator.run_once()
                except Exception as e:
                    # 不能完全静默 —— 但 close 是收尾路径，也不该抛出去
                    # 盖掉调用方真正的错误。记到 stderr 和账本。
                    import sys as _sys
                    print(f"[persona] 退出时固化失败：{e}", file=_sys.stderr)
                    try:
                        self.store.append("consolidate_error", {
                            "where": "close", "error": f"{type(e).__name__}: {e}"[:300],
                        })
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
