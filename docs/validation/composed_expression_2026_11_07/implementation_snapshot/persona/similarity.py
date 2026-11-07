"""话题相似度 —— 可插拔。

判定「用户有没有回到那个话题」是这个系统里最容易做错的一步：
太松会把换题当成接住，太紧会把绕一圈回来的话判成断线。

默认实现是汉字 bigram，零依赖但只认字面。真要用请换 embedding：
    from persona.similarity import EmbeddingSimilarity
    sim = EmbeddingSimilarity(embed=my_embed_fn)
    engine = TurnEngine(cfg, store, llm, similarity=sim)
"""
from __future__ import annotations

import math
import re
from typing import Callable, Protocol


class Similarity(Protocol):
    def __call__(self, a: str, b: str) -> float:
        """返回 0–1 的重合度。0 = 完全换题。"""
        ...


def bigrams(text: str) -> set[str]:
    """中文零依赖切词：连续汉字取 2 字窗口。"""
    out: set[str] = set()
    for seg in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        for i in range(len(seg) - 1):
            out.add(seg[i:i + 2])
    return out


def bigram_similarity(a: str, b: str) -> float:
    """默认实现。用 min 做分母，避免长文本把短句的重合度稀释掉。"""
    sa, sb = bigrams(a), bigrams(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / max(1, min(len(sa), len(sb)))


class EmbeddingSimilarity:
    """把任意 embed(text) -> vector 的函数包成 Similarity。

    没有 embedding 服务也能用，只是别指望它理解同义改写。
    """

    def __init__(self, embed: Callable[[str], list[float]], floor: float = 0.0):
        self.embed = embed
        self.floor = floor
        self._cache: dict[str, list[float]] = {}

    def _vec(self, text: str) -> list[float]:
        key = text or ""
        if key not in self._cache:
            self._cache[key] = self.embed(key)
        return self._cache[key]

    def __call__(self, a: str, b: str) -> float:
        va, vb = self._vec(a), self._vec(b)
        if not va or not vb:
            return 0.0
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(y * y for y in vb))
        if not na or not nb:
            return 0.0
        cos = dot / (na * nb)
        return max(0.0, min(1.0, cos))


class HybridSimilarity:
    """字面 + 语义取最大。

    字面能抓住「原话复述」，语义能抓住「换个说法说同一件事」。
    两边都不完美，取最大比取平均稳。
    """

    def __init__(self, semantic: Similarity | None = None):
        self.semantic = semantic

    def __call__(self, a: str, b: str) -> float:
        score = bigram_similarity(a, b)
        if self.semantic is not None:
            try:
                score = max(score, self.semantic(a, b))
            except Exception:
                pass
        return score


DEFAULT = bigram_similarity
