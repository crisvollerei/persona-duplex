"""相似度模块测试。

这一层是「话题有没有回来」的地基，判错会污染整个校准账本。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.similarity import (                                # noqa: E402
    DEFAULT,
    EmbeddingSimilarity,
    HybridSimilarity,
    bigram_similarity,
    bigrams,
)


class TestBigram(unittest.TestCase):
    def test_same_text_is_one(self):
        self.assertAlmostEqual(bigram_similarity("今天好累啊", "今天好累啊"), 1.0)

    def test_no_overlap_is_zero(self):
        self.assertEqual(bigram_similarity("今天好累啊", "天气不错"), 0.0)

    def test_partial_overlap(self):
        s = bigram_similarity("路上堵了一个半小时", "路上堵吗")
        self.assertGreater(s, 0.3)
        self.assertLess(s, 1.0)

    def test_empty_is_zero(self):
        self.assertEqual(bigram_similarity("", "今天"), 0.0)
        self.assertEqual(bigram_similarity("今天", ""), 0.0)

    def test_english_is_zero(self):
        """中文切词不认识英文 —— 这是已知限制，不是 bug。"""
        self.assertEqual(bigrams("hello world"), set())


class TestEmbedding(unittest.TestCase):
    def test_cosine(self):
        vecs = {
            "a": [1.0, 0.0],
            "b": [1.0, 0.0],
            "c": [0.0, 1.0],
        }
        sim = EmbeddingSimilarity(embed=lambda t: vecs[t])
        self.assertAlmostEqual(sim("a", "b"), 1.0)
        self.assertAlmostEqual(sim("a", "c"), 0.0)

    def test_caches(self):
        calls = []

        def embed(t):
            calls.append(t)
            return [1.0, 0.0]

        sim = EmbeddingSimilarity(embed=embed)
        sim("a", "b")
        sim("a", "b")
        self.assertEqual(len(calls), 2, "同样的文本不该重复 embed")

    def test_zero_vector_safe(self):
        sim = EmbeddingSimilarity(embed=lambda t: [0.0, 0.0])
        self.assertEqual(sim("a", "b"), 0.0)


class TestHybrid(unittest.TestCase):
    def test_takes_max(self):
        """字面重合 0，但语义说 0.9 —— 取最大。"""
        sem = EmbeddingSimilarity(embed=lambda t: [1.0, 0.0])
        h = HybridSimilarity(semantic=sem)
        self.assertAlmostEqual(h("完全不同的字", "another"), 1.0)

    def test_semantic_failure_falls_back(self):
        def boom(t):
            raise RuntimeError("embedding 服务挂了")

        h = HybridSimilarity(semantic=EmbeddingSimilarity(embed=boom))
        score = h("今天好累", "今天好累")
        self.assertAlmostEqual(score, 1.0, msg="embedding 挂了应退回字面")

    def test_no_semantic_is_pure_bigram(self):
        h = HybridSimilarity()
        self.assertAlmostEqual(h("今天好累", "今天好累"), 1.0)


class TestDefault(unittest.TestCase):
    def test_default_is_callable(self):
        self.assertAlmostEqual(DEFAULT("abc", "abc"), 0.0)  # 英文无 bigram
        self.assertAlmostEqual(DEFAULT("今天", "今天"), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
