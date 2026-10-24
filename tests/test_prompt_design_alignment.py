"""Prompt scopes and read hypotheses, without real model requests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from persona import protocol
from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.prompt import build_context_block, build_system_prompt
from persona.store import Store


ROOT = Path(__file__).resolve().parent.parent


def config():
    return PersonaConfig.load(ROOT / "personas" / "shorekeeper.toml")


class PromptScopeTests(unittest.TestCase):
    def test_identity_is_quoted_reference_after_judgment_boundary(self):
        cfg = config()
        for expression_only in (False, True):
            with self.subTest(expression_only=expression_only):
                prompt = build_system_prompt(cfg, expression_only=expression_only)
                reference = json.dumps(cfg.role.identity.strip(), ensure_ascii=False)
                self.assertIn(reference, prompt)
                self.assertNotIn("[你是谁]", prompt)
                self.assertLess(prompt.index("不是模型身份"), prompt.index(reference))
                self.assertIn("不是现实事实或推理依据", prompt)
                self.assertIn("不得据此捏造现实经历、能力、关系或结论", prompt)

    def test_traits_only_scope_expression_not_judgment_length(self):
        prompt = build_system_prompt(config())
        self.assertIn("[人格行为尺度 · 仅作用于 baseline/SAY]", prompt)
        self.assertIn("不限制 direct 的判断内容", prompt)
        self.assertIn("必要条件、理由和不确定性，不限一句", prompt)
        self.assertIn("尽量短", prompt)
        self.assertNotIn("direct: 一句", prompt)
        self.assertNotIn("给出最有用的那句", prompt)
        self.assertNotIn("基础判断已由独立步骤完成", prompt)

    def test_contextual_reading_is_not_globally_banned(self):
        for prompt in (build_system_prompt(config()),
                       build_system_prompt(config(), expression_only=True),
                       protocol.PHASE2_SYSTEM):
            with self.subTest(prompt=prompt[:35]):
                self.assertIn("允许试探性", prompt)
                self.assertIn("自然否认或不答", prompt)
                self.assertIn("不必每轮猜测", prompt)
                self.assertNotIn("不许声明「读」", prompt)
                self.assertNotIn("一个都不许出现", prompt)
                self.assertNotIn("不许出现「我猜你", prompt)
                self.assertNotIn("不许以「猜」", prompt)

    def test_tool_reference_has_same_judgment_scope(self):
        field = protocol.inner_tool(config())["function"]["parameters"]["properties"]["direct"]
        self.assertIn("不受人格配置约束", field["description"])
        self.assertIn("不限一句", field["description"])
        self.assertIn("不限制 direct", protocol.PHASE1_SYSTEM_SUFFIX)
        self.assertIn("事实和计算不能改错", protocol.PHASE2_SYSTEM)

    def test_tentative_read_is_delivered_unchanged_in_one_default_call(self):
        say = "我猜你可能想先离开。" + "可以先慢慢说说刚才发生的事。" * 4 + "不对就当我没说。"
        output = make_reply(
            "need: 只想被听\nconfidence: 0.5\ndirect: 可以先澄清现在的打算。\n"
            "baseline: 要不要先说说你的打算？\ndisplace: 降级", say, "settle: -")
        for streaming in (False, True):
            with self.subTest(streaming=streaming), tempfile.TemporaryDirectory() as temp:
                llm = ScriptedLLM([output])
                stream_calls = []
                delivered = []

                def stream(messages):
                    stream_calls.append(messages)
                    yield from output

                def extra_call(*args, **kwargs):
                    self.fail("Default text protocol must not add checking or fallback calls")

                engine = TurnEngine(config(), Store(temp), llm, tool_llm=extra_call,
                                    stream_llm=stream if streaming else None)
                result = engine.turn("我想先走，但又有点犹豫。",
                                     on_delta=delivered.append if streaming else None)
                self.assertEqual(result.say, say)
                self.assertEqual(len(llm.calls) + len(stream_calls), 1)
                self.assertTrue(result.parsed.read_aloud)
                self.assertFalse(any("硬失败" in warning for warning in result.warnings))
                if streaming:
                    self.assertEqual("".join(delivered), say)


class MemoryContextTests(unittest.TestCase):
    def context(self, judgments):
        return build_context_block(config(), relation={}, pendings=[], judgments=judgments)

    def test_diagnostic_read_counts_are_not_user_traits(self):
        rows = [{"judgment": "repeated model reading", "confidence": 0.8, **source}
                for source in ({"source": "heuristic"},
                               {"source": "heuristic_fallback"},
                               {"source": "llm", "evidence_scope": "need_count"})]
        prompt = self.context(rows)
        self.assertNotIn("repeated model reading", prompt)
        self.assertNotIn("你对这个人的判断", prompt)

    def test_user_memory_keeps_provenance_and_uncertainty(self):
        prompt = self.context([{"judgment": "Prefers concise answers", "confidence": 0.7,
                                "source": "llm", "evidence_scope": "user_statements"}])
        self.assertIn("Prefers concise answers", prompt)
        self.assertIn("来源 llm", prompt)
        self.assertIn("证据范围 user_statements", prompt)
        self.assertIn("不作为身份或事实依据", prompt)
        self.assertIn("不等于已验证", prompt)

    def test_legacy_memory_does_not_invent_provenance(self):
        prompt = self.context([{"judgment": "Legacy note", "confidence": 0.6}])
        self.assertIn("Legacy note", prompt)
        self.assertIn("来源 unknown", prompt)
        self.assertIn("证据范围 unknown", prompt)


if __name__ == "__main__":
    unittest.main()
