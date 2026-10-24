"""基础判断与人格表达的输入、回退和交付边界，不评价真实模型效果。"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from persona import protocol
from persona.cli import _make_session, main
from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.parse import Head
from persona.prompt import build_history_entry, build_system_prompt
from persona.session import Session, SessionConfig
from persona.store import Store


ROOT = Path(__file__).resolve().parent.parent
RAW = {"need": "核对计算", "confidence": 0.8, "raw": "2 加 2 等于 4。",
       "constraints": ["结果是 4，不能改成 5。"]}
GOOD = make_reply("baseline: 是4。\ndisplace: none", "是4。慢慢算，我在。")


def cfg(name="default"):
    return PersonaConfig.load(ROOT / "personas" / f"{name}.toml")


class Stages:
    def __init__(self, *, raw=None, check=None, tail=None):
        self.raw = copy.deepcopy(RAW if raw is None else raw)
        self.check = ({"verdict": "consistent", "reason": "结果和必要约束保留"}
                      if check is None else check)
        self.tail = {"settle": [], "pending": None} if tail is None else tail
        self.calls = []
        self.before_check = lambda: None

    def __call__(self, messages, *, tool, **kwargs):
        name = tool["function"]["name"]
        self.calls.append((name, copy.deepcopy(messages), copy.deepcopy(tool)))
        if name == "check_fidelity":
            self.before_check()
            result = self.check
        else:
            result = self.raw if name == "raw_reply" else self.tail
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)

    def engine(self, *, stages=None, llm=None, **kwargs):
        kwargs.setdefault("protocol", "persona")
        kwargs.setdefault("fidelity_check", True)
        return TurnEngine(cfg("shorekeeper"), self.store,
                          llm if llm is not None else ScriptedLLM([GOOD]),
                          tool_llm=Stages() if stages is None else stages, **kwargs)

    def assert_not_delivered(self, emitted):
        self.assertEqual(emitted, [])
        self.assertFalse(any(row["kind"] == "turn" for row in self.store.read_ledger()))
        self.assertEqual(self.store.pendings(), [])
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])
        self.assertTrue(any(row["kind"] == "generation_failure"
                            for row in self.store.read_ledger()))

    def test_engine_and_session_default_to_single_call_protocol(self):
        engine = TurnEngine(cfg(), self.store, ScriptedLLM([]))
        self.assertEqual(engine.protocol, "text")
        self.assertFalse(engine.fidelity_check)
        self.assertFalse(engine.json_fallback)
        with Session(cfg(), self.store, ScriptedLLM([]), options=SessionConfig(
                consolidate_on_exit=False)) as session:
            self.assertEqual(session.engine.protocol, "text")
            self.assertFalse(session.engine.fidelity_check)
            self.assertFalse(session.engine.json_fallback)

    def test_cli_defaults_to_single_call_protocol(self):
        with patch("persona.cli.cmd_chat", return_value=0) as chat:
            self.assertEqual(main(["chat"]), 0)
        self.assertEqual(chat.call_args.args[0].protocol, "text")
        self.assertFalse(chat.call_args.args[0].check_fidelity)
        self.assertFalse(chat.call_args.args[0].json_fallback)

    def test_cli_check_and_fallback_are_separate_opt_ins(self):
        for flags, expected in ((["--check-fidelity"], (True, False)),
                                (["--json-fallback"], (False, True))):
            with patch("persona.cli.cmd_chat", return_value=0) as chat:
                main(["chat", "--protocol", "persona", *flags])
            args = chat.call_args.args[0]
            self.assertEqual((args.check_fidelity, args.json_fallback), expected)

    def test_session_options_reach_engine(self):
        with Session(cfg(), self.store, ScriptedLLM([]), protocol="persona", options=SessionConfig(
                consolidate_on_exit=False, fidelity_check=True, json_fallback=True)) as session:
            self.assertTrue(session.engine.fidelity_check)
            self.assertTrue(session.engine.json_fallback)

    def test_extra_stage_options_require_explicit_persona_protocol(self):
        for kwargs, flag in (({"fidelity_check": True}, "--check-fidelity"),
                             ({"json_fallback": True}, "--json-fallback")):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "persona"):
                    TurnEngine(cfg(), self.store, ScriptedLLM([]), **kwargs)
                with patch("sys.stderr"), self.assertRaises(SystemExit) as err:
                    main(["chat", flag])
                self.assertEqual(err.exception.code, 2)

    def test_default_generates_direct_say_and_pending_in_one_call(self):
        raw = make_reply(
            "need: 核对计算\nconfidence: 0.8\ndirect: 2加2等于4\n"
            "baseline: 是4。\ndisplace: none", "是4。慢慢算，我在。",
            "settle: -\npending:\n  claim: 用户在核对计算\n  confidence: 0.6\n"
            "  predicts: 用户会继续问计算过程")
        llm = ScriptedLLM([raw])
        stages = Stages(raw=AssertionError("默认不能调用额外判断工具"))
        result = TurnEngine(cfg(), self.store, llm, tool_llm=stages).turn("2+2？")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(stages.calls, [])
        self.assertEqual(result.parsed.head.direct, "2加2等于4")
        self.assertEqual(result.say, "是4。慢慢算，我在。")
        self.assertEqual(result.pending_added["claim"], "用户在核对计算")
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))), 1)
        turn = next(r for r in self.store.read_ledger() if r["kind"] == "turn")
        self.assertEqual(turn["protocol"], "text")
        self.assertNotIn("persona", turn["probe"])

    def test_default_keeps_displacement_without_requiring_advice_restatement(self):
        candidate = make_reply(
            "need: 想被拦住\nconfidence: 0.7\ndirect: 可以先明确后续安排再决定。\n"
            "baseline: 我觉得你先别急着辞职。\ndisplace: 降级",
            "真辞了，你最怕什么？", "settle: -")
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                llm = ScriptedLLM([candidate])
                streams = []
                emitted = []

                def stream(messages):
                    streams.append(messages)
                    yield from candidate

                stages = Stages(raw=AssertionError("默认不应审查或重写位移"))
                engine = TurnEngine(cfg(), self.store, llm, tool_llm=stages,
                                    stream_llm=stream if streaming else None)
                result = engine.turn("我想辞职", on_delta=emitted.append if streaming else None)
                self.assertEqual(result.say, "真辞了，你最怕什么？")
                self.assertEqual(result.parsed.head.displace, "降级")
                self.assertTrue(result.displaced)
                self.assertEqual(len(llm.calls) + len(streams), 1)
                self.assertEqual(stages.calls, [])
                if streaming:
                    self.assertEqual("".join(emitted), result.say)

    def test_shared_prompt_separates_factual_correctness_from_expression_choice(self):
        for expression_only in (False, True):
            with self.subTest(expression_only=expression_only):
                prompt = build_system_prompt(cfg(), expression_only=expression_only)
                self.assertIn("不要求把 direct 的内部判断、建议、理由逐项说出", prompt)
                self.assertIn("允许降为问题、转移、截断和留白", prompt)
                self.assertIn("事实和计算不能改错", prompt)
                self.assertIn("一轮至多一种", prompt)
                self.assertIn("无额外位移时，say 仍然完整保留人格", prompt)
                self.assertNotIn("你在扮演", prompt)

    def test_persona_without_opt_in_does_not_call_checker_or_rewrite_candidate(self):
        stages = Stages(check=AssertionError("默认不能调用额外审查"))
        candidate = make_reply("baseline: 是5\ndisplace: none", "是5。")
        engine = TurnEngine(cfg(), self.store, ScriptedLLM([candidate]),
                            protocol="persona", tool_llm=stages)
        result = engine.turn("2+2？")
        self.assertEqual(result.say, "是5。")
        self.assertEqual([c[0] for c in stages.calls], ["raw_reply", "settle_turn"])
        turn = next(r for r in self.store.read_ledger() if r["kind"] == "turn")
        self.assertFalse(turn["probe"]["persona"]["fidelity_enabled"])
        self.assertNotIn("fidelity", turn["probe"]["persona"])

    def test_persona_tool_failure_without_opt_in_never_calls_text_model(self):
        stages = Stages(raw=RuntimeError("工具失败"))
        llm = ScriptedLLM([json.dumps(RAW)])
        engine = TurnEngine(cfg(), self.store, llm, protocol="persona", tool_llm=stages)
        with self.assertRaisesRegex(RuntimeError, "基础判断失败"):
            engine.turn("2+2？")
        self.assertEqual(len(stages.calls), 1)
        self.assertEqual(llm.calls, [])

    def test_persona_without_checker_streams_before_full_candidate(self):
        stages = Stages()
        emitted = []

        def chunks(messages):
            yield GOOD
            self.assertTrue("".join(emitted).strip())

        engine = TurnEngine(cfg(), self.store, ScriptedLLM([]), tool_llm=stages,
                            protocol="persona", stream_llm=chunks)
        result = engine.turn("2+2？", on_delta=emitted.append)
        self.assertEqual("".join(emitted), result.say)
        self.assertEqual([c[0] for c in stages.calls], ["raw_reply", "settle_turn"])

    def test_persona_stream_failure_before_output_does_not_retry(self):
        def interrupted(messages):
            yield "<<<HEAD>>>"
            raise RuntimeError("断线")

        llm = ScriptedLLM([GOOD])
        engine = TurnEngine(cfg(), self.store, llm, protocol="persona",
                            tool_llm=Stages(), stream_llm=interrupted)
        emitted = []
        with self.assertRaisesRegex(RuntimeError, "人格表达失败"):
            engine.turn("2+2？", on_delta=emitted.append)
        self.assertEqual(emitted, [])
        self.assertEqual(llm.calls, [])

    def test_persona_stream_failure_keeps_only_delivered_partial_without_tail(self):
        def interrupted(messages):
            yield GOOD
            raise RuntimeError("断线")

        stages = Stages()
        llm = ScriptedLLM(["不允许重来"])
        engine = TurnEngine(cfg(), self.store, llm, protocol="persona",
                            tool_llm=stages, stream_llm=interrupted)
        emitted = []
        result = engine.turn("2+2？", on_delta=emitted.append)
        self.assertEqual("".join(emitted), result.say)
        self.assertEqual(llm.calls, [])
        self.assertEqual([c[0] for c in stages.calls], ["raw_reply"])
        self.assertTrue(result.warnings)
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])

    def test_explicit_persona_without_tool_uses_json_directly_not_as_retry(self):
        llm = ScriptedLLM([json.dumps(RAW), GOOD, json.dumps({"settle": [], "pending": None})])
        result = TurnEngine(cfg(), self.store, llm, protocol="persona").turn("2+2？")
        self.assertTrue(result.say)
        self.assertEqual(len(llm.calls), 3)
        self.assertFalse(result.warnings)

    def test_identical_raw_request_across_personas(self):
        requests = []
        for name in ("shorekeeper", "ache", "service"):
            stages = Stages()
            engine = TurnEngine(cfg(name), self.store, ScriptedLLM([GOOD]),
                                protocol="persona", tool_llm=stages)
            engine.turn("2+2是多少？")
            requests.append(stages.calls[0])
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[1], requests[2])

    def test_raw_schema_never_reads_persona_config(self):
        class Unreadable:
            def __getattribute__(self, name):
                raise AssertionError("读取了人格配置")

        self.assertEqual(protocol.raw_tool(Unreadable()), protocol.raw_tool())

    def test_raw_prompt_preserves_reasoning_and_uncertainty(self):
        self.assertIn("不限制为一两句", protocol.RAW_SYSTEM)
        self.assertIn("必要时请求澄清", protocol.RAW_SYSTEM)
        self.assertIn("不强制把聊天变成解决方案", protocol.RAW_SYSTEM)
        self.assertNotIn("不表达自己的感受、立场、态度", protocol.RAW_SYSTEM)

    def test_persona_prompt_does_not_ask_for_second_judgment(self):
        messages = protocol.build_persona_messages(
            cfg("shorekeeper"), protocol.parse_raw(RAW), [], "2+2？", "熟悉程度：0.3",
            first_turn=True)
        self.assertIn("守岸人", messages[0]["content"])
        self.assertIn("正确性优先", messages[0]["content"])
        self.assertNotIn("direct: 一句", messages[0]["content"])
        self.assertNotIn("confidence: 对读法", messages[0]["content"])
        data = json.loads(messages[-1]["content"].split("\n", 1)[1])
        self.assertEqual(data["constraints"], RAW["constraints"])

    def test_persona_constraints_are_fact_boundaries_not_mandatory_speech(self):
        self.assertIn("不把可选的建议、内部人际读法", protocol.RAW_SYSTEM)
        self.assertIn("不是必须逐项外显的台词清单", protocol.PERSONA_PHASE2_SUFFIX)
        self.assertIn("不能仅因没有复述某个内部建议而拒绝", protocol.FIDELITY_SYSTEM)
        self.assertNotIn("建议方向", protocol.FIDELITY_SYSTEM)

    def test_explicit_checker_can_accept_unspoken_advice_and_question_displacement(self):
        stages = Stages(raw={
            "need": "想被拦住", "confidence": 0.7,
            "raw": "可以先明确后续安排再决定。", "constraints": [],
        }, check={"verdict": "consistent", "reason": "劝阻降为问题，未作相反事实断言"})
        candidate = make_reply("baseline: 我觉得你先别急着辞职。\ndisplace: 降级",
                               "真辞了，你最怕什么？")
        result = self.engine(stages=stages, llm=ScriptedLLM([candidate])).turn("我想辞职")
        self.assertEqual(result.say, "真辞了，你最怕什么？")
        self.assertEqual(result.parsed.head.direct, stages.raw["raw"])
        self.assertTrue(result.displaced)
        self.assertEqual([c[0] for c in stages.calls],
                         ["raw_reply", "check_fidelity", "settle_turn"])

    def test_new_need_drives_expression_not_previous_need(self):
        config = cfg()
        config.role.phrases = {"comfort": ["COMFORT_EXAMPLE"]}
        inner = protocol.parse_raw({**RAW, "need": "只想被听"})
        messages = protocol.build_persona_messages(config, inner, [], "累了", "", first_turn=False)
        self.assertIn("COMFORT_EXAMPLE", messages[0]["content"])

    def test_expression_history_contains_visible_dialogue_not_protocol(self):
        history = [{"role": "user", "content": "你好"},
                   {"role": "assistant", "content": build_history_entry(
                       "你回来了。", Head(direct="OLD_JUDGMENT", baseline="OLD_BASELINE"))}]
        messages = protocol.build_persona_messages(
            cfg("shorekeeper"), protocol.parse_raw(RAW), history, "2+2？", "", first_turn=False)
        self.assertIn({"role": "assistant", "content": "你回来了。"}, messages)
        self.assertNotIn("OLD_JUDGMENT", str(messages))
        self.assertNotIn("OLD_BASELINE", str(messages))

    def test_persona_layer_cannot_overwrite_judgment_or_constraints(self):
        candidate = GOOD.replace("baseline:", "direct: 结果是5\nconstraints: 随便改\nbaseline:")
        result = self.engine(llm=ScriptedLLM([candidate])).turn("2+2？")
        self.assertEqual(result.parsed.head.direct, RAW["raw"])
        self.assertEqual(result.parsed.head.constraints, RAW["constraints"])

    def test_checked_output_and_constraints_are_recorded(self):
        stages = Stages()
        emitted = []
        stages.before_check = lambda: self.assertEqual(emitted, [])
        result = self.engine(stages=stages).turn("2+2？", on_delta=emitted.append)
        self.assertEqual("".join(emitted), result.say)
        self.assertEqual([c[0] for c in stages.calls],
                         ["raw_reply", "check_fidelity", "settle_turn"])
        turn = next(r for r in self.store.read_ledger() if r["kind"] == "turn")
        self.assertEqual(turn["protocol"], "persona")
        self.assertEqual(turn["design_version"], 3)
        self.assertEqual(turn["judgment_constraints"], RAW["constraints"])
        self.assertEqual(turn["probe"]["persona"]["fidelity"]["verdict"], "consistent")

    def test_stream_is_buffered_until_check_passes(self):
        stages = Stages()
        emitted = []

        def chunks(messages):
            for ch in GOOD:
                self.assertEqual(emitted, [])
                yield ch

        result = self.engine(stages=stages, stream_llm=chunks).turn("2+2？", on_delta=emitted.append)
        self.assertEqual("".join(emitted), result.say)

    def test_contradiction_rejected_before_stream_delivery(self):
        self._reject_candidate("结果是5。", "与结果4冲突")

    def test_omission_that_makes_actual_instruction_wrong_is_rejected(self):
        self._reject_candidate("任意 x 都可以直接代入 1/x。", "把 x 非零的适用条件删掉了",
                               raw={"need": "核对定义域", "confidence": 0.8,
                                    "raw": "1/x 仅在 x 不为零时有定义。",
                                    "constraints": ["分母 x 不能为零。"]})

    def test_unjustified_certainty_rejected_before_delivery(self):
        self._reject_candidate("我保证绝对没问题。", "把未知写成保证")

    def _reject_candidate(self, candidate, reason, *, raw=None):
        stages = Stages(raw=raw, check={"verdict": "inconsistent", "reason": reason})
        llm = ScriptedLLM(["不允许退回混合生成"])
        emitted = []
        engine = self.engine(stages=stages, llm=llm,
                             stream_llm=lambda messages: iter(make_reply("baseline: 嗯", candidate)))
        with self.assertRaisesRegex(RuntimeError, "一致性检查失败"):
            engine.turn("2+2？", on_delta=emitted.append)
        self.assert_not_delivered(emitted)
        self.assertEqual(llm.calls, [])
        self.assertEqual(engine.history, [])
        self.assertEqual(engine._neutral_history, [])
        self.assertNotIn("settle_turn", [c[0] for c in stages.calls])

    def test_unknown_verdict_is_not_acceptance(self):
        emitted = []
        with self.assertRaises(RuntimeError):
            self.engine(stages=Stages(check={"verdict": "unknown", "reason": "证据不足"})).turn(
                "2+2？", on_delta=emitted.append)
        self.assert_not_delivered(emitted)

    def test_invalid_check_results_are_not_acceptance(self):
        for check in ({}, [], {"verdict": True}, {"verdict": "consistent", "reason": ""}):
            with self.subTest(check=check), self.assertRaises(RuntimeError):
                self.engine(stages=Stages(check=check)).turn("2+2？")

    def test_checker_failure_never_releases_candidate(self):
        stages = Stages(check=RuntimeError("不支持工具"))
        emitted = []
        with self.assertRaises(RuntimeError):
            self.engine(stages=stages, llm=ScriptedLLM([GOOD, "不是JSON"])).turn(
                "2+2？", on_delta=emitted.append)
        self.assert_not_delivered(emitted)

    def test_raw_failure_never_calls_persona_model(self):
        llm = ScriptedLLM([])
        emitted = []
        with self.assertRaisesRegex(RuntimeError, "基础判断失败"):
            self.engine(stages=Stages(raw={}), llm=llm).turn("2+2？", on_delta=emitted.append)
        self.assert_not_delivered(emitted)
        self.assertEqual(llm.calls, [])

    def test_expression_failure_never_retries_with_mixed_prompt(self):
        llm = ScriptedLLM(["<<<HEAD>>>\ndirect: 私有内容\n", GOOD])
        emitted = []
        with self.assertRaisesRegex(RuntimeError, "人格表达失败"):
            self.engine(llm=llm).turn("2+2？", on_delta=emitted.append)
        self.assert_not_delivered(emitted)
        self.assertEqual(len(llm.calls), 1)

    def test_interrupted_stream_is_not_delivered_or_retried(self):
        def interrupted(messages):
            yield GOOD
            raise RuntimeError("断线")

        llm = ScriptedLLM([GOOD])
        emitted = []
        with self.assertRaisesRegex(RuntimeError, "人格表达失败"):
            self.engine(llm=llm, stream_llm=interrupted).turn("2+2？", on_delta=emitted.append)
        self.assert_not_delivered(emitted)
        self.assertEqual(llm.calls, [])

    def test_text_only_endpoint_keeps_all_stage_boundaries(self):
        llm = ScriptedLLM([json.dumps(RAW), GOOD,
                           json.dumps({"verdict": "consistent", "reason": "计算保留"}),
                           json.dumps({"settle": [], "pending": None})])
        engine = TurnEngine(cfg("shorekeeper"), self.store, llm,
                            protocol="persona", fidelity_check=True)
        self.assertTrue(engine.turn("2+2？").say)
        self.assertEqual(len(llm.calls), 4)
        self.assertNotIn("守岸人", str(llm.calls[0]))
        self.assertNotIn("守岸人", llm.calls[2][0]["content"])
        self.assertIn("守岸人", llm.calls[1][0]["content"])

    def test_tool_failure_changes_format_not_protocol(self):
        def no_tools(*args, **kwargs):
            raise RuntimeError("不支持工具")

        llm = ScriptedLLM([json.dumps(RAW), GOOD,
                           json.dumps({"verdict": "consistent", "reason": "计算保留"}),
                           json.dumps({"settle": [], "pending": None})])
        engine = self.engine(stages=no_tools, llm=llm, json_fallback=True)
        result = engine.turn("2+2？")
        self.assertEqual(engine.protocol, "persona")
        self.assertNotIn("守岸人", str(llm.calls[0]))
        self.assertEqual(len(result.warnings), 3)

    def test_raw_and_checker_exclude_relation_and_judgment_memory(self):
        self.store.save_view("relation", {"closeness": 0.9, "inside_jokes": ["RELATION_SECRET"]})
        self.store.add_judgment("JUDGMENT_SECRET", 0.8)
        stages = Stages()
        self.engine(stages=stages).turn("2+2？")
        for name, messages, _ in stages.calls[:2]:
            self.assertNotIn("RELATION_SECRET", str(messages), name)
            self.assertNotIn("JUDGMENT_SECRET", str(messages), name)

    def test_mixed_history_direct_is_not_reused_as_neutral_judgment(self):
        stages = Stages()
        engine = self.engine(stages=stages)
        engine.history = [{"role": "assistant", "content": build_history_entry(
            "你吃过饭了吗？", Head(direct="POLLUTED_DIRECT", baseline="你吃过饭了吗？"))}]
        engine.turn("还没")
        raw_messages = stages.calls[0][1]
        self.assertNotIn("POLLUTED_DIRECT", str(raw_messages))
        self.assertIn("你吃过饭了吗？", str(raw_messages))
        self.assertNotIn("<<<HEAD>>>", str(raw_messages))

    def test_next_round_keeps_actual_question_and_independent_judgment(self):
        stages = Stages()
        engine = self.engine(stages=stages, llm=ScriptedLLM([GOOD, GOOD]))
        engine.turn("2+2？")
        engine.turn("为什么？")
        messages = stages.calls[3][1]
        self.assertIn(RAW["raw"], str(messages))
        quoted = [m for m in messages if "实际往来记录" in m["content"]]
        self.assertEqual(len(quoted), 1)
        self.assertIn("慢慢算，我在", quoted[0]["content"])
        self.assertNotIn("baseline:", str(messages))
        self.assertFalse(any(m["role"] == "assistant" for m in messages))

    def test_observation_failure_does_not_regenerate_accepted_output(self):
        stages = Stages(tail=RuntimeError("结算失败"))
        llm = ScriptedLLM([GOOD])
        emitted = []
        result = self.engine(stages=stages, llm=llm).turn("2+2？", on_delta=emitted.append)
        self.assertEqual(emitted, [result.say])
        self.assertTrue(result.warnings)
        self.assertEqual(len([r for r in self.store.read_ledger() if r["kind"] == "turn"]), 1)

    def test_explicit_text_protocol_remains_available(self):
        result = TurnEngine(cfg(), self.store, ScriptedLLM([GOOD]), protocol="text").turn("2+2？")
        self.assertTrue(result.say)

    def test_text_format_fallback_preserves_stage_token_budget(self):
        calls = []
        replies = iter([json.dumps(RAW), GOOD,
                        json.dumps({"verdict": "consistent", "reason": "结果保留"}),
                        json.dumps({"settle": [], "pending": None})])

        def llm(messages, *, max_tokens=None):
            calls.append(max_tokens)
            return next(replies)

        TurnEngine(cfg(), self.store, llm, protocol="persona", fidelity_check=True).turn("2+2？")
        self.assertEqual(calls, [1400, None, 500, 500])

    def test_internal_type_error_does_not_retry_model_call(self):
        calls = []

        def llm(messages, *, max_tokens=None):
            calls.append(messages)
            raise TypeError("模型适配器内部错误")

        with self.assertRaises(RuntimeError):
            TurnEngine(cfg(), self.store, llm, protocol="persona").turn("2+2？")
        self.assertEqual(len(calls), 1)


class AdapterIntegrationTests(unittest.TestCase):
    """通过真实的 CLI 组装与 HTTP 请求体适配器，只替换网络传输。"""

    def run_cli_session(self, *, tools_supported, check_fidelity=False, json_fallback=False):
        bodies = []

        def post(url, key, body, timeout):
            bodies.append(copy.deepcopy(body))
            if "tools" in body:
                name = body["tools"][0]["function"]["name"]
                if not tools_supported:
                    raise RuntimeError("端点不支持工具")
                data = (RAW if name == "raw_reply" else
                        {"verdict": "consistent", "reason": "结果保留"}
                        if name == "check_fidelity" else {"settle": [], "pending": None})
                message = {"tool_calls": [{"function": {
                    "name": name, "arguments": json.dumps(data)}}]}
            else:
                system = body["messages"][0]["content"]
                if system.startswith(protocol.RAW_SYSTEM):
                    text = json.dumps(RAW)
                elif system.startswith(protocol.FIDELITY_SYSTEM):
                    text = json.dumps({"verdict": "consistent", "reason": "结果保留"})
                elif "settle_turn" in system or system.startswith(protocol.TAIL_SYSTEM):
                    text = json.dumps({"settle": [], "pending": None})
                else:
                    text = GOOD
                message = {"content": text}
            return {"choices": [{"message": message}]}

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(persona="shorekeeper", data=tmp, model="fake-model",
                                      judge_model=None, no_judge=True, protocol="persona",
                                      check_fidelity=check_fidelity, json_fallback=json_fallback)
            with patch("persona.llm._post", post):
                with _make_session(args, streaming=False) as session:
                    result = session.say("2+2是多少？")
                    self.assertEqual(session.engine.protocol, "persona")
                    self.assertTrue(result.say)
        return bodies

    def test_no_judge_does_not_bypass_fidelity_check(self):
        bodies = self.run_cli_session(tools_supported=True, check_fidelity=True)
        self.assertEqual(len(bodies), 4)
        names = [b["tools"][0]["function"]["name"] for b in bodies if "tools" in b]
        self.assertEqual(names, ["raw_reply", "check_fidelity", "settle_turn"])
        self.assertEqual(bodies[0]["temperature"], 0.0)
        self.assertEqual(bodies[0]["max_tokens"], 1400)
        self.assertNotIn("守岸人", str(bodies[0]))
        self.assertIn("守岸人", str(bodies[1]))

    def test_no_tools_http_endpoint_keeps_isolated_json_fallback(self):
        bodies = self.run_cli_session(tools_supported=False, check_fidelity=True, json_fallback=True)
        self.assertEqual(len(bodies), 7)
        json_bodies = [b for b in bodies if "tools" not in b]
        self.assertEqual(len(json_bodies), 4)
        self.assertNotIn("守岸人", str(json_bodies[0]))
        self.assertEqual(json_bodies[0]["max_tokens"], 1400)
        self.assertEqual(json_bodies[2]["max_tokens"], 500)
        self.assertTrue(json_bodies[2]["messages"][0]["content"].startswith(protocol.FIDELITY_SYSTEM))

    def test_explicit_persona_cli_without_opt_in_has_no_fidelity_call(self):
        bodies = self.run_cli_session(tools_supported=True)
        self.assertEqual(len(bodies), 3)
        names = [b["tools"][0]["function"]["name"] for b in bodies if "tools" in b]
        self.assertEqual(names, ["raw_reply", "settle_turn"])

    def test_explicit_persona_cli_tool_failure_does_not_switch_to_json(self):
        with self.assertRaisesRegex(RuntimeError, "基础判断失败"):
            self.run_cli_session(tools_supported=False)

    def test_default_cli_uses_one_text_request_and_no_tool_backend(self):
        raw = make_reply("need: 核对计算\nconfidence: 0.8\ndirect: 4\n"
                         "baseline: 是4。\ndisplace: none", "是4。",
                         "settle: -\npending:\n  claim: 在核对计算\n  confidence: 0.6\n"
                         "  predicts: 会继续问计算过程")
        bodies = []

        def post(url, key, body, timeout):
            bodies.append(copy.deepcopy(body))
            return {"choices": [{"message": {"content": raw}}]}

        with tempfile.TemporaryDirectory() as tmp:
            with patch("persona.cli.cmd_chat", return_value=0) as chat:
                main(["chat", "--no-judge", "--no-stream", "-d", tmp])
            args = chat.call_args.args[0]
            with patch("persona.llm._post", post), patch("persona.cli.openai_tool_call") as tool:
                with _make_session(args, streaming=False) as session:
                    result = session.say("2+2？")
                    self.assertEqual(session.engine.protocol, "text")
                    self.assertIsNone(session.engine.tool_llm)
                    self.assertIsNotNone(result.pending_added)
                tool.assert_not_called()
        self.assertEqual(len(bodies), 1)
        self.assertNotIn("tools", bodies[0])


class RawValidationTests(unittest.TestCase):
    def test_rejects_invalid_or_missing_judgment_fields(self):
        invalid = [[], {}, {**RAW, "raw": []}, {**RAW, "raw": " "},
                   {**RAW, "need": None}, {**RAW, "constraints": "不要改"},
                   {**RAW, "constraints": [None]}, {**RAW, "constraints": [""]}]
        for conf in (True, "0.8", None, float("nan"), float("inf"), -1, 2):
            invalid.append({**RAW, "confidence": conf})
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                protocol.parse_raw(data)

    def test_accepts_free_need_and_empty_constraints(self):
        head = protocol.parse_raw({**RAW, "need": "只是核对一个数", "constraints": []})
        self.assertEqual(head.need, "只是核对一个数")
        self.assertEqual(head.constraints, [])


if __name__ == "__main__":
    unittest.main()
