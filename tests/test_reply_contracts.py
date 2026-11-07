"""Explicit reply requirements are separate from truth, protocol and delivery."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from persona import ReplyContract
from persona.config import PersonaConfig
from persona.contracts import validate_reply
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.prompt import build_user_turn
from persona.session import Session, SessionConfig
from persona.store import Store

ROOT = Path(__file__).resolve().parent.parent


def reply(say, tail="settle: -\ndelta: -\npending: -"):
    return make_reply("need: 只想被听\nconfidence: 0.6\ndirect: 参考\nbaseline: 常态\ndisplace: none", say, tail)


class TestMechanicalContracts(unittest.TestCase):
    def test_contract_roundtrip_and_immutable_fields(self):
        contract = ReplyContract.json_object(required_keys=["answer"], value_types={"answer": "number"})
        self.assertEqual(contract.required_keys, ("answer",))
        self.assertEqual(ReplyContract(**asdict(contract)), contract)

    def test_number_units_decimal_and_unknown(self):
        contract = ReplyContract.conclusion(unit="元", answer_type="number")
        for text in ("推导。\n结论：37.5元", "结论：37.50元。", "结论：未知"):
            self.assertTrue(contract.validate(text)["valid"], text)
        for text in ("37.5元", "结论：37.5", "结论：<37.5>元", "结论：37.5元\n谢谢"):
            self.assertFalse(contract.validate(text)["valid"], text)

    def test_contract_does_not_know_gold(self):
        contract = ReplyContract.conclusion(unit="元", answer_type="integer")
        self.assertTrue(contract.validate("结论：600元")["valid"])
        self.assertEqual(contract.validate("结论：600元")["scope"], "mechanical_format_not_semantic_truth")

    def test_unknown_permission_is_explicit(self):
        contract = ReplyContract.conclusion(answer_type="number", allow_unknown=False)
        self.assertFalse(contract.validate("结论：未知")["valid"])

    def test_time_shape_includes_clock_range(self):
        contract = ReplyContract.conclusion(answer_type="time")
        self.assertTrue(contract.validate("结论：00:27")["valid"])
        for text in ("结论：24:27", "结论：00:60", "结论：0:27"):
            self.assertFalse(contract.validate(text)["valid"])

    def test_json_objects_keys_types_and_boolean(self):
        contract = ReplyContract.json_object(required_keys=("answer",), value_types={"answer": "number"})
        self.assertTrue(contract.validate('{"answer":37.5}')["valid"])
        for text in ('{"answer":true}', '{"answer":"37.5"}', '{}', '[37.5]', '```json\n{"answer":37.5}\n```'):
            self.assertFalse(contract.validate(text)["valid"], text)

    def test_json_duplicate_and_nonfinite_are_not_valid(self):
        contract = ReplyContract.json_object()
        for text in ('{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e309}'):
            self.assertFalse(contract.validate(text)["valid"], text)

    def test_json_array_root_and_item_types(self):
        contract = ReplyContract.json_array(item_type="integer")
        self.assertTrue(contract.validate('[3,1,2]')["valid"])
        self.assertFalse(contract.validate('[3,true,2]')["valid"])
        self.assertFalse(contract.validate('{"values":[3,1,2]}')["valid"])

    def test_table_header_order_rows_and_width(self):
        contract = ReplyContract.markdown_table(("项目", "金额"), min_rows=2)
        good = "| 项目 | 金额 |\n| --- | ---: |\n| A | 30 |\n| B | 40 |"
        self.assertTrue(contract.validate(good)["valid"])
        for text in (good.replace("项目 | 金额", "金额 | 项目"), good.rsplit("\n",1)[0], good+"\n说明。", good.replace("A | 30", "A | 30 | 多余")):
            self.assertFalse(contract.validate(text)["valid"], text)

    def test_one_column_setext_heading_is_not_table(self):
        contract = ReplyContract.markdown_table(("项目",))
        self.assertFalse(contract.validate("项目\n---\nA")["valid"])
        self.assertTrue(contract.validate("|项目|\n|---|\n|A|")["valid"])

    def test_malformed_specs_fail_before_any_generation(self):
        for spec in (dict(kind="other"), dict(prefix="\n"), dict(answer_type="guess"),
                     dict(required_keys="answer"), dict(kind="json", required_keys=("x","x")),
                     dict(kind="json", value_types={"x": "float"}),
                     dict(kind="json", json_type="array", required_keys=("x",)),
                     dict(kind="markdown_table", columns=("x|y",)),
                     dict(kind="markdown_table", columns=("x",), min_rows=True)):
            with self.subTest(spec=spec), self.assertRaises(ValueError): ReplyContract(**spec)

    def test_none_is_not_requested_and_prompt_is_identical(self):
        text = " 原始输入\n不作更改。 "
        self.assertEqual(build_user_turn(text), text)
        self.assertEqual(build_user_turn(text, None), text)
        self.assertIsNone(validate_reply("plain", None)["valid"])


class TestContractIntegration(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(temporary.name)
        self.cfg = PersonaConfig.load(ROOT/"personas/default.toml")

    def test_none_messages_are_unmodified(self):
        llm = ScriptedLLM([reply("收到。")])
        engine = TurnEngine(self.cfg, self.store, llm)
        result = engine.turn("今天好累")
        self.assertEqual(llm.calls[0][-1], {"role":"user", "content":"今天好累"})
        self.assertNotIn("调用方明确指定的本轮", str(llm.calls[0]))
        self.assertEqual(result.contract_validation["status"], "not_requested")

    def test_invalid_contract_type_calls_no_model_and_starts_no_turn(self):
        llm = ScriptedLLM([reply("收到。")])
        engine = TurnEngine(self.cfg, self.store, llm)
        with self.assertRaises(TypeError): engine.turn("问题", contract={"kind":"json"})
        self.assertEqual(llm.calls, [])
        self.assertFalse(any(row["kind"] == "turn_start" for row in self.store.read_ledger()))

    def test_one_call_prompt_and_original_user_memory_are_separate(self):
        contract = ReplyContract.conclusion(unit="元", answer_type="number")
        llm = ScriptedLLM([reply("结论：300元")])
        engine = TurnEngine(self.cfg, self.store, llm)
        result = engine.turn("请计算余额", contract=contract)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn(contract.prompt(), llm.calls[0][-1]["content"])
        self.assertTrue(result.contract_validation["valid"])
        turn = next(row for row in self.store.read_ledger() if row["kind"] == "turn")
        self.assertEqual(turn["user"], "请计算余额")
        self.assertEqual(turn["contract_validation"], result.contract_validation)
        self.assertEqual(engine.history[-2]["content"], "请计算余额")
        for path in self.store.buffer_dir.glob("*.json"):
            self.assertNotIn("调用方明确指定的本轮", path.read_text(encoding="utf-8"))

    def test_format_failure_does_not_rewrite_stream_or_erase_valid_tail(self):
        raw = reply("还剩300元。", "settle: -\ndelta:\n  closeness: 0.01\npending: -")
        calls=[]
        def stream(messages):
            calls.append(messages)
            for index in range(0,len(raw),7): yield raw[index:index+7]
        engine = TurnEngine(self.cfg, self.store, ScriptedLLM([]), stream_llm=stream)
        closeness_before = self.store.relation()["closeness"]
        pieces=[]
        result=engine.turn("请计算余额", on_delta=pieces.append,
                           contract=ReplyContract.conclusion(unit="元", answer_type="number"))
        self.assertEqual(len(calls),1)
        self.assertEqual(''.join(pieces), "还剩300元。")
        self.assertEqual(result.say, "还剩300元。")
        self.assertFalse(result.contract_validation["valid"])
        self.assertTrue(result.validity["valid"])
        self.assertAlmostEqual(self.store.relation()["closeness"], closeness_before + .01)

    def test_protocol_failure_can_still_satisfy_reply_format(self):
        llm = ScriptedLLM([reply("结论：300元")+"\n<<<HEAD>>>\nneed: 只想被听"])
        result=TurnEngine(self.cfg,self.store,llm).turn("问题",contract=ReplyContract.conclusion(unit="元"))
        self.assertFalse(result.validity["valid"])
        self.assertTrue(result.contract_validation["valid"])

    def test_session_chat_and_iteration_forward_contract(self):
        contract=ReplyContract.json_object(required_keys=("answer",))
        llm=ScriptedLLM([reply('{"answer":1}'),reply('{"answer":2}')])
        with Session(self.cfg,self.store,llm, options=SessionConfig(memory_mode="frozen", consolidate_on_exit=False)) as session:
            self.assertEqual(session.chat("第一题",contract=contract),'{"answer":1}')
            streamed=session.say_stream("第二题",contract=contract)
            self.assertEqual(''.join(streamed),'{"answer":2}')
            self.assertTrue(streamed.result.contract_validation["valid"])
        self.assertEqual(len(llm.calls),2)

    def test_persona_contract_only_reaches_expression_not_raw_or_history(self):
        calls=[]
        def tools(messages, *, tool, **kwargs):
            name=tool["function"]["name"]
            calls.append((name,messages))
            if name == "raw_reply":
                return {"need":"只想被听", "confidence":.7, "raw":"余额300元", "constraints":[]}
            return {"settle":[], "pending":None}
        text=ScriptedLLM(["<<<HEAD>>>\nbaseline: 余额300元\ndisplace: none\n<<<SAY>>>\n结论：300元"])
        engine=TurnEngine(self.cfg,self.store,text,protocol="persona",tool_llm=tools)
        result=engine.turn("计算余额",contract=ReplyContract.conclusion(unit="元"))
        self.assertTrue(result.contract_validation["valid"])
        self.assertIn("调用方明确指定的本轮",text.calls[0][-1]["content"])
        self.assertNotIn("调用方明确指定的本轮",str(calls))
        self.assertEqual(engine._neutral_history[0]["content"],"计算余额")

    def test_tool_contract_only_reaches_expression(self):
        calls=[]
        def tools(messages, *, tool, **kwargs):
            name=tool["function"]["name"]
            calls.append((name,messages))
            if name == "inner":
                return {"need":"只想被听", "confidence":.7, "direct":"余额300元", "baseline":"余额300元", "displace":"none"}
            return {"settle":[], "pending":None}
        text=ScriptedLLM(["结论：300元"])
        engine=TurnEngine(self.cfg,self.store,text,protocol="tool",tool_llm=tools)
        result=engine.turn("计算余额",contract=ReplyContract.conclusion(unit="元"))
        self.assertTrue(result.contract_validation["valid"])
        self.assertIn("调用方明确指定的本轮",text.calls[0][-1]["content"])
        self.assertNotIn("调用方明确指定的本轮",str(calls))


if __name__ == "__main__": unittest.main()
