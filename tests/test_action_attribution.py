"""An application may attribute an action without obtaining a real-world capability."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

from persona import ActionContext, ActionSpec
from persona.config import PersonaConfig
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.session import Session, SessionConfig
from persona.store import Store


ROOT = Path(__file__).resolve().parents[1]
USER = "替我拟个标题。我自己垫稳画板；在这个童话里，替小鸟撑起雨伞。"
ACTIONS = (ActionSpec("assistant_text", "拟个标题"), ActionSpec("user_physical", "垫稳画板"),
           ActionSpec("story", "替小鸟撑起雨伞", "在这个童话里"))


def raw(say):
    return make_reply("need: 回答\nconfidence: 0.8\ndirect: 根据来信回应\nbaseline: 好\ndisplace: none",
                      say, "settle: -\ndelta: -\npending: -")


class ActionAttributionTests(unittest.TestCase):
    def test_both_forms_carry_same_annotations_rules_and_no_gold_reply(self):
        seq = list(reversed(ACTIONS))
        ctx = ActionContext(seq)
        seq.clear()
        ctx.validate_evidence(USER)
        with self.assertRaises(FrozenInstanceError):
            ctx.actions[0].kind = "user_physical"
        self.assertEqual(ctx.actions, tuple(reversed(ACTIONS)))
        manual, linked = ctx.manual_prompt().splitlines(), ctx.prompt().splitlines()
        source = [json.dumps(row, ensure_ascii=False) for row in ctx.record()["actions"]]
        rules = [line for line in manual if line.split("：")[0] in {a.kind for a in ACTIONS}]
        self.assertEqual(len(rules), 3)
        self.assertEqual([line for line in linked if line in rules], rules)
        for a, line in zip(ctx.actions, source):
            self.assertEqual(manual.count(line), 1)
            self.assertEqual(linked.count(line), 1)
            self.assertLess(manual.index(line), manual.index(rules[0]))
            self.assertTrue(linked[linked.index(line)-1].startswith(a.kind + "："))

    def test_invalid_owner_frame_or_duplicate_evidence_is_rejected(self):
        for args in (("assistant", "读这句"), ("assistant_text", " "),
                     ("story", "推门"), ("story", "推门", " "),
                     ("user_physical", "推门", "假想中")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ActionSpec(*args)
        for seq in ((), (ACTIONS[0], ACTIONS[0]),
                    (ActionSpec("assistant_text", "推门"), ActionSpec("user_physical", "推门"))):
            with self.subTest(seq=seq), self.assertRaises(ValueError):
                ActionContext(seq)
        with self.assertRaises(TypeError):
            ActionContext([{"kind":"user_physical", "quote":"推门"}])

    def test_invalid_input_contacts_neither_model_nor_turn_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            cfg, store, llm = PersonaConfig.load(ROOT/"personas/default.toml"), Store(td), ScriptedLLM([])
            engine = TurnEngine(cfg, store, llm)
            before = store.read_ledger()
            for text, actions, error in [
                    ("只有开灯", ActionContext((ActionSpec("user_physical", "开门"),)), ValueError),
                    ("替小鸟撑起雨伞", ActionContext((ACTIONS[2],)), ValueError),
                    ("现实中开门，再在故事里开门。", ActionContext((ActionSpec("story","开门","在故事里"),)), ValueError),
                    ("故事里有人；故事里请开门。", ActionContext((ActionSpec("story","请开门","故事里"),)), ValueError),
                    (USER, [], TypeError)]:
                with self.subTest(text=text), self.assertRaises(error):
                    engine.turn(text, actions=actions)
            self.assertEqual(store.read_ledger(), before)
            self.assertFalse(llm.calls)

    def test_distinct_actions_can_share_one_story_frame(self):
        ctx = ActionContext((ActionSpec("story", "点灯", "在童话里"), ActionSpec("story", "挂旗", "在童话里")))
        ctx.validate_evidence("在童话里，请帮月亮点灯，替星星挂旗。")

    def test_other_protocols_do_not_silently_discard_evidence(self):
        for protocol in ("auto", "tool", "persona"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as td:
                store, llm = Store(td), ScriptedLLM([])
                eng = TurnEngine(PersonaConfig.load(ROOT/"personas/default.toml"), store, llm, protocol=protocol)
                with self.assertRaisesRegex(ValueError, "text protocol"):
                    eng.turn(USER, actions=ActionContext(ACTIONS))
                self.assertFalse(llm.calls)
                self.assertFalse(any(r["kind"] == "turn_start" for r in store.read_ledger()))

    def test_untrusted_annotations_do_not_become_semantic_verdicts_or_corrections(self):
        # The model deliberately makes a physically impossible claim. Keep the observed output.
        reply = "我已经亲手将画板垫稳了。"
        ctx = ActionContext(ACTIONS)
        for path in ("full", "compact", "direct"):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as td:
                output = (raw(reply) if path == "full" else reply if path == "direct" else
                          make_reply("need: 回答\nconfidence: .8\ndirect: 按情境回答", reply, ""))
                llm, store = ScriptedLLM([output]), Store(td)
                engine = TurnEngine(PersonaConfig.load(ROOT/"personas/default.toml"), store, llm,
                                    generation_path=path)
                result = engine.turn(USER, actions=ctx)
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(result.say, reply)
                self.assertEqual(result.action_reference["output_validation"], "not_assessed")
                self.assertTrue(llm.calls[0][0]["content"].endswith("\n\n" + ctx.prompt()))
                self.assertEqual(llm.calls[0][-1]["content"], USER)
                self.assertEqual(engine.history[-2]["content"], USER)
                for event in store.read_ledger():
                    if event["kind"] in ("turn_start","turn"):
                        self.assertEqual(event["action_reference"], ctx.record())

    def test_session_annotated_stream_then_plain_chat_has_no_residual_prompt_or_authority(self):
        ctx = ActionContext(ACTIONS)
        with tempfile.TemporaryDirectory() as td:
            llm = ScriptedLLM([raw("伞下见。"), raw("嗯。")])
            store = Store(td)
            with Session(PersonaConfig.load(ROOT/"personas/default.toml"), store, llm,
                         options=SessionConfig(memory_mode="frozen")) as s:
                turn = s.say_stream(USER, actions=ctx)
                self.assertEqual("".join(turn), "伞下见。")
                self.assertEqual(turn.result.action_reference["actions"], ctx.record()["actions"])
                self.assertEqual(s.chat("现在呢？"), "嗯。")
            self.assertEqual(len(llm.calls), 2)
            self.assertNotIn("[调用方标注的本轮动作归属]", str(llm.calls[1]))
            self.assertNotIn("action_reference", [r for r in store.read_ledger() if r["kind"] == "turn"][-1])


if __name__ == "__main__":
    unittest.main()
