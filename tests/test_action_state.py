import json
from pathlib import Path
import tempfile
import unittest

from persona import ActionContext, ActionSpec, ActionState, ActionStateContext
from persona.config import PersonaConfig
from persona.llm import ScriptedLLM
from persona.session import Session, SessionConfig
from persona.store import Store


ROOT = Path(__file__).resolve().parents[1]
USER = ("昨天甲章在抽屉。我想把甲章送至前台，但我还没动；刚看过，甲章仍在桌上。"
        "乙印本打算入黄箱，经办人回报：乙印已转入青箱。请替我拟个摘要。")
A = ActionSpec("user_physical", "把甲章送至前台")
B = ActionSpec("user_physical", "乙印本打算入黄箱")
DRAFT = ActionSpec("assistant_text", "拟个摘要")
PENDING = ActionState(A.quote, "pending", "我还没动；刚看过，甲章仍在桌上",
                      planned_quote="送至前台", before_quote="昨天甲章在抽屉", current_quote="甲章仍在桌上")
CONFIRMED = ActionState(B.quote, "confirmed", "经办人回报：乙印已转入青箱", planned_quote="打算入黄箱",
                        result_quote="乙印已转入青箱")


class ActionStateTests(unittest.TestCase):
    def test_action_scope_different_result_from_plan_and_separate_historical_observation(self):
        ctx = ActionStateContext((A, B, DRAFT), (CONFIRMED, PENDING))
        ctx.validate_evidence(USER)
        audit = ctx.record()["action_state"]
        self.assertEqual(audit["output_validation"], "not_assessed")
        rows = {r["action_id"]: r for r in audit["rows"]}
        projected = {r["action_id"]: r for r in audit["projection"]}
        for action_id, expected in (("a1", "甲章仍在桌上"), ("a2", "乙印已转入青箱")):
            source, field = projected[action_id]["current_ref"].split(".")
            self.assertEqual(source, action_id)
            self.assertEqual(rows[source][field], expected)
        self.assertEqual(projected["a1"]["last_known_ref"], "a1.before_quote")
        self.assertEqual(projected["a2"]["completion"], "reported_complete")
        self.assertEqual(set(rows), {"a1","a2"})
        baseline = ActionContext((A,B,DRAFT)).prompt()
        self.assertTrue(ctx.manual_prompt().startswith(baseline))
        self.assertTrue(ctx.prompt().startswith(ctx.manual_prompt()))
        self.assertEqual(json.loads(ctx.manual_prompt().splitlines()[-1]), audit["rows"])
        self.assertEqual(json.loads(ctx.prompt().splitlines()[-1]), audit["projection"])

    def test_not_done_by_speaker_does_not_make_last_observation_a_current_value(self):
        text = "昨天甲章在抽屉。我想把甲章送至前台，但我自己还没动，其他人可能动过。"
        for state in (ActionState(A.quote,"pending","我自己还没动", before_quote="昨天甲章在抽屉"),
                      ActionState(A.quote,"unknown","其他人可能动过", before_quote="昨天甲章在抽屉"),
                      ActionState(A.quote,"unknown",planned_quote="送至前台")):
            with self.subTest(state=state):
                ctx=ActionStateContext((A,), (state,))
                ctx.validate_evidence(text)
                p=ctx.record()["action_state"]["projection"][0]
                self.assertIsNone(p["current_ref"])
                self.assertEqual(p["completion"], "reported_not_done" if state.status == "pending" else "unknown")

    def test_one_quote_cannot_confirm_two_different_actions_or_a_story(self):
        story = ActionSpec("story", "扶起月亮", "在童话里")
        for states in ((PENDING,PENDING),
                       (ActionState("扶起月亮","unknown"),),
                       (ActionState(DRAFT.quote,"unknown"),),
                       (ActionState("未曾标注的行动","unknown"),),
                       (CONFIRMED, ActionState(A.quote, "confirmed", CONFIRMED.evidence_quote,
                                               result_quote=CONFIRMED.result_quote))):
            with self.subTest(states=states), self.assertRaises(ValueError):
                ActionStateContext((A, B, DRAFT, story), states)

    def test_fiction_and_drafted_claims_do_not_supply_evidence_for_a_real_action(self):
        declared = ActionState(A.quote, "confirmed", "甲章已经送达", result_quote="已经送达")
        for spec in (ActionSpec("story", "在戏里说甲章已经送达", "只是戏词"),
                     ActionSpec("assistant_text", "把甲章已经送达写成台词")):
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "distinct real source"):
                ActionStateContext((A,spec),(declared,))

    def test_missing_wrong_or_ambiguous_sources_do_not_enter_the_session(self):
        bad = [dict(status="confirmed", evidence_quote="打算送至前台"),
               dict(status="confirmed", evidence_quote="刚看到它在库房", result_quote="前台"),
               dict(status="confirmed", evidence_quote=A.quote, result_quote="送至前台"),
               dict(status="pending", evidence_quote="尚未操作", current_quote="库房"),
               dict(status="unknown", current_quote="甲章仍在桌上"),
               dict(status="pending", evidence_quote="尚未操作", result_quote="前台"),
               dict(status="pending"), dict(status="unknown", before_quote=" ")]
        for fields in bad:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                ActionState(A.quote,**fields)
        with tempfile.TemporaryDirectory() as td:
            llm=ScriptedLLM([])
            store=Store(td)
            with Session(PersonaConfig.load(ROOT/"personas/default.toml"),store,llm,
                         options=SessionConfig(generation_path="direct",memory_mode="frozen")) as s:
                for text in (USER.replace("昨天甲章在抽屉","昨晚没有去看"), USER+"又说昨天甲章在抽屉。"):
                    with self.assertRaises(ValueError):
                        s.say(text, actions=ActionStateContext((A,), (PENDING,)))
            self.assertFalse(llm.calls)
            self.assertFalse(any(e["kind"]=="turn_start" for e in store.read_ledger()))

    def test_actual_delivery_does_not_confirm_a_proposed_transition_or_leak_into_next_turn(self):
        ctx = ActionStateContext((A,B,DRAFT),(PENDING,CONFIRMED))
        # Keep the incorrect delivered claim observable; it cannot mutate the input reference.
        reply = "甲章已经送到前台。"
        with tempfile.TemporaryDirectory() as td:
            store=Store(td)
            llm=ScriptedLLM([reply, "我在。"])
            with Session(PersonaConfig.load(ROOT/"personas/default.toml"),store,llm,
                         options=SessionConfig(generation_path="direct", memory_mode="frozen")) as s:
                actual=s.say(USER,actions=ctx)
                self.assertEqual(actual.say,reply)
                self.assertEqual(actual.action_reference,ctx.record())
                self.assertEqual(s.chat("我们换个话题。"),"我在。")
            self.assertEqual(len(llm.calls),2)
            self.assertTrue(llm.calls[0][0]["content"].endswith(ctx.prompt()))
            self.assertNotIn("[调用方提供的本轮现实动作状态]", llm.calls[1][0]["content"])
            self.assertEqual(llm.calls[0][-1]["content"],USER)
            turns=[e for e in store.read_ledger() if e["kind"]=="turn"]
            self.assertEqual(turns[0]["action_reference"],ctx.record())
            self.assertNotIn("action_reference",turns[1])


if __name__ == "__main__":
    unittest.main()
