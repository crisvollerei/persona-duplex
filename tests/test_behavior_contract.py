"""人格基准、观察结算与持久化行为的离线回归测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from persona import protocol
from persona.config import PersonaConfig
from persona.consolidate import Consolidator, SELF_STORY_HEADER
from persona.engine import TurnEngine
from persona.llm import ScriptedLLM, make_reply
from persona.metrics import compute, verdicts
from persona.parse import Head, parse, safe_say
from persona.prompt import build_history_entry, build_system_prompt
from persona.settle import extract_features, judge, judge_delayed
from persona.store import Store, rebuild_views
from persona.stream import StreamParser


ROOT = Path(__file__).resolve().parent.parent


def cfg(name="default"):
    return PersonaConfig.load(ROOT / "personas" / f"{name}.toml")


def reply(say="我在。", *, baseline="我在。", displace="none", tail="settle: -"):
    return make_reply(
        f"need: 只想被听\nconfidence: 0.6\ndirect: 可以先休息。\n"
        f"baseline: {baseline}\ndisplace: {displace}", say, tail)


class BaselineTests(unittest.TestCase):
    def test_persona_expression_is_not_extra_displacement(self):
        turn = parse(reply("黑海岸的灯还亮着。", baseline="黑海岸的灯还亮着。"))
        self.assertNotEqual(turn.head.direct, turn.say)
        self.assertFalse(turn.has_displacement)
        self.assertTrue(turn.is_consistent())

    def test_comparison_uses_baseline_not_direct(self):
        head = Head(direct="可以先休息。", baseline="我想再陪你一会儿。", displace="转移")
        self.assertFalse(protocol.displacement_ok(head, head.baseline)[0])
        self.assertTrue(protocol.displacement_ok(head, head.direct)[0])

    def test_legacy_head_is_not_regraded_using_direct(self):
        turn = parse(make_reply("direct: 我在。\ndisplace: 悬置", "我在。"))
        self.assertTrue(turn.is_consistent())
        self.assertIn("未知", protocol.displacement_ok(turn.head, turn.say)[1])

    def test_metrics_exclude_legacy_and_missing_baselines(self):
        rows = [
            {"kind": "turn", "direct": "x", "say": "x", "displace": "降级"},
            {"kind": "turn", "displacement_basis": "persona_baseline",
             "baseline": "", "say": "x", "displace": "none"},
            {"kind": "turn", "displacement_basis": "persona_baseline",
             "baseline": "我在。", "say": "我在。", "displace": "none"},
            {"kind": "turn", "displacement_basis": "persona_baseline",
             "baseline": "我想再陪你一会儿。", "say": "茶还没凉。", "displace": "转移"},
        ]
        result = compute(rows)
        self.assertEqual(result.unknown_basis, 2)
        self.assertEqual(result.displaced, 1)
        self.assertEqual(result.displacement_rate, 0.5)
        self.assertEqual(result.inconsistent, 0)
        self.assertFalse(any(level == "warn" and name == "位移率"
                             for level, name, _ in verdicts(result)))

    def test_baseline_metadata_never_becomes_say(self):
        for field in ("baseline", "BASELINE"):
            raw = f"{field}: 私有的人格基准"
            self.assertEqual(safe_say(raw), "")
            self.assertEqual(protocol.clean_phase2(raw), "")
            for parser_type in (StreamParser, protocol.StreamCleanPhase2):
                stream = parser_type()
                emitted = []
                for ch in raw:
                    emitted.extend(stream.feed(ch))
                final = stream.finish()
                if isinstance(final, list):
                    emitted.extend(final)
                self.assertEqual("".join(emitted).strip(), "")

    def test_structured_baseline_is_hidden_at_every_stream_split(self):
        raw = reply("茶还没凉。", baseline="我想再陪你一会儿。", displace="转移")
        for cut in range(len(raw) + 1):
            stream = StreamParser()
            emitted = stream.feed(raw[:cut]) + stream.feed(raw[cut:])
            result = stream.finish()
            self.assertEqual("".join(emitted), "茶还没凉。", cut)
            self.assertEqual(result.head.baseline, "我想再陪你一会儿。")

    def test_same_chunk_head_after_visible_text_never_leaks_baseline(self):
        raw = "我在这里。\n<<<HEAD>>>\nbaseline: 私有基准\n还有私有续行"
        for cut in range(len(raw) + 1):
            for parser_type in (StreamParser, protocol.StreamCleanPhase2):
                stream = parser_type()
                emitted = stream.feed(raw[:cut]) + stream.feed(raw[cut:])
                final = stream.finish()
                if isinstance(final, list):
                    emitted.extend(final)
                self.assertEqual("".join(emitted).strip(), "我在这里。", (cut, parser_type))

    def test_bare_field_inside_say_is_not_emitted(self):
        raw = "<<<SAY>>>我在。\nbaseline: 私有基准\n私有续行<<<TAIL>>>settle: -"
        for cut in range(len(raw) + 1):
            stream = StreamParser()
            emitted = stream.feed(raw[:cut]) + stream.feed(raw[cut:])
            stream.finish()
            self.assertEqual("".join(emitted).strip(), "我在。", cut)

    def test_same_bin_trait_values_change_prompt(self):
        config = cfg("shorekeeper")
        config.role.traits["warmth"].value = 0.70
        before = build_system_prompt(config)
        config.role.traits["warmth"].value = 0.71
        after = build_system_prompt(config)
        self.assertNotEqual(before, after)
        self.assertIn("warmth=0.7000", before)
        self.assertIn("warmth=0.7100", after)

    def test_no_displacement_limit_preserves_persona(self):
        prompt = build_system_prompt(cfg("service"))
        self.assertIn("仍然保持完整的人格语感", prompt)
        self.assertNotIn("想什么说什么", prompt)


class NeutralLayerTests(unittest.TestCase):
    def test_multiround_history_removes_styled_say_and_metadata(self):
        history = [
            {"role": "system", "content": "你是黑海岸的守岸人"},
            {"role": "user", "content": "累了"},
            {"role": "assistant", "content": build_history_entry(
                "黑海岸的灯还亮着。", Head(direct="可以先休息。", baseline="我在这里。"))},
            {"role": "assistant", "content": "无法中性化的旧口吻"},
        ]
        messages = protocol.build_raw_messages(history, "然后呢")
        text = str(messages)
        self.assertIn("可以先休息。", text)
        for private in ("黑海岸", "守岸人", "baseline:", "direct:", "旧口吻"):
            self.assertNotIn(private, text)

    def test_raw_layer_cannot_assign_persona_displacement(self):
        params = protocol.raw_tool(cfg())["function"]["parameters"]
        self.assertNotIn("displace", params["properties"])
        head = protocol.to_head({"raw": "可以先休息。", "displace": "反差"})
        self.assertEqual(head.displace, "none")
        self.assertEqual(head.baseline, "")

    def test_tool_expression_receives_persona_and_context(self):
        messages = protocol.build_phase2_messages(
            Head(direct="先休息", baseline="我在这里。"), [], "累了",
            system_prompt="你是守岸人", context="暗号：潮汐")
        self.assertIn("你是守岸人", messages[0]["content"])
        self.assertIn("暗号：潮汐", messages[0]["content"])
        self.assertIn("我在这里。", str(messages))
        self.assertNotIn("不要说透", messages[0]["content"])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)

    def test_model_watch_completes_turn_and_retains_pending(self):
        engine = TurnEngine(cfg(), self.store, ScriptedLLM([
            reply(tail="pending:\n  claim: 会提起吃饭\n  confidence: 0.6\n  predicts: 会说自己吃过饭"),
            reply(tail="settle: p_0001 | watch | 没涉及吃饭"),
        ]))
        engine.turn("累了")
        result = engine.turn("嗯")
        self.assertEqual(result.settled, [])
        self.assertEqual(len(self.store.pendings()), 1)
        turns = [r for r in self.store.read_ledger() if r["kind"] == "turn"]
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[-1]["probe"]["settle"][0]["verdict"], "WATCH")

    def test_claim_without_evidence_does_not_force_settlement(self):
        config = cfg()
        self.store.add_pending("吃饭", 0.3, "会说吃过饭", "t_0", config.limits)
        engine = TurnEngine(config, self.store, ScriptedLLM([
            reply(tail="settle: p_0001 | confirmed | ")]))
        self.assertEqual(engine.turn("嗯").settled, [])
        self.assertEqual(config.role.bias, 0.0)

    def test_ledger_captures_basis_and_effective_parameters(self):
        engine = TurnEngine(cfg("shorekeeper"), self.store, ScriptedLLM([reply()]))
        engine.turn("累了")
        turn = [r for r in self.store.read_ledger() if r["kind"] == "turn"][0]
        self.assertEqual(turn["displacement_basis"], "persona_baseline")
        self.assertEqual(turn["design_version"], 2)
        self.assertTrue(turn["parameters"]["traits"])
        self.assertEqual(turn["baseline"], "我在。")

    def test_persona_stage_assigns_baseline_and_cannot_overwrite_direct(self):
        def tool(messages, *, tool, **kwargs):
            if tool["function"]["name"] == "raw_reply":
                return {"raw": "可以先休息。", "need": "只想被听", "confidence": 0.6,
                        "constraints": [], "displace": "反差"}
            if tool["function"]["name"] == "check_fidelity":
                return {"verdict": "consistent", "reason": "陪伴意象没有现实事实冲突"}
            return {"settle": [], "pending": None}

        output = reply("茶还没凉。", baseline="我想再陪你一会儿。", displace="转移")
        output = output.replace("direct: 可以先休息。", "direct: 被人格污染的改写")
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                emitted = []
                engine = TurnEngine(cfg(), self.store, ScriptedLLM([output]),
                                    protocol="persona", tool_llm=tool,
                                    stream_llm=lambda messages: iter(output))
                result = engine.turn("别走", on_delta=emitted.append if streaming else None)
                self.assertEqual(result.parsed.head.direct, "可以先休息。")
                self.assertEqual(result.parsed.head.baseline, "我想再陪你一会儿。")
                self.assertEqual(result.parsed.head.displace, "转移")
                self.assertEqual(result.say, "茶还没凉。")
                if streaming:
                    self.assertEqual("".join(emitted), result.say)

    def test_tool_partial_stream_is_not_generated_twice(self):
        def tool(messages, *, tool, **kwargs):
            if tool["function"]["name"] == "inner":
                return {"direct": "休息一下", "baseline": "我在。", "displace": "none"}
            return {"settle": [], "pending": None}

        def interrupted(messages):
            yield "我在这里。\n"
            yield "<<<HEAD>>>baseline: 私有的基准\n"
            raise RuntimeError("断线")

        fallback = ScriptedLLM(["不应该重新生成"])
        emitted = []
        engine = TurnEngine(cfg(), self.store, fallback, protocol="tool",
                            tool_llm=tool, stream_llm=interrupted)
        result = engine.turn("累了", on_delta=emitted.append)
        self.assertEqual(result.say, "我在这里。")
        self.assertEqual("".join(emitted).strip(), result.say)
        self.assertEqual(fallback.calls, [])

    def test_hurt_observation_does_not_require_or_close_pending(self):
        engine = TurnEngine(cfg(), self.store, ScriptedLLM([reply(), reply()]))
        engine.turn("今天好累啊哈哈")
        result = engine.turn("谢谢，我知道了。")
        self.assertEqual(result.settled, [])
        self.assertTrue(engine._repair_book().active())
        observed = [r for r in self.store.read_ledger() if r["kind"] == "relation_observation"]
        self.assertEqual(observed[0]["status"], "tentative")

    def test_elapsed_time_and_repetition_are_not_evidence(self):
        for streak in (0, 1, 10):
            features = extract_features("嗯", minimal_streak=streak, gap_hours=60)
            self.assertFalse(judge(features).settle)
            self.assertFalse(judge_delayed(features, expired=False).settle)
        expired = judge_delayed(features, expired=True)
        self.assertEqual(expired.kind, "EXPIRED")
        self.assertNotIn("不重要", expired.reason)


class GrowthAndRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.config = cfg("shorekeeper")
        self.store.init_traits(self.config.role.traits_state(), persona=self.config.role.name)

    def test_regular_consolidation_does_not_push_traits_or_anchors(self):
        before = self.store.traits()
        for i in range(3):
            self.store.add_judgment(f"用户不想听建议{i}", 0.9, affects={"intervention": -1})
        for i in range(2):
            self.store.push_buffer({"turn_id": f"t_{i}", "user": "我不想听建议", "need": "只想被听"})
        con = Consolidator(self.store)
        with patch.object(con, "push_traits") as traits, patch.object(con, "push_anchors") as anchors:
            result = con.run_once()
        traits.assert_not_called()
        anchors.assert_not_called()
        self.assertEqual(result["traits"], [])
        self.assertEqual(self.store.traits(), before)

    def test_model_affects_are_not_accepted_as_growth_evidence(self):
        text = json.dumps([{"judgment": "用户希望少些建议", "confidence": 0.8,
                            "affects": {"intervention": -1}}], ensure_ascii=False)
        records = Consolidator(self.store, llm=lambda prompt: text)._consolidate([])
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].get("affects"))

    def test_story_identifies_persona_and_distinguishes_speakers(self):
        self.store.self_story_path.write_text(SELF_STORY_HEADER, encoding="utf-8")
        prompts = []
        con = Consolidator(self.store, llm=lambda prompt: prompts.append(prompt) or "我记得对方最近很累。")
        con._story_entry_llm([{"user": "我累了", "say": "我在这里"}], [{"judgment": "用户最近疲惫"}])
        prompt = prompts[0]
        self.assertIn(self.config.role.name, prompt)
        self.assertIn("用户是「对方」，不是你", prompt)
        self.assertIn("用户：我累了", prompt)
        self.assertIn("角色：我在这里", prompt)
        self.assertIn("不要编造变化", prompt)

    def test_story_rebuild_preserves_recorded_entries_and_legacy_prefix(self):
        self.store.self_story_path.write_text("旧的陪伴记录\n", encoding="utf-8")
        con = Consolidator(self.store)
        con._rewrite_story([], [{"judgment": "对方喜欢慢慢说", "confidence": 0.6}])
        expected = self.store.self_story_path.read_text(encoding="utf-8")
        self.store.self_story_path.unlink()
        rebuild_views(self.store)
        self.assertEqual(self.store.self_story_path.read_text(encoding="utf-8"), expected)
        self.assertTrue(expected.startswith("旧的陪伴记录"))

    def test_pending_recovery_keeps_confidence_and_sequence(self):
        pending = self.store.add_pending("会聊睡眠", 0.73, "会报入睡时间", "t_17", self.config.limits,
                                         turn_seq=17)
        self.store.append("pending_add", {"pending_id": pending["id"], "claim": pending["claim"],
                                          "confidence": pending["confidence"], "predicts": pending["predicts"],
                                          "turn_id": pending["turn_id"], "turn_seq": 17})
        self.store._view_path("pending").write_text("broken", encoding="utf-8")
        recovered = self.store.pendings()[0]
        self.assertEqual(recovered["confidence"], 0.73)
        self.assertEqual(recovered["turn_seq"], 17)

    def test_anchor_frequency_survives_missing_view(self):
        con = Consolidator(self.store)
        con._record_anchor_push()
        before = con._anchor_frequency()
        self.store._view_path("anchor_pushes").unlink()
        self.assertAlmostEqual(con._anchor_frequency(), before, places=3)
        rebuild_views(self.store)
        self.assertAlmostEqual(con._anchor_frequency(), before, places=3)


if __name__ == "__main__":
    unittest.main()
