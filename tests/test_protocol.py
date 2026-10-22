"""两阶段 tool 协议的测试。

这一层的价值不在「格式更整齐」，在**因果**：

    phase 1 只产出 direct，没有任何输出可见文本的途径
    phase 2 只拿 direct，产出 say

所以 phase 1 的那个 direct 不可能是为了配某句 say 而倒推的。
文本协议做不到 —— direct 和 say 在同一次生成里。

但两阶段也不是万能：phase 2 仍是自由生成，可能把 direct 原样复述。
所以有 displacement_ok() 做检查，不合格就记下来。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona import protocol                                   # noqa: E402
from persona.config import DISPLACE_NONE, PersonaConfig        # noqa: E402
from persona.engine import TurnEngine                           # noqa: E402
from persona.llm import ScriptedLLM, make_reply                # noqa: E402
from persona.parse import Head                                  # noqa: E402
from persona.store import Store                                 # noqa: E402


def _cfg() -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


class FakeToolLLM:
    """假的 tool 模型。

    一次 turn 会调它两次：先 inner（内心），再 settle_turn（结算）。
    按 tool 名字分别取脚本，这样测试能精确控制两段。
    """

    def __init__(self, inners, tails=None):
        self.inners = list(inners)
        self.tails = list(tails) if tails is not None else None
        self.calls: list[list[dict]] = []
        self.tool_names: list[str] = []

    def __call__(self, messages, *, tool=None, max_tokens=None):
        name = (tool or {}).get("function", {}).get("name", "")
        self.tool_names.append(name)
        self.calls.append(messages)
        if name == "settle_turn":
            if self.tails is None:
                return {"settle": [], "pending": None}
            if not self.tails:
                return {"settle": [], "pending": None}
            item = self.tails.pop(0)
        else:
            if not self.inners:
                raise RuntimeError("脚本用尽")
            item = self.inners.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _inner(direct="你其实早就想走了，只是在等一个人说可以。",
           displace="降级", need="要被推一把", conf=0.6):
    return {"need": need, "confidence": conf, "direct": direct, "displace": displace}


class TestSchema(unittest.TestCase):
    def test_need_enum_comes_from_config(self):
        """need 的词表来自人格配置 —— 写死会让 guarded/service 的枚举被拒。"""
        for name in ("default", "guarded", "service"):
            cfg = PersonaConfig.load(ROOT / "personas" / f"{name}.toml")
            tool = protocol.inner_tool(cfg)
            enum = tool["function"]["parameters"]["properties"]["need"]["enum"]
            self.assertEqual(enum, list(cfg.needs), f"{name} 的 need 枚举对不上")

    def test_schema_is_strict(self):
        tool = protocol.inner_tool(_cfg())
        self.assertTrue(tool["function"]["strict"])
        params = tool["function"]["parameters"]
        self.assertFalse(params["additionalProperties"])
        self.assertEqual(set(params["required"]),
                         {"need", "confidence", "direct", "displace"})


class TestParseInner(unittest.TestCase):
    def test_normal(self):
        h = protocol.parse_inner(_inner())
        self.assertEqual(h.displace, "降级")
        self.assertAlmostEqual(h.confidence, 0.6)

    def test_unknown_displace_falls_back(self):
        """不同端点的 schema 实现质量参差，枚举值不认识要能活着回来。"""
        h = protocol.parse_inner(_inner(displace="乱写的"))
        self.assertEqual(h.displace, DISPLACE_NONE)

    def test_confidence_clamped(self):
        self.assertEqual(protocol.parse_inner(_inner(conf=5)).confidence, 1.0)
        self.assertEqual(protocol.parse_inner(_inner(conf=-1)).confidence, 0.0)

    def test_missing_fields_survive(self):
        h = protocol.parse_inner({})
        self.assertEqual(h.direct, "")
        self.assertEqual(h.displace, DISPLACE_NONE)


class TestDisplacementCheck(unittest.TestCase):
    def test_paraphrase_rejected(self):
        """声称位移，但 say 基本是 direct 的复述 —— 不合格。"""
        inner = Head(direct="不是矫情。三天睡四小时还问自己是不是矫情，这本身就说明你对自己太狠了。",
                     displace="降级")
        ok, why = protocol.displacement_ok(inner, "不矫情。三天睡四小时还怀疑自己矫情，这本身就说明你对自己太狠了。")
        self.assertFalse(ok)
        self.assertIn("复述", why)

    def test_real_displacement_passes(self):
        inner = Head(direct="你其实早就想走了，只是在等一个人说可以。", displace="降级")
        ok, _ = protocol.displacement_ok(inner, "你都想过哪些了。")
        self.assertTrue(ok)

    def test_empty_say_rejected(self):
        ok, why = protocol.displacement_ok(Head(direct="x", displace="none"), "")
        self.assertFalse(ok)

    def test_missing_direct_rejected(self):
        ok, why = protocol.displacement_ok(Head(direct="", displace="降级"), "嗯。")
        self.assertFalse(ok)
        self.assertIn("direct", why)

    def test_none_displace_is_permissive(self):
        inner = Head(direct="今天先别想事，去睡吧。", displace="none")
        self.assertTrue(protocol.displacement_ok(inner, "早点睡。")[0])


class TestTwoPhaseEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_phase1_has_no_say_channel(self):
        """phase 1 的 schema 里不能有任何可见文本字段。

        这是整个方案成立的前提：只要 phase 1 有地方写 say，
        direct 就可能是为了配那句话倒推出来的。
        """
        tool = protocol.inner_tool(_cfg())
        props = set(tool["function"]["parameters"]["properties"])
        self.assertNotIn("say", props)
        self.assertNotIn("tail", props)
        self.assertEqual(props, {"need", "confidence", "direct", "displace"})

    def test_phase2_receives_direct(self):
        """phase 2 必须真的看到 direct —— 否则因果链断在这里。

        注意 phase 2 走的是 self.llm（纯文本），不是 tool_llm。
        所以要看的是 llm 收到的消息，不是 tool_llm 的。
        """
        seen: list[list[dict]] = []

        def capture_llm(messages):
            seen.append(messages)
            return "今天有点累，改天吧。"

        tool = FakeToolLLM([_inner(direct="我不太想聊这个。")])
        e = TurnEngine(_cfg(), self.store, capture_llm,
                       tool_llm=tool, protocol="tool")
        e.turn("……")
        self.assertTrue(seen, "phase 2 没调用 llm")
        joined = str(seen[0])
        self.assertIn("我不太想聊这个", joined, "phase 2 没拿到 direct")

    def test_end_to_end(self):
        tool = FakeToolLLM([_inner()])
        # phase 2 只回一句话 —— 没有标记，不需要解析
        e = TurnEngine(_cfg(), self.store,
                       ScriptedLLM(["你都想过哪些了。"]),
                       tool_llm=tool, protocol="tool")
        res = e.turn("我觉得我该换个工作")
        self.assertEqual(res.say, "你都想过哪些了。")
        self.assertEqual(res.parsed.head.displace, "降级")
        self.assertIsNotNone(res.parsed.tail, "tail 必须存在（哪怕是空骨架）")

    def test_phase1_failure_falls_back_to_text(self):
        """兜底机制不能自己变成新的故障点。"""
        tool = FakeToolLLM([RuntimeError("网络断了")])
        e = TurnEngine(_cfg(), self.store,
                       ScriptedLLM([make_reply(
                           "need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                           "文本协议救回来了。", "settle: -")]),
                       tool_llm=tool, protocol="tool")
        res = e.turn("hi")
        self.assertEqual(res.say, "文本协议救回来了。")
        self.assertTrue(any("退回文本协议" in w for w in res.warnings))

    def test_empty_direct_falls_back(self):
        tool = FakeToolLLM([_inner(direct="")])
        e = TurnEngine(_cfg(), self.store,
                       ScriptedLLM([make_reply(
                           "need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                           "兜底了。", "settle: -")]),
                       tool_llm=tool, protocol="tool")
        res = e.turn("hi")
        self.assertEqual(res.say, "兜底了。")
        self.assertTrue(any("direct 为空" in w for w in res.warnings))

    def test_no_stream_still_works(self):
        tool = FakeToolLLM([_inner()])
        e = TurnEngine(_cfg(), self.store, ScriptedLLM(["你都想过哪些了。"]),
                       tool_llm=tool, protocol="tool")
        self.assertEqual(e.turn("x").say, "你都想过哪些了。")

    def test_with_streaming_emits_pieces(self):
        tool = FakeToolLLM([_inner()])
        raw = "你都想过哪些了。"

        def gen(messages):
            for i in range(0, len(raw), 3):
                yield raw[i:i + 3]

        seen: list[str] = []
        e = TurnEngine(_cfg(), self.store, ScriptedLLM(["不该被调用"]),
                       stream_llm=gen, tool_llm=tool, protocol="tool")
        res = e.turn("x", on_delta=seen.append)
        self.assertEqual("".join(seen), raw)
        self.assertEqual(res.say, raw)

    def test_paraphrase_flagged_but_say_kept(self):
        """phase 2 复述了 direct —— 记警告，但不能把已经吐出去的话撤回。"""
        tool = FakeToolLLM([_inner(
            direct="不是矫情。三天睡四小时还问自己是不是矫情，这本身就说明你对自己太狠了。",
            displace="降级")])
        same = "不是矫情。三天睡四小时还问自己是不是矫情，这本身就说明你对自己太狠了。"
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([same]),
                       tool_llm=tool, protocol="tool")
        res = e.turn("我是不是有点矫情")
        self.assertEqual(res.say, same)
        self.assertTrue(any("不合格" in w for w in res.warnings))

    def test_ledger_records_protocol(self):
        tool = FakeToolLLM([_inner()])
        e = TurnEngine(_cfg(), self.store, ScriptedLLM(["你都想过哪些了。"]),
                       tool_llm=tool, protocol="tool")
        e.turn("x")
        turns = [r for r in self.store.read_ledger() if r["kind"] == "turn"]
        self.assertEqual(turns[0]["probe"]["protocol"]["mode"], "tool")


class TestTailPhase(unittest.TestCase):
    """第三阶段：结算 + 新预言。

    两阶段的 phase 1/2 只管「这一轮说什么」，不含 TAIL 语义。
    不补这一步，pending 永远挂不上 —— 整个校准账本会停摆。
    实测过：没有 phase 3 时 20 轮下来 pending_add = 0。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_pending_gets_added(self):
        tool = FakeToolLLM(
            [_inner()],
            tails=[{"settle": [], "pending": {
                "claim": "下一句会继续讲今天的细节",
                "confidence": 0.6,
                "predicts": "再补一件今天发生的事"}}],
        )
        e = TurnEngine(_cfg(), self.store, ScriptedLLM(["你都想过哪些了。"]),
                       tool_llm=tool, protocol="tool")
        res = e.turn("我觉得我该换个工作")
        self.assertIsNotNone(res.pending_added, "phase 3 没挂上预言")
        self.assertEqual(len(self.store.pendings()), 1)

    def test_settle_closes_previous_pending(self):
        cfg = _cfg()
        p = self.store.add_pending("用户会继续讲细节", 0.6, "补一件具体事",
                                   "t_0", cfg.limits)
        tool = FakeToolLLM(
            [_inner()],
            tails=[{"settle": [{"id": p["id"], "outcome": "confirmed",
                                "evidence": "用户接着说了一件具体的事"}],
                    "pending": None}],
        )
        e = TurnEngine(cfg, self.store, ScriptedLLM(["嗯。"]),
                       tool_llm=tool, protocol="tool")
        res = e.turn("后来还有一件事")
        self.assertEqual([(s["id"], s["outcome"]) for s in res.settled],
                         [(p["id"], "CONFIRMED")])

    def test_settle_tool_has_no_say_channel(self):
        """第三阶段也不该有输出可见文本的通道。"""
        tool = protocol.tail_tool([{"id": "p_0001", "claim": "c", "predicts": "p"}])
        props = set(tool["function"]["parameters"]["properties"])
        self.assertEqual(props, {"settle", "pending"})

    def test_settle_id_limited_to_existing(self):
        """模型编一个不存在的 id 是最常见的失败 —— schema 层挡住。"""
        tool = protocol.tail_tool([{"id": "p_0001", "claim": "c", "predicts": "p"}])
        enum = tool["function"]["parameters"]["properties"]["settle"]["items"][
            "properties"]["id"]["enum"]
        self.assertEqual(enum, ["p_0001"])

    def test_tail_failure_does_not_break_turn(self):
        """phase 3 失败只是这一轮没学到东西，不能影响已发出的回复。"""
        tool = FakeToolLLM([_inner()], tails=[RuntimeError("网络断了")])
        e = TurnEngine(_cfg(), self.store, ScriptedLLM(["你都想过哪些了。"]),
                       tool_llm=tool, protocol="tool")
        res = e.turn("x")
        self.assertEqual(res.say, "你都想过哪些了。")
        self.assertTrue(any("结算阶段失败" in w for w in res.warnings))

    def test_parse_tail_drops_bad_items(self):
        tail = protocol.parse_tail({
            "settle": [
                {"id": "p_1", "outcome": "confirmed", "evidence": "e"},
                {"id": "", "outcome": "confirmed", "evidence": "e"},
                {"id": "p_2", "outcome": "maybe", "evidence": "e"},
                "不是字典",
            ],
            "pending": {"claim": "", "confidence": 0.5, "predicts": "p"},
        })
        self.assertEqual(len(tail.settle), 1)
        self.assertIsNone(tail.pending, "claim 为空的预言不该挂")


class TestAutoUpgrade(unittest.TestCase):
    def test_auto_starts_as_text(self):
        e = TurnEngine(_cfg(), self.store if hasattr(self, "store") else
                       Store(Path(tempfile.mkdtemp())), ScriptedLLM([]),
                       protocol="auto")
        self.assertEqual(e.protocol, "auto")

    def test_auto_upgrades_after_repeated_failures(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            store = Store(Path(tmp.name))
            # 前两轮都不守协议（裸文本 → 缺 HEAD），应触发升级
            e = TurnEngine(_cfg(), store,
                           ScriptedLLM(["我完全没按协议写。", "又没按协议写。"]),
                           tool_llm=FakeToolLLM([]), protocol="auto")
            e.turn("a")
            e.turn("b")
            self.assertEqual(e.protocol, "tool", "连续失效后没有升级")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
