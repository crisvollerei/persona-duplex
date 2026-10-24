"""修复回归测试。

这一批问题有个共同点：它们都发生在**模型不守协议 / 判定有分歧 / 走到备用分支**
的时候。测试里的假数据太规整，正好绕开了这些路径 —— 所以 193 项全绿也抓不到。

跟 test_regression_run1.py 是同一类：真实跑一遍，把踩过的坑钉死。

    · DIRECT 泄露：解析失败时退回原文，把内心独白当成回复发出去
    · 判定优先级倒置：模型 TAIL 说 refuted，账本记 CONFIRMED
    · 流式路径漏检「猜你是……」—— 而那是默认路径
    · pending_max 被实现成总量上限，WATCH 一挂就再也不产生新预言
    · 重启后 p_ 编号从 0 重来，两条不同的预言撞成一个 id
    · rebuild 不还原修复账本，受伤之后就永远不再修复
    · bias 校准只改内存，进程一退就归零
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.config import PersonaConfig                        # noqa: E402
from persona.engine import TurnEngine                            # noqa: E402
from persona.llm import ScriptedLLM, make_reply                  # noqa: E402
from persona.metrics import compute                              # noqa: E402
from persona.parse import parse                                  # noqa: E402
from persona.session import Session, SessionConfig               # noqa: E402
from persona.store import Store, rebuild_views                   # noqa: E402
from persona.stream import StreamParser                          # noqa: E402


def _cfg() -> PersonaConfig:
    return PersonaConfig.load(ROOT / "personas" / "default.toml")


def _head_only(direct: str = "我其实很怕你走") -> str:
    return ("<<<HEAD>>>need: 只是想说说话\nconfidence: 0.6\n"
            f"direct: {direct}\ndisplace: 截断")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()


class TestDirectNeverLeaks(Base):
    """硬约束 #2：DIRECT 永不进入输出。

    唯一的例外是修复 —— 承认损伤是请求修复，不是索取信用。
    """

    def test_head_only_reply_does_not_leak_direct(self):
        """模型只写了 HEAD，没有 SAY。退回原文就把 direct 发了出去。"""
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([_head_only()]))
        res = e.turn("hello")
        self.assertNotIn("我其实很怕你走", res.say)
        self.assertNotIn("<<<", res.say)

    def test_empty_say_does_not_leak_direct(self):
        """写了空 SAY 段，同样不能退回原文。"""
        raw = _head_only() + "\n<<<SAY>>>\n<<<TAIL>>>settle: -"
        res = TurnEngine(_cfg(), self.store, ScriptedLLM([raw])).turn("hello")
        self.assertNotIn("我其实很怕你走", res.say)

    def test_bare_reply_still_passes_through(self):
        """完全没有标记的回复是正常回复，必须照常显示。"""
        res = TurnEngine(_cfg(), self.store,
                         ScriptedLLM(["今天挺累的吧。"])).turn("累死了")
        self.assertEqual(res.say, "今天挺累的吧。")

    def test_say_before_first_marker_is_kept(self):
        """模型先开口再补协议 —— 之前那半句是真内容，不能丢。"""
        raw = "先说一句。" + _head_only() + "\n<<<SAY>>>后面的话\n<<<TAIL>>>settle: -"
        res = TurnEngine(_cfg(), self.store, ScriptedLLM([raw])).turn("hi")
        self.assertIn("先说一句。", res.say)
        self.assertIn("后面的话", res.say)
        self.assertNotIn("我其实很怕你走", res.say)

    def test_parse_never_returns_head_text_as_say(self):
        t = parse(_head_only())
        self.assertEqual(t.say, "")
        self.assertIsNotNone(t.head)

    def test_parse_reattaches_text_after_displace(self):
        """displace 之后才是真正的正文 —— 这段可以当 SAY。"""
        raw = _head_only() + "\n真正的回复在这里"
        self.assertEqual(parse(raw).say, "真正的回复在这里")

    def test_no_displace_means_no_recovery(self):
        """没写 displace 就无法确认 HEAD 结束 —— 宁可空着。"""
        raw = "<<<HEAD>>>need: a\nconfidence: 0.5\ndirect: 秘密"
        self.assertEqual(parse(raw).say, "")


class TestSettlePriority(Base):
    """settle.py 写的优先级是：模型 TAIL > 语域 > 话题。

    实现曾经反过来 —— 启发式先跑并关掉 pending，模型判定再补就无处生效。
    """

    def _seed(self):
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "路上堵吗",
                       "settle: -\npending:\n  claim: 用户想被听\n"
                       "  confidence: 0.7\n  predicts: 会继续说细节"),
        ]))
        e.turn("今天好累，路上堵了一个半小时")
        return e

    def test_model_refuted_beats_literal_overlap(self):
        """用户字面接住了话题，但模型判 refuted —— 以模型为准。"""
        e = self._seed()
        e.llm = ScriptedLLM([make_reply(
            "need: a\nconfidence: 0.7\ndirect: y\ndisplace: none", "嗯。",
            "settle: p_0001 | refuted | 用户只是附和")])
        res = e.turn("是啊，路上堵了一个半小时，还没吃饭")
        self.assertEqual([(s["id"], s["outcome"]) for s in res.settled],
                         [("p_0001", "REFUTED")])

    def test_model_confirmed_beats_topic_left(self):
        """用户换题了，但模型从语义上读出「其实接住了」—— 也要以模型为准。"""
        e = self._seed()
        e.llm = ScriptedLLM([make_reply(
            "need: a\nconfidence: 0.7\ndirect: y\ndisplace: none", "嗯。",
            "settle: p_0001 | confirmed | 用户换了说法但说的是同一件事")])
        res = e.turn("今天骑车回去的")
        self.assertEqual([(s["id"], s["outcome"]) for s in res.settled],
                         [("p_0001", "CONFIRMED")])

    def test_hurt_observation_does_not_override_model_evidence(self):
        """关系观测独立记录，不再关闭或覆盖预测。"""
        e = self._seed()
        e.llm = ScriptedLLM([make_reply(
            "need: a\nconfidence: 0.7\ndirect: y\ndisplace: none", "嗯。",
            "settle: p_0001 | confirmed | 用户说没事")])
        res = e.turn("谢谢，我知道了。")
        self.assertEqual(res.settled[0]["outcome"], "CONFIRMED")
        self.assertTrue(e._repair_book().active())

    def test_no_semantic_evidence_keeps_watching(self):
        e = self._seed()
        e.llm = ScriptedLLM([make_reply(
            "need: a\nconfidence: 0.7\ndirect: y\ndisplace: none", "嗯。",
            "settle: -")])
        res = e.turn("是啊，路上堵了一个半小时")
        self.assertEqual(res.settled, [])
        self.assertEqual(len(self.store.pendings()), 1)

    def test_probe_records_who_settled(self):
        """账本要能分清「模型判的」和「启发式判的」—— 否则调阈值时看不清。"""
        e = self._seed()
        e.llm = ScriptedLLM([make_reply(
            "need: a\nconfidence: 0.7\ndirect: y\ndisplace: none", "嗯。",
            "settle: p_0001 | refuted | 附和")])
        e.turn("是啊，路上堵了")
        by = [s.get("by") for s in e._probe["settle"]]
        self.assertIn("model", by)


class TestStreamingDetectsReadAloud(Base):
    """「猜你是……」是硬失败，流式和整段必须一样能抓到。

    之前 StreamParser.finish 只调了 detect_declarations，漏了 read_aloud ——
    而 chat 默认走流式，等于这条检查在默认路径上从未生效。
    """

    RAW = ("<<<HEAD>>>need: a\nconfidence: 0.6\ndirect: x\ndisplace: none\n"
           "<<<SAY>>>猜你不想说。\n\n换个话题吧。\n<<<TAIL>>>settle: -")

    def test_stream_parser_flags_read_aloud(self):
        """流式路径必须和整段路径一样能观测到「猜」。

        注意：只观测、不改写。早先试过在流式层删句子，结果把紧跟的
        「不对就当我没说」一起删了 —— 恰好删掉退路，位移被削平。
        """
        sp = StreamParser()
        for ch in self.RAW:
            sp.feed(ch)
        self.assertTrue(sp.finish().read_aloud)

    def test_engine_streaming_path_warns(self):
        def gen(messages):
            for i in range(0, len(self.RAW), 4):
                yield self.RAW[i:i + 4]

        e = TurnEngine(_cfg(), self.store, ScriptedLLM([self.RAW]), stream_llm=gen)
        res = e.turn("hi", on_delta=lambda p: None)
        self.assertTrue(any("读" in w for w in res.warnings))
        # 句子原样保留 —— 模型说的话不归解析层改
        self.assertIn("猜你", res.say)


class TestStreamingHoldback(Base):
    def test_head_marker_in_say_is_held_back(self):
        """模型在 SAY 里重开协议 —— 不能把 <<<HEAD>>> 和后面的 direct 吐出去。"""
        sp = StreamParser()
        out = []
        for ch in "<<<SAY>>>你好<<<HEAD>>>need: a\ndirect: 秘密":
            out.extend(sp.feed(ch))
        self.assertNotIn("<", "".join(out))
        self.assertEqual("".join(out), "你好")

    def test_partial_head_marker_held_back(self):
        sp = StreamParser()
        out = []
        out.extend(sp.feed("<<<SAY>>>你好。"))
        out.extend(sp.feed("<<<HE"))
        self.assertEqual("".join(out), "你好。")
        out.extend(sp.feed("AD>>>direct: 秘密"))
        self.assertEqual("".join(out), "你好。")
        self.assertNotIn("秘密", sp.finish().say)

    def test_repeated_say_marker_never_printed(self):
        """模型在 SAY 里又写一次 <<<SAY>>> —— 标记本身也不能吐给用户。"""
        sp = StreamParser()
        out = []
        for ch in "<<<SAY>>>第一段<<<SAY>>>第二段<<<TAIL>>>settle: -":
            out.extend(sp.feed(ch))
        self.assertEqual("".join(out), "第一段第二段")
        self.assertNotIn("<<<", sp.finish().say)


class TestStreamingFailure(Base):
    def test_no_duplicate_output_after_partial_stream(self):
        """流吐了一半断掉，退回整段重来会让用户看到重复的半句。"""
        raw = make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none", "好的。")

        def broken(messages):
            yield "<<<SAY>>>好的。"
            raise RuntimeError("connection reset")

        seen: list[str] = []
        llm = ScriptedLLM([raw])
        e = TurnEngine(_cfg(), self.store, llm, stream_llm=broken)
        res = e.turn("hi", on_delta=lambda p: seen.append(p))
        self.assertEqual("".join(seen), res.say)
        self.assertEqual(llm.calls, [])

    def test_no_retry_when_nothing_emitted(self):
        """默认即使首片前断线，也不另发整段请求。"""
        raw = make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none", "好的。")

        def broken(messages):
            yield ""
            raise RuntimeError("连接在首片就断了")

        llm = ScriptedLLM([raw])
        e = TurnEngine(_cfg(), self.store, llm, stream_llm=broken)
        seen = []
        with self.assertRaisesRegex(RuntimeError, "首片"):
            e.turn("hi", on_delta=seen.append)
        self.assertEqual(seen, [])
        self.assertEqual(llm.calls, [])
        self.assertFalse(any(row["kind"] == "turn" for row in self.store.read_ledger()))


class TestPendingLimits(Base):
    def test_pending_max_is_per_turn_not_total(self):
        """按总数卡的话，一条 WATCH 就能让预测链路停摆 72 小时。"""
        cfg = _cfg()
        self.assertIsNotNone(self.store.add_pending("a", 0.6, "x", "t1", cfg.limits))
        self.assertIsNone(self.store.add_pending("b", 0.6, "y", "t1", cfg.limits))
        self.assertIsNotNone(self.store.add_pending("c", 0.6, "z", "t2", cfg.limits))

    def test_total_cap_still_exists(self):
        cfg = _cfg()
        for i in range(cfg.limits.pending_total_max):
            self.store.add_pending(f"c{i}", 0.6, "p", f"t{i}", cfg.limits)
        self.assertIsNone(
            self.store.add_pending("overflow", 0.6, "p", "t99", cfg.limits))

    def test_ids_continue_after_restart(self):
        """重启后 p_ 从 0 重来，两条不同的预言会撞成一个 id。"""
        cfg = _cfg()
        first = self.store.add_pending("a", 0.6, "x", "t1", cfg.limits)
        self.assertEqual(first["id"], "p_0001")
        self.store.append("pending_add", {
            "pending_id": "p_0001", "claim": "a", "confidence": 0.6,
            "predicts": "x", "turn_id": "t1", "turn_seq": 1,
        })
        reopened = Store(self.store.root)
        second = reopened.add_pending("b", 0.6, "y", "t2", cfg.limits)
        self.assertNotEqual(second["id"], "p_0001")
        self.assertEqual(len({p["id"] for p in reopened.pendings()}), 2)


class TestRebuildFidelity(Base):
    """state/ 坏了能重建 —— 但重建必须真的还原，不能只是不报错。"""

    def _hurt(self, store: Store):
        e = TurnEngine(_cfg(), store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: c\n"
                               "  confidence: 0.7\n  predicts: p"),
            make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none", "嗯。"),
        ]))
        e.turn("我很累。")
        e.turn("谢谢，我知道了。")

    def test_repair_book_survives_rebuild(self):
        """修复账本不重放的话，受伤之后系统就再也不知道自己要修复什么。"""
        self._hurt(self.store)
        before = self.store.load_view("repair", {}).get("items", [])
        self.assertTrue(before)
        for p in self.store.state_dir.glob("*"):
            p.unlink()
        rebuild_views(self.store)
        self.assertEqual(self.store.load_view("repair", {}).get("items", []), before)

    def test_unresolved_survives_rebuild(self):
        self._hurt(self.store)
        self.assertTrue(self.store.relation()["unresolved"])
        for p in self.store.state_dir.glob("*"):
            p.unlink()
        rebuild_views(self.store)
        self.assertTrue(self.store.relation()["unresolved"])

    def test_pending_turn_seq_survives_rebuild(self):
        """turn_seq 丢了，延迟结算就永远判不出来。"""
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: c\n"
                               "  confidence: 0.6\n  predicts: p"),
        ]))
        e.turn("hi")
        for p in self.store.state_dir.glob("*"):
            p.unlink()
        rebuild_views(self.store)
        self.assertEqual(self.store.pendings()[0]["turn_seq"], 1)


class TestBiasPersistence(Base):
    """预测命中不再充当人格成长分数，历史变更仍可分析。"""

    def _learn(self, store: Store, cfg: PersonaConfig):
        e = TurnEngine(cfg, store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: c\n"
                               "  confidence: 0.3\n  predicts: p"),
            make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none",
                       "对，就是这样", "settle: p_0001 | confirmed | 接住了"),
        ]))
        e.turn("我很累。")
        e.turn("对，就是这样，我也是这么想的")

    def test_prediction_success_does_not_change_bias(self):
        cfg = _cfg()
        self._learn(self.store, cfg)
        recs = [r for r in self.store.read_ledger() if r["kind"] == "bias_update"]
        self.assertEqual(recs, [])
        self.assertEqual(cfg.role.bias, 0.0)

    def test_legacy_bias_update_not_silently_applied(self):
        cfg = _cfg()
        self._learn(self.store, cfg)
        self.store.append("bias_update", {"before": 0.0, "after": 0.005})
        fresh = _cfg()
        TurnEngine(fresh, Store(self.store.root), ScriptedLLM([]))
        self.assertEqual(fresh.role.bias, 0.0)

    def test_ledger_replays_to_same_bias(self):
        """历史 bias 末值仍可查，但不会按新初值重算。"""
        from persona.metrics import final_bias

        cfg = _cfg()
        self._learn(self.store, cfg)
        self.store.append("bias_update", {"before": 0.0, "after": 0.005})
        self.assertAlmostEqual(final_bias(self.store.read_ledger(), initial=0.2), 0.005)


class TestMetricsCounters(Base):
    def test_expired_counted(self):
        """过期走的是 pending_expire，metrics 只认 pending_settle 的话永远是 0。"""
        cfg = _cfg()
        p = self.store.add_pending("a", 0.6, "x", "t1", cfg.limits)
        self.store._save_pendings([{**p, "created": "2020-01-01T00:00:00+08:00"}])
        TurnEngine(cfg, self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none", "嗯。"),
        ])).turn("嗯")
        self.assertEqual(compute(self.store.read_ledger()).expired, 1)

    def test_legacy_missing_baseline_is_unknown(self):
        """不能用旧 direct 冒充人格基准。"""
        m = compute([{"kind": "turn", "say": "嗯", "direct": "",
                      "displace": "降级"}])
        self.assertEqual(m.inconsistent, 0)
        self.assertEqual(m.unknown_basis, 1)


class TestRepairTiming(Base):
    def test_same_turn_repair_not_claimed(self):
        """修复不在受伤的当轮发 —— 当场道歉是在处理自己的尴尬。"""
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: c\n"
                               "  confidence: 0.7\n  predicts: p"),
            make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none",
                       "刚才那句是不是有点重。"),
        ]))
        e.turn("我很累。")
        res = e.turn("谢谢，我知道了。")
        self.assertEqual(res.settled, [])
        self.assertEqual(len(self.store.pendings()), 1)
        self.assertIsNone(res.repair_opened, "受伤当轮就认领了修复")
        self.assertEqual(e._repair_book().items[0].state, "open")

    def test_next_turn_repair_is_claimed(self):
        e = TurnEngine(_cfg(), self.store, ScriptedLLM([
            make_reply("need: a\nconfidence: 0.7\ndirect: x\ndisplace: none",
                       "嗯。", "settle: -\npending:\n  claim: c\n"
                               "  confidence: 0.7\n  predicts: p"),
            make_reply("need: a\nconfidence: 0.6\ndirect: y\ndisplace: none", "嗯。"),
            make_reply("need: a\nconfidence: 0.6\ndirect: z\ndisplace: none",
                       "刚才那句是不是有点重。"),
        ]))
        e.turn("我很累。")
        e.turn("谢谢，我知道了。")
        res = e.turn("……")
        self.assertIsNotNone(res.repair_opened)
        self.assertEqual(res.repair_opened["action"], "attempted")


class TestSayStreamResult(unittest.TestCase):
    def test_result_available_after_iteration(self):
        raw = make_reply("need: a\nconfidence: 0.6\ndirect: x\ndisplace: none",
                         "嗯，然后呢。")

        def gen(messages):
            for i in range(0, len(raw), 5):
                yield raw[i:i + 5]

        with tempfile.TemporaryDirectory() as tmp:
            with Session(_cfg(), Store(Path(tmp)), ScriptedLLM(["不应被调用"]),
                         stream_llm=gen,
                         options=SessionConfig(consolidate_interval=999)) as s:
                turn = s.say_stream("hi")
                self.assertEqual("".join(turn), "嗯，然后呢。")
                self.assertIsNotNone(turn.result)
                self.assertEqual(turn.result.say, "嗯，然后呢。")


if __name__ == "__main__":
    unittest.main(verbosity=2)
