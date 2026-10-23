"""单轮编排：组装 → 生成 → 解析 → 落账。

一次生成，产出「说出去的话」+「状态变更」。
观测在线写，结算下一轮填。

三条生成路径（由 protocol 决定，见 protocol.py）：

    text     一次生成，HEAD → SAY → TAIL。因果顺序靠 token 位置保证
    tool     _generate_two_phase：phase1 出 HEAD、phase2 出 SAY、phase3 结算
    persona  _generate_persona：raw（看不到人设）→ 人设化 → 结算

三条路径的共同底线：**direct 永不进用户可见输出**。
统一在 parse.safe_say 兜底，任何一条路径都不例外。

另一条贯穿全部路径的约定：**已经吐给用户的内容不能重来**。
兜底回退时必须先看「是否已经输出过」，否则用户会看到两遍，
而账本只记一遍 —— 账本在说谎，比不给回复更糟。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from .parse import SettleItem

from .config import DISPLACE_NONE, PersonaConfig
from .parse import (ParsedTurn, Tail, has_exit_offer, parse, plain_text,
                    safe_say)
from .prompt import (
    build_context_block,
    build_history_entry,
    build_system_prompt,
    build_user_turn,
    stance_hint,
)
from .repair import (    RepairBook,
    is_repair_attempt,
    judge_repair_response,
    pick_level,
    repair_level,
)
from .settle import (
    CONFIRMED,
    CONFIRMED_DELAYED,
    OUTCOME_LABEL,
    REFUTED,
    REFUTED_DELAYED,
    SUSPECTED_HURT,
    WATCH,
    TurnFeatures,
    extract_features,
    formality_score,
    judge,
    judge_delayed,
    _is_minimal_reply,
)
from .similarity import DEFAULT as DEFAULT_SIMILARITY
from .store import Store, now_iso, parse_iso

# 在线可写的字段白名单。慢层（judgments / self_story）永远不在这里。
ONLINE_WRITABLE = {"closeness", "unresolved", "sore_points", "inside_jokes"}


@dataclass
class TurnResult:
    turn_id: str
    say: str
    parsed: ParsedTurn
    stance: str
    displaced: bool
    settled: list[dict] = field(default_factory=list)
    pending_added: dict | None = None
    relation_changed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    repair_opened: dict | None = None
    repair_closed: dict | None = None

    @property
    def display(self) -> str:
        return self.say


class TurnEngine:
    """一轮 = 一次生成。不重试、不并行、不阻塞。"""

    def __init__(
        self,
        cfg: PersonaConfig,
        store: Store,
        llm: Callable[[list[dict]], str],
        *,
        similarity=DEFAULT_SIMILARITY,
        stream_llm: Callable[[list[dict]], Iterator[str]] | None = None,
        judge=None,
        protocol: str = "text",
        tool_llm=None,
    ):
        """protocol: "text"（默认）| "tool" | "auto"

        "auto" = 先用 text，一旦这一轮出现协议失效（缺 HEAD/缺 TAIL/
        direct 与 say 无法区分），自动切到 tool 并记住，之后都用 tool。
        这是「兜底」的实现方式：正常情况不多付延迟，出问题才升级。
        """
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.stream_llm = stream_llm
        self.protocol = protocol
        self.tool_llm = tool_llm
        self._protocol_failures = 0
        self.sim = similarity
        self.judge = judge
        self.history: list[dict] = []
        self._last_user = ""
        self._last_say = ""
        self._topic: list[str] = []
        self._pending_topic: dict[str, list[str]] = {}
        self._turn_seq = 0
        self._last_formality_rise = 0.0
        # 上一轮的 need，用来触发下一轮的分场景句式注入
        self._last_need = ""
        # 每条预言被「极短回应」晾过几次。连续两次就判 refuted ——
        # 否则「嗯」「算了」这种回复会让预言永远关不掉。
        self._minimal_streak: dict[str, int] = {}
        # 上一轮开出修复的轮次号。同一轮只开一条 —— 用户被伤到一次，
        # 不是每条预言伤一次。
        self._repair_opened_turn = -1
        self._restore_bias()

    def _sync_traits(self) -> None:
        """把账本里的人格参数同步到内存配置。

        每轮调用一次。成本是一次账本读取 + 几个浮点赋值 ——
        相对一次 LLM 调用可以忽略。

        为什么必须每轮同步：固化线程和对话线程是两个线程，
        后者持有 cfg 对象。固化改了账本，cfg 不会自己变。
        """
        state = self.store.traits()
        if state:
            self.cfg.role.load_traits_state(state)

    def _restore_bias(self) -> None:
        """把上次会话学到的 bias 和轮次接回来。

        bias 是唯一会跨轮累积的学习量。不恢复的话，每次启动都退回 TOML 里的
        初值，校准永远在头几轮原地打转。
        """
        for rec in reversed(self.store.read_ledger()):
            if rec.get("kind") != "bias_update":
                continue
            try:
                self.cfg.role.bias = float(rec["after"])
            except (KeyError, TypeError, ValueError):
                pass
            break
        # turn_seq 只在内存里，重启归零会让「延迟结算」永远判不成立
        self._turn_seq = sum(
            1 for r in self.store.read_ledger() if r.get("kind") == "turn"
        )

    # ------------------------------------------------------------ 主流程

    def turn(self, user_text: str,
             on_delta: Callable[[str], None] | None = None) -> TurnResult:
        """跑一轮。

        on_delta 传了就边走流式边回调 —— 用户不必盯着空白等整段生成完。
        """
        turn_id = self.store.next_id("t")
        warnings: list[str] = []
        self._turn_seq += 1
        turn_started = time.monotonic()
        self._probe = {}       # 本轮原始观测值，落账用

        # ---- 1. 修复反应判定。这个必须同步：它决定上下文里要不要放
        #        「你伤到对方了」的提示。但它只看语域，不调模型。
        repair_closed = self._judge_repair_response(user_text, warnings)

        # ---- 2. 把话题判定发射出去。注意这里只是发射，不等 ——
        #        它会在主模型生成的同时并行跑，生成完再收。
        pendings_snapshot = self.store.pendings()
        judge_handle = self._submit_topic_judge(user_text, pendings_snapshot)

        # ---- 3. 组装上下文。此时预言都还挂着 —— 模型需要在上下文里
        #        看见它们，才能在 TAIL 里给出自己的判定。
        #
        # 先把人格参数从账本同步回来。固化线程是另一个线程，它写账本，
        # 但不会改这里的 cfg 对象 —— 不同步的话，中期层和长期层的推动
        # 只落在账本上，对实际对话零影响，要等下次重启才生效。
        # 实测踩过：账本里 formality 已到 0.70（跨过 0.65 分界），
        # 线上 prompt 里「用词正式一些」始终不出现。
        self._sync_traits()

        relation = self.store.relation()
        pendings = pendings_snapshot
        judgments = self.store.judgments()

        # 先算出这一轮大概是什么 need —— 用它触发分场景句式注入。
        # 这里读的是**上一轮**的 need，因为这一轮还没生成。
        # 差一轮不影响：句式是语感参考，不是精确控制。
        need_hint = self._last_need
        first_turn = self._turn_seq <= 1
        sys_prompt = build_system_prompt(self.cfg, need=need_hint,
                                         first_turn=first_turn)
        ctx = build_context_block(
            self.cfg,
            relation=relation,
            pendings=pendings,
            judgments=judgments,
            style_note=self._style_note(),
            repair_note=self._repair_note(),
        )

        prior_conf = self._prior_confidence()
        stance = self.cfg.stance_for(prior_conf)

        messages = [
            {"role": "system", "content": sys_prompt + "\n\n" + ctx},
            *self.history[-12:],
            {"role": "user", "content": stance_hint(self.cfg, prior_conf)
                                    + "\n\n" + build_user_turn(user_text)},
        ]

        # ---- 4. 生成。两条路：
        #        text  = 单次生成三段协议（默认）
        #        tool  = 两阶段，direct 结构上先于 say（兜底）
        #        兜底触发时 tool 会自己退回 text，不会把这一轮搞丢。
        if self.protocol == "persona":
            raw, parsed = self._generate_persona(
                sys_prompt, ctx, user_text, messages, on_delta, warnings
            )
        elif self.protocol == "tool":
            raw, parsed = self._generate_two_phase(
                sys_prompt, ctx, user_text, stance, messages, on_delta, warnings
            )
        else:
            raw, parsed = self._generate(messages, on_delta)

        # ---- 5. 收判定。grace 很短，拿不到就立刻用字面兜底，
        #        所以这里不会拖慢对话。
        collect_started = time.monotonic()
        topic_verdicts, topic_source = self._collect_topic_judge(
            judge_handle, user_text, pendings_snapshot, turn_started
        )
        self._probe["judge"] = {
            "source": topic_source,
            # judge_elapsed：判定器真正花了多久（含在生成期间并行跑掉的那段）
            # elapsed：整轮耗时。两者不能混用 —— 拿整轮耗时比 grace，
            # 会把「生成慢」误报成「判定超时」。
            "judge_elapsed": round(_judge_elapsed(judge_handle), 3),
            "collect_wait": round(time.monotonic() - collect_started, 3),
            "elapsed": round(time.monotonic() - turn_started, 3),
            "verdicts": {k: v.verdict for k, v in topic_verdicts.items()},
        }
        if topic_source == "late":
            warnings.append("话题判定未在 grace 内返回，本轮用字面兜底")

        # ---- 6. 结算。模型自己的 TAIL 判定是主路径，必须和启发式一起送进去，
        #        否则启发式先关掉 pending，模型判定就无处生效。
        model_settle = {it.pending_id: it for it in (parsed.tail.settle if parsed.tail else [])}
        model_watch = set(parsed.tail.watch) if parsed.tail else set()
        settled = self._settle_with(user_text, warnings, topic_verdicts,
                                    model_settle=model_settle,
                                    model_watch=model_watch)

        warnings.extend(parsed.errors)

        # auto 模式：协议失效就升级到 tool，之后不再降级回来。
        #
        # 两条修过的判据：
        #  · 失效 = parsed.errors 非空（不只 head is None）。
        #    README 明说「缺 TAIL」算失效，而 tool 协议本来就是因为
        #    「缺 TAIL」才需要兜底的 —— 旧判据永远升不了级。
        #  · 成功一轮要清零，否则 fail->success->fail 也会升级，
        #    那不是「连续」失效。
        if self.protocol == "auto":
            if parsed.errors:
                self._protocol_failures += 1
            else:
                self._protocol_failures = 0
            if self._protocol_failures >= 2:
                self.protocol = "tool"
                warnings.append("文本协议连续失效，已切换到 tool 协议（两阶段）")

        if parsed.declared:
            warnings.append("位移被声明了（硬失败）：" + " / ".join(parsed.declared[:2]))

        head = parsed.head
        if head and not self.cfg.can_displace(head.displace):
            warnings.append(f"角色 legibility={self.cfg.role.legibility_max} 不允许位移"
                            f"（模型写了 {head.displace}）")

        # 绝不能退回 raw.strip()：那里面是 HEAD，direct 就在里面。
        # 解析不出 SAY 就空着 —— 空回复是可接受的失败，泄露内心不是。
        say = parsed.say or plain_text(raw)

        # 「猜你……」只观测，不改写。
        #
        # 设计文档 §7：假读错（明知故问）是位移的最高明形式，自带免责空间；
        # 硬线是「问完之后留不留退路」。所以「猜」本身不是错，
        # 不给退路的断言才是 —— 而这一层用正则分不干净。
        #
        # 曾经试过在输出侧硬删，结果把「猜一句：失眠跟工作有关」
        # 连同紧跟的「不对就当我没说」一起删了 —— 恰好删掉了退路本身。
        # 用户看到的回复变短变硬，位移被削平。所以退回到只统计。
        if parsed.read_aloud:
            with_exit = sum(1 for h in parsed.read_aloud if has_exit_offer(h))
            detail = " / ".join(h[:28] for h in parsed.read_aloud[:2])
            if with_exit:
                warnings.append(
                    f"「读」说出口了（{with_exit}/{len(parsed.read_aloud)} 条留了退路）：{detail}")
            else:
                warnings.append(f"「读」说出口且没留退路（硬失败）：{detail}")
        self._probe["read_aloud"] = parsed.read_aloud

        if not parsed.is_consistent():
            warnings.append("HEAD 说有位移，但 direct 与 say 无法区分 —— 疑似装深沉")

        # ---- 7. 这一句是不是修复动作。
        #        修复不能在受伤的当轮发 —— 当场道歉往往是在处理自己的尴尬。
        #        所以本轮刚开的修复，本轮不认领。
        repair_opened = self._maybe_repair(say, warnings)

        # ---- 4. 落账
        displaced = parsed.has_displacement
        if head and head.confidence:
            stance = self.cfg.stance_for(head.confidence)

        delta = self._sanitize_delta(parsed.tail.delta if parsed.tail else {})
        relation_changed = (
            self.store.apply_relation_delta(delta, self.cfg.limits) if delta else {}
        )

        pending_added = self._maybe_add_pending(parsed, turn_id)

        self.store.append("turn", {
            "turn_id": turn_id,
            "user": user_text,
            "say": say,
            "need": head.need if head else "",
            "confidence": head.confidence if head else None,
            "direct": head.direct if head else "",
            "displace": head.displace if head else DISPLACE_NONE,
            "stance": stance,
            "relation_delta": relation_changed,
            "settled": [{k: s.get(k) for k in ("id", "outcome")} for s in settled],
            "repair": {"opened": bool(repair_opened), "closed": bool(repair_closed)},
            "warnings": warnings,
            # 观测：没有这些数值，聊三十轮也不知道该把阈值往哪调
            "probe": self._probe,
        })

        if head and head.need:
            self.store.push_buffer({
                "turn_id": turn_id,
                "user": user_text,
                "say": say,
                "direct": head.direct,
                "need": head.need,
                "displace": head.displace,
            })

        # 历史里存带格式的版本，而不是纯 say。
        # 只存 say 会让模型看到「前几轮我都在说大白话」，然后跟着不守协议 ——
        # 这是 13 轮里 9 轮丢格式的直接原因。
        self.history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant",
             "content": build_history_entry(say, head, parsed.tail)},
        ])
        self._last_user = user_text
        self._last_say = say
        self._last_need = head.need if head else ""
        self._topic.append(user_text)

        return TurnResult(
            turn_id=turn_id, say=say, parsed=parsed, stance=stance,
            displaced=displaced, settled=settled, pending_added=pending_added,
            relation_changed=relation_changed, warnings=warnings,
            repair_opened=repair_opened, repair_closed=repair_closed,
        )

    # ------------------------------------------------------------ 生成

    def _generate(self, messages, on_delta):
        """流式优先，失败退回整段。流式只是体验优化，不是正确性依赖。

        回退时有个必须处理的细节：如果流已经吐了一部分，用户已经看到了。
        退回整段再吐一遍，用户会看到重复的半句。所以只在**一片都没吐**时
        才允许回退重来；已经吐过的，用流式那半截的解析结果收尾。
        """
        if on_delta is not None and self.stream_llm is not None:
            from .stream import StreamParser

            sp = StreamParser()
            emitted = False

            def forward(piece: str) -> None:
                nonlocal emitted
                if piece.strip():
                    emitted = True
                on_delta(piece)

            try:
                for chunk in self.stream_llm(messages):
                    for piece in sp.feed(chunk):
                        forward(piece)
                parsed = sp.finish()
                return sp.say_text or "", parsed
            except Exception:
                if emitted:
                    # 已经吐出去了，重来会造成重复。就地收尾。
                    try:
                        parsed = sp.finish()
                    except Exception:
                        parsed = parse("")
                    return sp.say_text or "", parsed

        raw = self.llm(messages)
        return raw, parse(raw)


    # ------------------------------------------------------------ person 协议

    def _generate_persona(self, sys_prompt: str, ctx: str, user_text: str,
                          fallback_messages: list[dict], on_delta,
                          warnings: list[str]):
        """三层协议：素回复（无人设）→ 人设化（带人设）→ 结算。

        和 "tool" 的区别：tool 的 phase 1 仍然看得到人设（只是要求它别用），
        这一条的素回复调用**完全看不到人设**。实测前者会被污染。

        素回复这层只多一次调用，代价约 1.2s；但它是「人格可控」的前提。
        """
        from . import protocol

        if self.tool_llm is None:
            warnings.append("未配置 tool_llm，退回文本协议")
            return self._generate(fallback_messages, on_delta)

        # ---- 1. 素回复：不看人设，只看对话
        msgs_raw = protocol.build_raw_messages(self.history, user_text)
        try:
            data = self.tool_llm(msgs_raw, tool=protocol.raw_tool(self.cfg),
                                 max_tokens=400)
            head = protocol.to_head(data)
            if not head.direct:
                raise RuntimeError("素回复为空")
        except Exception as e:
            warnings.append(f"素回复阶段失败（{e}），退回文本协议")
            self._protocol_failures += 1
            return self._generate(fallback_messages, on_delta)

        self._probe["persona"] = {"raw": head.direct[:120],
                                  "need": head.need, "displace": head.displace}

        # ---- 2. 人设化：把素回复变成这个角色会说的话
        # 末尾重申长度 —— 实测：brevity 的提示在第 25 行，而中间隔着几十行
        # 多句长回复的示例，模型跟着示例走。规则输给示例，所以要放到最后。
        brev = self.cfg.role.traits.get("brevity")
        guard = ""
        if brev is not None:
            if brev.value >= 0.75:
                guard = ("\n\n[这一轮的长度]\n"
                         "一两句话就够。不要超过三句。不要分点，不要展开。"
                         "说完就停。")
            elif brev.value <= 0.40:
                guard = "\n\n[这一轮的长度]\n可以说完整，不限句数。"
        msgs2 = [{"role": "system", "content": sys_prompt + "\n\n" + ctx + guard},
                 *self.history[-12:],
                 {"role": "user", "content": user_text},
                 {"role": "assistant",
                  "content": f"（素回复：{head.direct}／位移方式：{head.displace}）"},
                 {"role": "user", "content": "用你的方式说出口。"}]

        say = ""
        say_raw = ""
        try:
            if on_delta is not None and self.stream_llm is not None:
                sp = protocol.StreamCleanPhase2()
                for piece in self.stream_llm(msgs2):
                    if not piece:
                        continue
                    say_raw += piece
                    for clean in sp.feed(piece):
                        on_delta(clean)
                for clean in sp.finish():
                    on_delta(clean)
                say = sp.text
            else:
                say_raw = self.llm(msgs2)
                say = protocol.clean_phase2(say_raw)
        except Exception as e:
            self._protocol_failures += 1
            # 已经吐给用户的内容不能再吐一遍 —— 否则用户看到两遍，
            # 而账本只记一遍（账本在说谎）。
            # 实测：人设化吐了半句后连接断，退回文本协议 ->
            # 用户看到「你都想过哪些了。你都想过哪些了。」
            if say_raw.strip():
                warnings.append(f"人设化阶段失败（{e}），已输出的部分就地收尾")
                say = protocol.clean_phase2(say_raw)
                if not say:
                    return self._generate(fallback_messages, on_delta)
            else:
                warnings.append(f"人设化阶段失败（{e}），退回文本协议")
                return self._generate(fallback_messages, on_delta)

        if not say:
            self._probe.setdefault("persona", {})["say_raw"] = say_raw[:300]
            self._protocol_failures += 1
            if say_raw.strip():
                warnings.append("人设化产出被清洗为空，已输出的部分就地收尾")
                say = safe_say(say_raw)
                if not say:
                    return self._generate(fallback_messages, on_delta)
            else:
                warnings.append("人设化产出为空，退回文本协议")
                return self._generate(fallback_messages, on_delta)

        # ---- 3. 结算
        tail = self._tool_tail(parsed=protocol.to_parsed(head, say),
                               user_text=user_text,
                               pendings=self.store.pendings(),
                               warnings=warnings)
        parsed = protocol.to_parsed(head, say)
        parsed.tail = tail
        return say, parsed

    # ------------------------------------------------------------ 两阶段

    def _generate_two_phase(self, sys_prompt: str, ctx: str, user_text: str,
                            stance: str, fallback_messages: list[dict],
                            on_delta, warnings: list[str]):
        """phase 1 用 tool call 出 direct，phase 2 用纯文本流式出 say。

        失败一律退回文本协议 —— 这是兜底机制，不能自己变成新的故障点。
        """
        from . import protocol

        if self.tool_llm is None:
            warnings.append("未配置 tool_llm，退回文本协议")
            return self._generate(fallback_messages, on_delta)

        # ---- phase 1：只要内心
        msgs1 = protocol.build_phase1_messages(
            sys_prompt, ctx, self.history, user_text, stance)
        try:
            data = self.tool_llm(msgs1, tool=protocol.inner_tool(self.cfg),
                                 max_tokens=400)
            inner = protocol.parse_inner(data)
            if not inner.direct:
                raise RuntimeError("phase 1 的 direct 为空")
        except Exception as e:
            warnings.append(f"两阶段 phase 1 失败（{e}），退回文本协议")
            self._protocol_failures += 1
            return self._generate(fallback_messages, on_delta)

        self._probe["protocol"] = {"mode": "tool", "direct": inner.direct[:120]}

        # ---- phase 2：把 direct 说成 say
        msgs2 = protocol.build_phase2_messages(inner, self.history, user_text)
        say = ""
        say_raw = ""
        try:
            if on_delta is not None and self.stream_llm is not None:
                # 流式时不能「先吐完再清理」—— 用户已经看到标记了。
                # 也不能对整段做清理再判断：标记出现在中间片段时，
                # 整段切分会把标记两侧的正文也切掉（实测就是这么丢掉
                # 第 3 轮那句完整回复的）。
                # 正确做法：流式逐片过滤，只把干净的增量交给回调。
                sp = protocol.StreamCleanPhase2()
                for piece in self.stream_llm(msgs2):
                    if not piece:
                        continue
                    say_raw += piece
                    for clean in sp.feed(piece):
                        on_delta(clean)
                for clean in sp.finish():
                    on_delta(clean)
                say = sp.text
            else:
                say_raw = self.llm(msgs2)
                say = protocol.clean_phase2(say_raw)
        except Exception as e:
            self._protocol_failures += 1
            # 同上：已经吐过的不能重来。
            if say_raw.strip():
                warnings.append(f"两阶段 phase 2 失败（{e}），已输出的部分就地收尾")
                say = protocol.clean_phase2(say_raw)
                if not say:
                    return self._generate(fallback_messages, on_delta)
            else:
                warnings.append(f"两阶段 phase 2 失败（{e}），退回文本协议")
                return self._generate(fallback_messages, on_delta)

        if not say:
            # 把原始输出留在 probe 里，否则「为什么为空」永远查不出来
            self._probe.setdefault("protocol", {})["phase2_raw"] = say_raw[:400]
            self._protocol_failures += 1
            if say_raw.strip():
                warnings.append("phase 2 产出被清洗为空，已输出的部分就地收尾")
                say = safe_say(say_raw)
                if not say:
                    return self._generate(fallback_messages, on_delta)
            else:
                warnings.append(f"phase 2 产出为空（原始 0 字），退回文本协议")
                return self._generate(fallback_messages, on_delta)

        ok, why = protocol.displacement_ok(inner, say)
        if not ok:
            # phase 2 复述了 direct —— 这一轮已经吐给用户了，不能重来。
            # 记下来，下一轮升级到 tool 之后的重复检查会更严。
            warnings.append(f"两阶段产出不合格：{why}")
            self._protocol_failures += 1

        # ---- phase 3：结算 + 新预言。
        # 两阶段的 phase 1/2 只管「这一轮说什么」，不含 TAIL 语义。
        # 不补这一步，pending 永远挂不上 —— 整个校准账本会停摆。
        # 单独一次调用，不阻塞已发出的 SAY（用户已经看到回复了）。
        tail = self._tool_tail(parsed=protocol.to_parsed(inner, say),
                               user_text=user_text,
                               pendings=self.store.pendings(),
                               warnings=warnings)

        parsed = protocol.to_parsed(inner, say)
        parsed.tail = tail
        return say, parsed

    def _tool_tail(self, *, parsed, user_text: str, pendings: list[dict],
                   warnings: list[str]):
        """tool 协议的第三阶段：结算上一轮的预言 + 挂这一轮的新预言。

        走 tool call，复用 inner 那套「结构化输出」的思路。失败就退回空骨架 ——
        这一轮只是没学到东西，不影响对话。
        """
        from . import protocol

        if self.tool_llm is None:
            return Tail()

        msgs = protocol.build_tail_messages(
            user_text=user_text, say=parsed.say, pendings=pendings,
            history=self.history,
        )
        try:
            data = self.tool_llm(msgs, tool=protocol.tail_tool(pendings),
                                 max_tokens=500)
            return protocol.parse_tail(data)
        except Exception as e:
            warnings.append(f"tool 协议结算阶段失败（{e}），本轮不记结算")
            return Tail()

    # ------------------------------------------------------------ 话题判定

    def _submit_topic_judge(self, user_text: str, pendings: list[dict]):
        """发射一次异步判定。没有判定器就返回 None。"""
        if self.judge is None or not pendings:
            return None
        from .topicjudge import JudgeRequest

        extra = {p["id"]: list(self._pending_topic.get(p["id"], []))
                 for p in pendings}
        req = JudgeRequest(
            user_text=user_text,
            prev_ai_say=self._last_say,
            prev_user_text=self._last_user,
            pendings=pendings,
            extra_refs=extra,
        )
        try:
            return self.judge.submit(req)
        except Exception:
            return None

    def _collect_topic_judge(self, handle, user_text: str,
                             pendings: list[dict], turn_started: float):
        """收判定。超时或失败一律退回字面，绝不让对话变慢。"""
        if handle is None:
            return {}, "none"
        try:
            verdicts, source = handle.collect()
            return verdicts, source
        except Exception:
            return {}, "bigram"

    # ------------------------------------------------------------ 修复

    def _repair_book(self) -> RepairBook:
        return RepairBook.from_list(self.store.load_view("repair", {}).get("items", []))

    def _save_repair_book(self, book: RepairBook) -> None:
        self.store.save_view("repair", {"items": book.to_list()})

    def _repair_note(self) -> str:
        """把待修复的事塞进上下文，提醒模型该做什么。"""
        book = self._repair_book()
        active = book.active()
        if not active:
            return ""
        lines = ["[你刚才可能伤到对方了]"]
        for r in active:
            if r.state == "open":
                lines.append(f"- {r.note}")
                lines.append(f"  还没处理。用 {r.level} 的力度，短，别解释。")
            else:
                lines.append(f"- {r.note}（已经提过一次，别反复道歉）")
        return "\n".join(lines)

    def _judge_repair_response(self, user_text: str,
                               warnings: list[str]) -> dict | None:
        """上一轮发过修复的话，这一轮看对方接不接。"""
        book = self._repair_book()
        repairing = [r for r in book.active() if r.state == "repairing"]
        if not repairing:
            return None

        verdict = judge_repair_response(
            user_text, baseline_formality=self._baseline_formality()
        )
        r = repairing[0]

        if verdict == "healed":
            book.close(r.note, healed=True)
            self._save_repair_book(book)
            self._drop_unresolved(r.note)
            self.store.append("repair_healed", {"note": r.note})
            return {**r.as_dict(), "verdict": "healed"}

        if verdict == "rejected":
            book.close(r.note, healed=False)
            self._save_repair_book(book)
            # 修复失败 → 转成长期雷区，别再提
            self._to_sore_point(r.note)
            self._drop_unresolved(r.note)
            self.store.append("repair_rejected", {"note": r.note})
            warnings.append("修复被拒绝 —— 转成雷区，不要再提")
            return {**r.as_dict(), "verdict": "rejected"}

        # 不清楚：再给一次机会，但只给一次
        if r.attempts >= 2:
            book.close(r.note, healed=False)
            self._save_repair_book(book)
            self._to_sore_point(r.note)
            self._drop_unresolved(r.note)
            self.store.append("repair_residual", {"note": r.note})
            return {**r.as_dict(), "verdict": "residual"}
        return None

    def _maybe_repair(self, say: str, warnings: list[str]) -> dict | None:
        """检测到受伤 → 记一笔，并决定这一句是不是修复。

        修复不在受伤的当轮发 —— 当场道歉往往是在处理自己的尴尬，
        不是在照顾对方。所以 turn_seq 等于本轮的修复（本轮刚开的），
        本轮不认领：它要么是模型提前道歉，要么是误判。
        """
        book = self._repair_book()
        active = book.active()
        if not active:
            return None

        for r in active:
            # 受伤的当轮不认领（turn_seq 相同）—— 那时模型还在处理自己的尴尬。
            if r.turn_seq >= self._turn_seq:
                continue
            # 状态门槛：open = 还没试过，repairing = 试过一次但没回应。
            # 两种都允许再试，但 attempts >= 2 就不再试了 ——
            # 「反复道歉是二次伤害」，第二次还没反应就该转雷区。
            #
            # 原来只允许 open，导致 attempts 永远停在 1，
            # 下面「attempts >= 2 转雷区」那条规则从未生效。
            if r.state not in ("open", "repairing"):
                continue
            if r.attempts >= 2:
                continue
            if is_repair_attempt(say):
                level = repair_level(say)
                r.level = level
                book.mark_repairing(r.note)
                self._save_repair_book(book)
                self.store.append("repair_attempt", {"note": r.note, "level": level})
                return {**r.as_dict(), "action": "attempted"}
        return None

    def _to_sore_point(self, note: str) -> None:
        rel = self.store.relation()
        sore = list(rel.get("sore_points", []))
        if note not in sore:
            sore.append(note)
            self.store.save_view("relation", {**rel, "sore_points": sore})

    def _drop_unresolved(self, note: str) -> None:
        rel = self.store.relation()
        unresolved = [u for u in rel.get("unresolved", []) if u != note]
        self.store.save_view("relation", {**rel, "unresolved": unresolved})

    # ------------------------------------------------------------ 结算

    def _settle_with(self, user_text: str, warnings: list[str],
                     topic_verdicts: dict | None = None,
                     model_settle: dict[str, "SettleItem"] | None = None,
                     model_watch: set[str] | None = None) -> list[dict]:
        """拿用户这一句话，去核对上一轮的预言。

        判定优先级（见 settle.py 模块 docstring）：
            1. 模型自己在 TAIL 里给的判定 —— 主路径
            2. 语域突变 → SUSPECTED_HURT
            3. 话题延续度
            4. 都不确定 → WATCH

        所以模型判定必须作为 semantic **传进** judge()，而不是等启发式
        判完之后再补一刀 —— 那时候 pending 已经关了，补也补不上。
        """
        out: list[dict] = []
        topic_verdicts = topic_verdicts or {}
        model_settle = model_settle or {}
        model_watch = model_watch or set()

        # 先清扫过期，再读剩下的。顺序反了会把过期的那条也送进判定循环，
        # 在 probe 里留下一条并不存在的 WATCH 样本。
        expired = self.store.sweep_expired(self.cfg.watch_ttl_hours)
        for p in expired:
            self.store.append("pending_expire", {"pending_id": p["id"],
                                                 "reason": "timeout"})
            out.append({**p, "outcome": "EXPIRED", "label": OUTCOME_LABEL["EXPIRED"]})

        pendings = self.store.pendings()
        if not pendings:
            return out

        baseline_len = self._baseline_length()
        baseline_formality = self._baseline_formality()
        self._probe.setdefault("settle", [])

        for p in pendings:
            gap = _hours_since(p.get("created", ""))
            extra = list(self._pending_topic.get(p["id"], []))
            tv = topic_verdicts.get(p["id"])
            model_item = model_settle.get(p["id"])
            feat = extract_features(
                user_text,
                pending_claim=p.get("claim", ""),
                predicts=p.get("predicts", ""),
                prev_ai_say=self._last_say,
                prev_user_text=self._last_user,
                extra_refs=extra,
                gap_hours=gap,
                semantic=model_item.outcome if model_item else None,
                similarity=self.sim,
                topic=tv.verdict if tv else None,
                topic_why=tv.why if tv else "",
                minimal_streak=self._minimal_streak.get(p["id"], 0),
            )

            # 模型明确说 watch -> 直接 WATCH，不要退回字面判定。
            # 丢掉这个意图的话，「只是附和没接话题」可能被判成 CONFIRMED，
            # 凭空提高命中率、污染校准账本。
            if p["id"] in model_watch:
                results.append({"pid": p["id"], "verdict": "WATCH",
                                "settle": False,
                                "why": "模型判 watch（再等等）"})
                continue

            verdict = judge(feat, baseline_formality=baseline_formality,
                            baseline_length=baseline_len)

            # 模型判了就以模型为准，证据也用模型写的
            if model_item is not None and verdict.kind not in (SUSPECTED_HURT,):
                delayed = self._is_delayed(p)
                outcome = model_item.outcome.upper()
                if delayed and outcome in ("CONFIRMED", "REFUTED"):
                    outcome = f"{outcome}'"
                evidence = model_item.evidence or "模型在 TAIL 里自己判的"
                closed = self.store.close_pending(p["id"], outcome, evidence,
                                                  delayed=delayed)
                if closed:
                    self.store.append("pending_settle", {
                        "pending_id": p["id"], "outcome": outcome,
                        "evidence": evidence, "by": "model",
                    })
                    out.append({**closed, "label": OUTCOME_LABEL.get(outcome, "校准账本")})
                    self._last_formality_rise = feat.formality - baseline_formality
                    self._apply_outcome(p, outcome)
                    self._probe["settle"].append({
                        "pid": p["id"], "overlap": round(feat.overlap, 3),
                        "formality": round(feat.formality, 3),
                        "formality_base": round(baseline_formality, 3),
                        "length": feat.length, "topic": feat.topic,
                        "verdict": outcome, "settle": True, "by": "model",
                    })
                continue

            if verdict.kind == WATCH and gap > self.cfg.watch_ttl_hours / 3 \
                    and feat.mentions_topic:
                verdict = judge_delayed(feat, expired=False)

            # 把算这个结论用的原始数值留下 —— 调阈值全靠它
            self._probe["settle"].append({
                "pid": p["id"],
                "overlap": round(feat.overlap, 3),
                "formality": round(feat.formality, 3),
                "formality_base": round(baseline_formality, 3),
                "length": feat.length,
                "topic": feat.topic,
                "verdict": verdict.kind,
                "settle": verdict.settle,
                "by": "heuristic",
            })

            if not verdict.settle:
                # 极短回应才累计；换题不累计（换题是明确的信号，不是「没表态」）
                if _is_minimal_reply(feat):
                    self._minimal_streak[p["id"]] = \
                        self._minimal_streak.get(p["id"], 0) + 1
                warnings.append(f"预言 {p['id']} 挂起观察：{verdict.reason}")
                continue

            self._minimal_streak.pop(p["id"], None)
            outcome = self._outcome_for(verdict, p, feat)
            closed = self.store.close_pending(
                p["id"], outcome, verdict.reason, delayed=verdict.delayed
            )
            if closed:
                self.store.append("pending_settle", {
                    "pending_id": p["id"], "outcome": outcome,
                    "evidence": verdict.reason,
                    "topic_source": tv.verdict if tv else None,
                })
                out.append({**closed, "label": OUTCOME_LABEL.get(outcome, "校准账本")})
                self._last_formality_rise = feat.formality - baseline_formality
                self._apply_outcome(p, outcome)

        return out

    def _is_delayed(self, p: dict) -> bool:
        """挂起超过两轮才算延迟结算。turn_seq 重启后会归零，所以取 max 兜底。"""
        try:
            seq = int(p.get("turn_seq", self._turn_seq))
        except (TypeError, ValueError):
            seq = self._turn_seq
        return self._turn_seq - seq >= 2

    @staticmethod
    def _outcome_for(verdict, p: dict, feat: TurnFeatures) -> str:
        if verdict.kind == SUSPECTED_HURT:
            return SUSPECTED_HURT
        if verdict.kind in (CONFIRMED, REFUTED):
            return verdict.kind
        if verdict.kind in (CONFIRMED_DELAYED, REFUTED_DELAYED):
            return verdict.kind
        return CONFIRMED

    def _apply_outcome(self, p: dict, outcome: str) -> None:
        """结算之后的两件事：调 bias（校准）、记关系（受伤）。"""
        if outcome == SUSPECTED_HURT:
            # 同一轮只开一条修复 —— 用户被伤到一次，不是每条预言伤一次。
            if self._repair_opened_turn != self._turn_seq:
                self._repair_opened_turn = self._turn_seq
                self._open_repair(p)
            return

        try:
            conf = float(p.get("confidence", 0.0))
        except (TypeError, ValueError):
            return
        before = self.cfg.role.bias
        if outcome in (CONFIRMED, CONFIRMED_DELAYED) and conf < 0.4:
            self.cfg.role.bias = min(0.4, self.cfg.role.bias + 0.005)
        elif outcome in (REFUTED, REFUTED_DELAYED) and conf > 0.7:
            self.cfg.role.bias = max(-0.4, self.cfg.role.bias - 0.005)
        after = self.cfg.role.bias
        if after != before:
            # bias 是唯一会跨轮累积的学习量。不落账的话，进程一退就归零，
            # 「同一份账本换策略重放」也就不成立 —— 重放出来的起点是错的。
            self.store.append("bias_update", {
                "before": round(before, 4), "after": round(after, 4),
                "delta": round(after - before, 4),
                "pending_id": p.get("id"), "outcome": outcome,
            })

    def _open_repair(self, p: dict) -> None:
        """检测到受伤：记进未解决 + 开一条修复。

        关键：**受伤是「用户」的状态，不是「预言」的状态。**
        一轮里可能有好几条预言同时判成 SUSPECTED_HURT（说话的人只有一个，
        被伤到的也只有一个人）。所以要按「哪一轮」去重，不能按「哪条预言」。

        实测踩过：按 pending id 做 note 唯一性，结果一轮里 3 条预言同时
        受伤就开了 3 条修复，关系账本里堆了 3 条未解决；用户只被伤了一次。
        反过来用固定 note 又会让第二次受伤被静默吞掉。
        正解是 note 带**轮次**，不带 pending id。
        """
        note = f"可能被第 {self._turn_seq} 轮那句话伤到，还没修复"
        if note not in self.store.relation().get("unresolved", []):
            self.store.apply_relation_delta({"unresolved": [note]}, self.cfg.limits)

        conf = float(p.get("confidence", 0.0) or 0.0)
        # 受伤判定本身已经说明线索很强；这里的 rise 用于在 soft/plain/direct 之间选。
        # 把真实升幅和「已确认受伤」这个事实一起考虑，避免永远落到 soft。
        rise = max(self._last_formality_rise, 0.35)
        level = pick_level(confidence=max(conf, 0.6), formality_rise=rise)

        book = self._repair_book()
        book.open(note, self._turn_seq, now_iso(), level)
        self._save_repair_book(book)
        self.store.append("repair_open", {"note": note, "level": level,
                                          "turn_seq": self._turn_seq,
                                          "confidence": round(conf, 3),
                                          "formality_rise": round(rise, 3)})

    # ------------------------------------------------------------ 辅助

    def _maybe_add_pending(self, parsed: ParsedTurn, turn_id: str) -> dict | None:
        tail = parsed.tail
        if not tail or not tail.pending:
            return None
        added = self.store.add_pending(
            tail.pending.claim, tail.pending.confidence,
            tail.pending.predicts, turn_id, self.cfg.limits,
            turn_seq=self._turn_seq,
        )
        if added:
            self.store.append("pending_add", {
                "pending_id": added["id"], "claim": added["claim"],
                "confidence": added["confidence"],
                "predicts": added["predicts"], "turn_id": turn_id,
                "turn_seq": self._turn_seq,
            })
            self._pending_topic[added["id"]] = list(self._topic[-4:])
        return added

    def _prior_confidence(self) -> float:
        for rec in reversed(self.store.read_ledger()[-40:]):
            if rec.get("kind") == "turn" and rec.get("confidence") is not None:
                return float(rec["confidence"])
        return 0.5

    def _sanitize_delta(self, raw: dict) -> dict:
        out = {}
        for k, v in (raw or {}).items():
            if k not in ONLINE_WRITABLE:
                continue
            if k == "closeness":
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    continue
            else:
                out[k] = v
        return out

    def _baseline_formality(self) -> float:
        return float(self.store.load_view("style", {"formality": 0.3})
                     .get("formality", 0.3))

    def _baseline_length(self) -> float | None:
        st = self.store.load_view("style", {})
        return st.get("length") if st.get("length") else None

    def _style_note(self) -> str:
        if self._baseline_formality() > 0.6:
            return "这个人和你说话一直比较客气克制，别当他在生气。"
        return ""

    def update_style_baseline(self, user_text: str) -> None:
        """慢速跟踪用户的语域基线。这是「尺子」，要稳。

        每次更新都落账。不落账的话，state/style.json 一删，尺子就退回默认值 ——
        而「受伤」是靠语域升幅判的，尺子错了，受伤检测整体失效。
        """
        alpha = 0.15
        cur = self._baseline_formality()
        blended = cur * (1 - alpha) + formality_score(user_text) * alpha
        view = self.store.load_view("style", {})
        view["formality"] = round(blended, 4)
        n = len([c for c in user_text if not c.isspace()])
        lens = (view.get("length_hist", []) + [n])[-20:]
        view["length_hist"] = lens
        view["length"] = round(sum(lens) / len(lens), 1) if lens else None
        self.store.save_view("style", view)
        self.store.append("style_update", {
            "formality": view["formality"],
            "length": view["length"],
            "n": n,
        })


def _judge_elapsed(handle) -> float:
    """判定器自己花了多久。没有判定器就是 0。"""
    if handle is None:
        return 0.0
    started = getattr(handle, "started", None)
    finished = getattr(handle, "finished_at", None)
    if started is None:
        return 0.0
    return max(0.0, (finished or time.monotonic()) - started)


def _hours_since(iso: str) -> float:
    try:
        return max(0.0, (parse_iso(now_iso()) - parse_iso(iso)).total_seconds() / 3600)
    except Exception:
        return 0.0


def stream_turn(engine: TurnEngine, user_text: str) -> Iterator[str]:
    """真正的流式：边生成边吐 SAY。"""
    box: dict[str, str] = {}

    def collect(piece: str) -> None:
        box["text"] = box.get("text", "") + piece

    engine.turn(user_text, on_delta=collect)
    for ch in box.get("text", ""):
        yield ch
