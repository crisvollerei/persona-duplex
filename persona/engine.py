"""单轮编排：组装 → 生成 → 解析 → 落账。

默认一次生成 HEAD（含 direct）、SAY、TAIL（含 pending），记忆在后台异步固化。
观测在线写，结算下一轮填；多调用协议须显式选择。

三条生成路径（由 protocol 决定，见 protocol.py）：

    text     默认：一次生成 HEAD → SAY → TAIL，按输出结构分工
    tool     _generate_two_phase：phase1 出 HEAD、phase2 出 SAY、phase3 结算
    persona  显式实验：独立 raw → 人格表达 → 交付 → 观察

三条路径的共同底线：**direct 永不进用户可见输出**。
统一在 parse.safe_say 兜底，任何一条路径都不例外。

另一条贯穿全部路径的约定：**已经吐给用户的内容不能重来**。
默认不重试；显式选择的兜底路径必须先看「是否已经输出过」，否则用户会看到两遍，
而账本只记一遍 —— 账本在说谎，比不给回复更糟。
"""
from __future__ import annotations

import inspect
import time
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from .parse import SettleItem

from .config import DISPLACE_NONE, PersonaConfig
from .parse import (Head, ParsedTurn, Tail, has_exit_offer, parse, plain_text,
                    safe_say)
from .prompt import (
    build_context_block,
    build_history_entry,
    build_system_prompt,
    build_user_turn,
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
from .audit import manifest, model_info, traced


def _atomic(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.store.transaction():
            return method(self, *args, **kwargs)
    return call

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
    validity: dict = field(default_factory=dict)

    @property
    def display(self) -> str:
        return self.say


class TurnEngine:
    """默认一轮一次生成，不自动增加审查、重试或协议升级。"""

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
        fidelity_check: bool = False,
        json_fallback: bool = False,
        audit_options: dict | None = None,
    ):
        """text 默认单次生成；persona / tool / auto 为显式实验协议。"""
        if protocol not in ("persona", "text", "tool", "auto"):
            raise ValueError(f"未知协议: {protocol}")
        if protocol != "persona" and (fidelity_check or json_fallback):
            raise ValueError("fidelity_check / json_fallback 仅适用于显式选择的 persona 协议")
        self.cfg = cfg
        cfg.validate()
        self.store = store
        self.llm = llm
        self.stream_llm = stream_llm
        self.protocol = protocol
        self.tool_llm = tool_llm
        self.fidelity_check = fidelity_check
        self.json_fallback = json_fallback
        self._protocol_failures = 0
        self.sim = similarity
        self.judge = judge
        self.history: list[dict] = []
        # 只收录独立判断层的结果，不从单次生成的 HEAD 推断中性历史。
        self._neutral_history: list[dict] = []
        self._last_user = ""
        self._last_say = ""
        self._topic: list[str] = []
        self._pending_topic: dict[str, list[str]] = {}
        self._turn_seq = 0
        self._last_formality_rise = 0.0
        # 上一轮的 need，用来触发下一轮的分场景句式注入
        self._last_need = ""
        # 极短回应次数仅供观测，不参与预测真假判定。
        self._minimal_streak: dict[str, int] = {}
        # 上一轮开出修复的轮次号。同一轮只开一条 —— 用户被伤到一次，
        # 不是每条预言伤一次。
        self._repair_opened_turn = -1
        self._restore_turn_seq()
        self._restore_runtime()
        self._turn_lock = threading.Lock()
        self._calls: list[dict] = []
        self._current_turn_id = ""
        self.run = manifest(cfg, protocol=protocol, llm=llm, stream_llm=stream_llm,
                            tool_llm=tool_llm, options={
                                "fidelity_check": fidelity_check, "json_fallback": json_fallback,
                                "judge": model_info(judge), "similarity": model_info(similarity),
                                **(audit_options or {})})
        self.store.append("run_start", self.run)
        def emit(kind, row):
            self.store.append(kind, {"run_id": self.run["run_id"],
                                     "turn_id": self._current_turn_id, **row})
        self.llm = traced(llm, "text", lambda: self._calls, emit)
        self.stream_llm = traced(stream_llm, "stream", lambda: self._calls, emit)
        self.tool_llm = traced(tool_llm, "tool", lambda: self._calls, emit)

    def _runtime(self) -> dict:
        names = ("history", "_neutral_history", "_last_user", "_last_say", "_topic",
                 "_pending_topic", "_last_need", "_minimal_streak", "_turn_seq",
                 "_repair_opened_turn", "_protocol_failures")
        return {"persona": self.cfg.role.name, "protocol": self.protocol,
                **{name: deepcopy(getattr(self, name)) for name in names}}

    def _restore_runtime(self) -> None:
        state = self.store.load_view("runtime", {})
        if state.get("persona") != self.cfg.role.name:
            return
        for name in ("history", "_neutral_history", "_last_user", "_last_say", "_topic",
                     "_pending_topic", "_last_need", "_minimal_streak", "_repair_opened_turn",
                     "_protocol_failures"):
            if name in state:
                setattr(self, name, deepcopy(state[name]))
        if self.protocol == "auto" and state.get("protocol") == "tool":
            self.protocol = "tool"

    def _sync_traits(self) -> None:
        """读取已明确提交的实验参数变更；默认记忆固化不推动人格参数。"""
        state = self.store.traits()
        if state:
            self.cfg.role.load_traits_state(state)

    def _restore_turn_seq(self) -> None:
        """恢复轮次，不把历史预测校准当成人格成长。

        旧 bias_update 留作历史分析；当前 bias 使用明确的人格配置。
        """
        # turn_seq 只在内存里，重启归零会让「延迟结算」永远判不成立
        ledger = self.store.read_ledger()
        self._turn_seq = max([int(r.get("turn_seq", 0)) for r in ledger
                              if r.get("kind") in ("turn_start", "turn")]
                             + [sum(r.get("kind") == "turn" for r in ledger)])

    # ------------------------------------------------------------ 主流程

    def turn(self, user_text: str,
             on_delta: Callable[[str], None] | None = None) -> TurnResult:
        with self._turn_lock:
            self._current_turn_id = self.store.next_id("t")
            self._calls = []
            self._failure_recorded = False
            self._delivery_failed = False
            self._delivered = []
            deliver = None
            if on_delta is not None:
                def deliver(piece):
                    try:
                        on_delta(piece)
                    except Exception:
                        self._delivery_failed = True
                        raise
                    self._delivered.append(piece)
            before = self._runtime()
            try:
                return self._turn(user_text, deliver)
            except Exception as exc:
                self.store.recover_commits()
                for name, value in before.items():
                    if name not in ("persona", "protocol"):
                        setattr(self, name, value)
                self._restore_turn_seq()
                self._restore_runtime()
                if not self._failure_recorded:
                    committed = any(r.get("kind") == "turn" and
                                    r.get("turn_id") == self._current_turn_id
                                    for r in self.store.read_ledger())
                    self._failure({"stage": "delivery" if self._delivery_failed else
                                   "publication" if committed else "turn",
                                   "turn_committed": committed,
                                   "delivered": "".join(self._delivered),
                                   "error_type": type(exc).__name__, "user": user_text})
                raise

    def _failure(self, payload):
        self._failure_recorded = True
        self.store.append("generation_failure", {
            "run_id": self.run["run_id"], "turn_id": self._current_turn_id,
            "protocol": self.protocol, "calls": deepcopy(self._calls), **payload})

    def _turn(self, user_text: str, on_delta=None) -> TurnResult:
        """跑一轮。

        传入 on_delta 时逐片回调；persona 实验开启 fidelity_check 才先缓冲检查。
        """
        turn_id = self._current_turn_id
        warnings: list[str] = []
        self._turn_seq += 1
        turn_started = time.monotonic()
        self._probe = {}       # 本轮原始观测值，落账用
        self.store.append("turn_start", {"turn_id": turn_id, "turn_seq": self._turn_seq,
                                        "run_id": self.run["run_id"], "user": user_text})

        # ---- 1. 修复反应判定。这个必须同步：它决定上下文里要不要放
        #        暂定修复提示。只按用户的明确内容收尾，不调模型。
        repair_closed = self._judge_repair_response(user_text, warnings)

        # ---- 2. 把话题判定发射出去。注意这里只是发射，不等 ——
        #        它会在主模型生成的同时并行跑，生成完再收。
        pendings_snapshot = self.store.pendings()
        judge_handle = self._submit_topic_judge(user_text, pendings_snapshot)

        # ---- 3. 组装上下文。此时预言都还挂着 —— 模型需要在上下文里
        #        看见它们，才能在 TAIL 里给出自己的判定。
        #
        # 显式实验可能更新账本参数；读取它们不代表本轮自动学习人格。
        self._sync_traits()

        relation = self.store.relation()
        pendings = pendings_snapshot
        judgments = deepcopy(self.memory_snapshot) if getattr(self, "memory_snapshot", None) is not None \
            else self.store.user_judgments()

        # 先算出这一轮大概是什么 need —— 用它触发分场景句式注入。
        # 这里读的是**上一轮**的 need，因为这一轮还没生成。
        # 差一轮不影响：句式是语感参考，不是精确控制。
        need_hint = self._last_need
        first_turn = self._turn_seq <= 1
        sys_prompt = build_system_prompt(self.cfg, need=need_hint,
                                         first_turn=first_turn, user_text=user_text)
        ctx = build_context_block(
            self.cfg,
            relation=relation,
            pendings=pendings,
            judgments=judgments,
            style_note=self._style_note(),
            repair_note=self._repair_note(),
        )

        stance = self.cfg.stance_for(0.0)

        messages = [
            {"role": "system", "content": sys_prompt + "\n\n" + ctx},
            *self.history[-12:],
            {"role": "user", "content": build_user_turn(user_text)},
        ]

        # ---- 4. text 默认一次生成三段；其他协议仅在显式选择时执行。
        generation_protocol = self.protocol
        if self.protocol == "persona":
            raw, parsed = self._generate_persona(
                ctx, user_text, on_delta, warnings
            )
        elif self.protocol == "tool":
            raw, parsed = self._generate_two_phase(
                sys_prompt, ctx, user_text, messages, on_delta, warnings
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
            "grace": self.cfg.judge_grace,
            # judge_elapsed：判定器真正花了多久（含在生成期间并行跑掉的那段）
            # elapsed：整轮耗时。两者不能混用 —— 拿整轮耗时比 grace，
            # 会把「生成慢」误报成「判定超时」。
            "judge_elapsed": round(_judge_elapsed(judge_handle), 3),
            "collect_wait": round(time.monotonic() - collect_started, 3),
            "elapsed": round(time.monotonic() - turn_started, 3),
            "verdicts": {k: v.verdict for k, v in topic_verdicts.items()},
            "details": {k: {"verdict": v.verdict, "why": getattr(v, "why", ""),
                             "status": getattr(v, "status", "unknown")}
                        for k, v in topic_verdicts.items()},
            "request": asdict(judge_handle.req) if getattr(judge_handle, "req", None) else None,
            "error_type": type(judge_handle._error).__name__ if getattr(judge_handle, "_error", None) else None,
        }
        if topic_source == "late":
            warnings.append("话题判定未在 grace 内返回，本轮用字面兜底")

        incomplete = self._probe.get("stream_interrupted") or self._probe.get("output_blocked")
        metadata_usable = not parsed.structural_errors and not incomplete
        usable_tail = parsed.tail if parsed.tail_usable and metadata_usable else None

        with self.store.transaction():
            # ---- 6. 结算。模型自己的 TAIL 判定是主路径，必须和启发式一起送进去，
            #        否则启发式先关掉 pending，模型判定就无处生效。
            model_settle = {it.pending_id: it for it in (usable_tail.settle if usable_tail else [])}
            model_watch = set(usable_tail.watch) if usable_tail else set()
            settled = self._settle_with(user_text, warnings, topic_verdicts,
                                        model_settle=model_settle,
                                        model_watch=model_watch)

            warnings.extend(parsed.errors)
            warnings.extend(e for e in parsed.structural_errors if e not in warnings)
            if parsed.tail is not None and usable_tail is None:
                warnings.append("输出结构无效或交付不完整，不采纳本轮 TAIL 的状态变更")

            # auto 模式：协议失效就升级到 tool，之后不再降级回来。
            #
            # 两条修过的判据：
            #  · 失效包括缺段与字段无效（不只 head is None）。
            #    README 明说「缺 TAIL」算失效，而 tool 协议本来就是因为
            #    「缺 TAIL」才需要兜底的 —— 旧判据永远升不了级。
            #  · 成功一轮要清零，否则 fail->success->fail 也会升级，
            #    那不是「连续」失效。
            if self.protocol == "auto":
                if parsed.errors or parsed.structural_errors or (parsed.head and parsed.head.errors):
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
                # 摘要可能截掉后半句退路；只对完整 SAY 作措辞观测，不判定心理或操控。
                with_exit = has_exit_offer(say)
                detail = " / ".join(h[:28] for h in parsed.read_aloud[:2])
                if with_exit:
                    warnings.append(f"检测到读法措辞，完整回复中有退路提示（仅观测）：{detail}")
                else:
                    warnings.append(f"检测到读法措辞，未识别到显式退路；需结合语境核对：{detail}")
            self._probe["read_aloud"] = parsed.read_aloud

            if not parsed.is_consistent():
                warnings.append("HEAD 声称额外位移，但 baseline 与 say 字面相似；需核对语义")

            # ---- 7. 这一句是不是修复动作。
            #        修复不能在受伤的当轮发 —— 当场道歉往往是在处理自己的尴尬。
            #        所以本轮刚开的修复，本轮不认领。
            repair_opened = self._maybe_repair(say, warnings)

            # ---- 4. 落账
            displaced = parsed.has_displacement
            if head is not None:
                stance = self.cfg.stance_for(head.confidence)

            delta = self._sanitize_delta(usable_tail.delta if usable_tail else {})
            relation_changed = (
                self.store.apply_relation_delta(delta, self.cfg.limits) if delta else {}
            )

            pending_added = self._maybe_add_pending(parsed, turn_id) if usable_tail else None

            validity = parsed.validity(protocol=generation_protocol, probe=self._probe,
                                       allowed=self.cfg.can_displace(head.displace) if head else True)
            self.store.append("turn", {
                "turn_id": turn_id,
                "turn_seq": self._turn_seq,
                "run_id": self.run["run_id"],
                "input": {"relation": relation, "pendings": pendings,
                          "judgments": judgments, "history": deepcopy(self.history[-12:]),
                          "need_hint": need_hint, "context": ctx,
                          "config": asdict(self.cfg)},
                "calls": deepcopy(self._calls),
                "validity": validity,
                "limits": self.cfg.limits.as_dict(),
                "user": user_text,
                "say": say,
                "need": head.need if head else "",
                "confidence": head.confidence if head else None,
                "direct": head.direct if head else "",
                "baseline": head.baseline if head else "",
                "displace": head.displace if head else DISPLACE_NONE,
                "displacement_basis": "persona_baseline" if head and head.baseline else "unknown",
                "design_version": 3 if generation_protocol == "persona" else 2,
                "persona": self.cfg.role.name,
                "parameters": {
                    "bias": self.cfg.role.bias,
                    "attention": self.cfg.role.attention,
                    "legibility_max": self.cfg.role.legibility_max,
                    "traits": self.cfg.role.traits_state(),
                },
                "protocol": generation_protocol,
                "judgment_constraints": head.constraints if head else [],
                "stance": stance,
                "relation_delta": relation_changed,
                "settled": [{k: s.get(k) for k in ("id", "outcome")} for s in settled],
                "repair": {"opened": bool(repair_opened), "closed": bool(repair_closed)},
                "warnings": warnings,
                # 观测：没有这些数值，聊三十轮也不知道该把阈值往哪调
                "probe": self._probe,
            })

            if head and head.need and not incomplete and validity["valid"]:
                self.store.push_buffer({
                    "turn_id": turn_id,
                    "user": user_text,
                    "say": say,
                    "direct": head.direct,
                    "baseline": head.baseline,
                    "persona": self.cfg.role.name,
                    "need": head.need,
                    "displace": head.displace,
                })

            # 完整有效的元数据可作格式示范；坏结构只保留可见对话，不洗成正常样例。
            history_content = (build_history_entry(say, head, parsed.tail)
                               if metadata_usable and validity["valid"] else say)
            self.history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": history_content},
            ])
            self._last_user = user_text
            self._last_say = say
            self._last_need = head.need if head and metadata_usable and not head.errors else ""
            self._topic.append(user_text)
            self.history = self.history[-12:]
            self._topic = self._topic[-20:]
            self.update_style_baseline(user_text)
            self.store.save_view("runtime", self._runtime())

            return TurnResult(
                turn_id=turn_id, say=say, parsed=parsed, stance=stance,
                displaced=displaced, settled=settled, pending_added=pending_added,
                relation_changed=relation_changed, warnings=warnings,
                repair_opened=repair_opened, repair_closed=repair_closed,
                validity=validity,
            )

    # ------------------------------------------------------------ 生成

    def _generate(self, messages, on_delta):
        """一次请求，不因流式失败另发整段请求；已交付片段就地收尾。"""
        if on_delta is not None and self.stream_llm is not None:
            from .stream import StreamParser

            sp = StreamParser()
            delivered: list[str] = []
            callback_failed = False

            def forward(piece: str) -> None:
                nonlocal callback_failed
                try:
                    on_delta(piece)
                except Exception:
                    callback_failed = True
                    raise
                delivered.append(piece)

            try:
                for chunk in self.stream_llm(messages):
                    for piece in sp.feed(chunk):
                        forward(piece)
                parsed = sp.finish()
                if sp.output_blocked:
                    self._probe["output_blocked"] = True
                visible = "".join(delivered)
                if parsed.say.startswith(visible):
                    remainder = parsed.say[len(visible):]
                    if remainder:
                        forward(remainder)
                return parsed.say, parsed
            except Exception as exc:
                if callback_failed:
                    raise
                self._probe["stream_interrupted"] = True
                visible = "".join(delivered)
                self._failure({
                    "protocol": self.protocol, "stage": "流式生成",
                    "error_type": type(exc).__name__, "delivered": visible,
                })
                if visible.strip():
                    # 不补发扣留内容，也不提交尚未完成的 TAIL。
                    parsed = sp.finish(interrupted=True)
                    parsed.say = visible
                    parsed.errors.append("生成流式中断，仅记录已输出片段，不自动重试")
                    return visible, parsed
                raise

        raw = self.llm(messages)
        return raw, parse(raw)


    # ------------------------------------------------------------ person 协议

    def _persona_structured(self, messages, tool, *, max_tokens, warnings):
        """未配置工具时直接用 JSON；工具失败默认报错，格式回退须显式开启。"""
        from . import protocol

        if self.tool_llm is not None:
            try:
                return self.tool_llm(messages, tool=tool, max_tokens=max_tokens)
            except Exception:
                if not self.json_fallback:
                    raise
                warnings.append(f"{tool['function']['name']} 工具调用失败，改用同阶段 JSON")
        text_messages = protocol.structured_messages(messages, tool)
        try:
            inspect.signature(self.llm).bind(text_messages, max_tokens=max_tokens)
        except (TypeError, ValueError):
            raw = self.llm(text_messages)
        else:
            raw = self.llm(text_messages, max_tokens=max_tokens)
        return protocol.parse_object(raw)

    def _generate_persona(self, ctx: str, user_text: str, on_delta, warnings: list[str]):
        """独立判断 → 人格表达 → 交付 → 观察；默认不增加补救调用。

        显式开启 fidelity_check 才缓冲并执行无人设检查，不把模型检查当真值。
        失败不回到混合协议，也不把内部基础回答直接作为可见回复。
        """
        from . import protocol
        from .stream import StreamParser

        stage = "基础判断"
        probe = self._probe["persona"] = {
            "isolated": True, "fidelity_enabled": self.fidelity_check,
            "json_fallback_enabled": self.json_fallback,
        }
        head = None
        say = ""
        emitted: list[str] = []
        interrupted = False
        try:
            msgs_raw = protocol.build_raw_messages(
                self._neutral_history, user_text, neutral_history=True, dialogue=self.history)
            data = self._persona_structured(
                msgs_raw, protocol.raw_tool(), max_tokens=1400, warnings=warnings)
            head = protocol.parse_raw(data)
            probe.update(raw=head.direct[:120], need=head.need)

            stage = "人格表达"
            msgs2 = protocol.build_persona_messages(
                self.cfg, head, self.history, user_text, ctx,
                first_turn=not self.history)
            if on_delta is not None and self.stream_llm is not None:
                sp = StreamParser(expression_only=True)
                try:
                    for piece in self.stream_llm(msgs2):
                        for clean in sp.feed(piece):
                            if not self.fidelity_check:
                                on_delta(clean)
                                emitted.append(clean)
                    persona_turn = sp.finish()
                    if sp.output_blocked:
                        interrupted = True
                        self._probe["output_blocked"] = True
                        probe["output_blocked"] = True
                        warnings.extend(persona_turn.errors)
                except Exception:
                    if self._delivery_failed:
                        raise
                    if self.fidelity_check or not any(p.strip() for p in emitted):
                        raise
                    # 只保留实际交付的片段，不补写、不重试，也不据残句生成新观察。
                    interrupted = True
                    self._probe["stream_interrupted"] = True
                    probe["stream_interrupted"] = True
                    warnings.append("人格表达流式中断，仅记录已输出片段，不自动重试")
                    persona_turn = protocol.to_parsed(Head(), "".join(emitted))
                if not self.fidelity_check and not interrupted:
                    visible = "".join(emitted)
                    if persona_turn.say.startswith(visible):
                        remainder = persona_turn.say[len(visible):]
                        if remainder:
                            on_delta(remainder)
                            emitted.append(remainder)
            else:
                persona_turn = parse(self.llm(msgs2), expression_only=True)
            say = safe_say(persona_turn.say)
            if not say:
                raise ValueError("人格表达为空或包含协议残留")
            if persona_turn.head:
                head.baseline = persona_turn.head.baseline
                head.displace = persona_turn.head.displace
                head.errors.extend(e for e in persona_turn.head.errors
                                   if e not in ("missing_need", "missing_confidence", "missing_direct"))
            probe["displace"] = head.displace
            if persona_turn.structural_errors:
                warnings.append("人格表达结构无效，不生成后续观察")

            if self.fidelity_check:
                if persona_turn.structural_errors:
                    raise ValueError("人格表达结构无效")
                stage = "一致性检查"
                check = self._persona_structured(
                    protocol.build_fidelity_messages(head, user_text, say, self._neutral_history,
                                                     dialogue=self.history),
                    protocol.fidelity_tool(), max_tokens=500, warnings=warnings)
                if not isinstance(check, dict) or check.get("verdict") not in (
                        "consistent", "inconsistent", "unknown") \
                        or not isinstance(check.get("reason"), str) or not check["reason"].strip():
                    raise ValueError("一致性检查结果缺失或格式无效")
                probe["fidelity"] = {"verdict": check["verdict"], "reason": check["reason"][:500]}
                if check["verdict"] != "consistent":
                    raise ValueError("人格表达未通过一致性检查")
        except Exception as exc:
            if self._delivery_failed:
                raise
            self._failure({
                "protocol": "persona", "stage": stage, "probe": probe,
                "error_type": type(exc).__name__,
                "user": user_text, "judgment": head.direct if head else "",
                "judgment_constraints": head.constraints if head else [], "candidate": say,
                "delivered": "".join(emitted),
            })
            delivery = "本轮已有部分输出，不自动重试" if emitted else "本轮未输出回复"
            raise RuntimeError(f"分层生成的{stage}失败，{delivery}；可查看账本") from exc

        # 开启检查或未走流式时统一交付；正常流式路径不能重复发送。
        if on_delta is not None and not emitted:
            on_delta(say)
        parsed = protocol.to_parsed(head, say)
        parsed.structural_errors.extend(persona_turn.structural_errors)
        parsed.tail = None
        if not interrupted and not parsed.structural_errors:
            parsed.tail = self._tool_tail(parsed=parsed, user_text=user_text,
                                          pendings=self.store.pendings(), warnings=warnings,
                                          isolated=True)
        self._neutral_history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": head.direct},
        ])
        self._neutral_history = self._neutral_history[-12:]
        return say, parsed

    # ------------------------------------------------------------ 两阶段

    def _generate_two_phase(self, sys_prompt: str, ctx: str, user_text: str,
                            fallback_messages: list[dict],
                            on_delta, warnings: list[str]):
        """phase 1 用 tool call 出 direct，phase 2 用纯文本流式出 say。

        失败一律退回文本协议 —— 这是兜底机制，不能自己变成新的故障点。
        """
        from . import protocol

        if self.tool_llm is None:
            warnings.append("未配置 tool_llm，退回文本协议")
            return self._generate(fallback_messages, on_delta)

        # ---- phase 1：内部参考（素回答、人格基准、额外位移）
        msgs1 = protocol.build_phase1_messages(
            sys_prompt, ctx, self.history, user_text)
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
        msgs2 = protocol.build_phase2_messages(
            inner, self.history, user_text, system_prompt=sys_prompt, context=ctx)
        say = ""
        say_raw = ""
        emitted: list[str] = []
        interrupted = False

        def forward(piece):
            if piece:
                on_delta(piece)
                emitted.append(piece)

        try:
            if on_delta is not None and self.stream_llm is not None:
                # 流式时不能「先吐完再清理」—— 用户已经看到标记了。
                # 也不能对整段做清理再判断：标记出现在中间片段时，
                # 整段切分会把标记两侧的正文也切掉（实测就是这么丢掉
                # 第 3 轮那句完整回复的）。
                # 正确做法：流式逐片过滤，只把干净的增量交给回调。
                from .stream import StreamParser

                sp = StreamParser()
                for piece in self.stream_llm(msgs2):
                    if not piece:
                        continue
                    say_raw += piece
                    for clean in sp.feed(piece):
                        forward(clean)
                phase2 = sp.finish()
                visible = "".join(emitted)
                if phase2.say.startswith(visible):
                    remainder = phase2.say[len(visible):]
                    if remainder:
                        forward(remainder)
                say = phase2.say
                if sp.output_blocked:
                    interrupted = True
                    self._probe["output_blocked"] = True
                    warnings.append("流式输出遇到协议残留，已截断可疑内容，不生成后续观察")
            else:
                say_raw = self.llm(msgs2)
                say = protocol.clean_phase2(say_raw)
        except Exception as e:
            if self._delivery_failed:
                raise
            self._protocol_failures += 1
            # 同上：已经吐过的不能重来。
            if any(p.strip() for p in emitted):
                interrupted = True
                self._probe["stream_interrupted"] = True
                self._failure({
                    "protocol": "tool", "stage": "人格表达", "error_type": type(e).__name__,
                    "delivered": "".join(emitted),
                })
                warnings.append(f"两阶段 phase 2 失败（{e}），已输出的部分就地收尾")
                say = safe_say("".join(emitted))
            else:
                warnings.append(f"两阶段 phase 2 失败（{e}），退回文本协议")
                return self._generate(fallback_messages, on_delta)

        if not say:
            # 把原始输出留在 probe 里，否则「为什么为空」永远查不出来
            self._probe.setdefault("protocol", {})["phase2_raw"] = say_raw[:400]
            self._protocol_failures += 1
            if any(p.strip() for p in emitted):
                warnings.append("phase 2 产出被清洗为空，已输出的部分就地收尾")
                say = safe_say("".join(emitted))
            else:
                warnings.append("phase 2 无可见产出，退回文本协议")
                return self._generate(fallback_messages, on_delta)

        ok, why = protocol.displacement_ok(inner, say)
        if not ok:
            # phase 2 复述了人格基准却声称额外位移 —— 这一轮已经吐给用户了，不能重来。
            # 记下来，下一轮升级到 tool 之后的重复检查会更严。
            warnings.append(f"两阶段产出不合格：{why}")
            self._protocol_failures += 1

        # ---- phase 3：结算 + 新预言。
        # 两阶段的 phase 1/2 只管「这一轮说什么」，不含 TAIL 语义。
        # 不补这一步，pending 永远挂不上 —— 整个校准账本会停摆。
        # 单独一次调用，不阻塞已发出的 SAY（用户已经看到回复了）。
        parsed = protocol.to_parsed(inner, say)
        parsed.tail = None
        if not interrupted:
            parsed.tail = self._tool_tail(parsed=parsed, user_text=user_text,
                                          pendings=self.store.pendings(), warnings=warnings)
        return say, parsed

    def _tool_tail(self, *, parsed, user_text: str, pendings: list[dict],
                   warnings: list[str], isolated: bool = False):
        """tool 协议的第三阶段：结算上一轮的预言 + 挂这一轮的新预言。

        走 tool call，复用 inner 那套「结构化输出」的思路。失败就退回空骨架 ——
        这一轮只是没学到东西，不影响对话。
        """
        from . import protocol

        if self.tool_llm is None and not isolated:
            return Tail()

        msgs = protocol.build_tail_messages(
            user_text=user_text, say=parsed.say, pendings=pendings,
            history=self.history,
        )
        try:
            if isolated:
                data = self._persona_structured(msgs, protocol.tail_tool(pendings),
                                               max_tokens=500, warnings=warnings)
            else:
                data = self.tool_llm(msgs, tool=protocol.tail_tool(pendings), max_tokens=500)
            if not isinstance(data, dict) or not isinstance(data.get("settle"), list) \
                    or "pending" not in data:
                raise ValueError("观察结果缺失或格式无效")
            return protocol.parse_tail(data)
        except Exception as e:
            self._probe["observation_failed"] = True
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

    @_atomic
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
            book.pause(r.note)
            self._save_repair_book(book)
            self._drop_unresolved(r.note)
            self.store.append("repair_paused", {"note": r.note, "reason": "explicit_stop",
                                                "evidence": user_text, "status": "tentative"})
            warnings.append("用户要求停止修复话题；停止追问，不推断长期雷区")
            return {**r.as_dict(), "verdict": "rejected"}

        # 不清楚：再给一次机会，但只给一次
        if r.attempts >= 2:
            book.pause(r.note)
            self._save_repair_book(book)
            self._drop_unresolved(r.note)
            self.store.append("repair_paused", {"note": r.note, "reason": "no_clear_response",
                                                "status": "unknown"})
            return {**r.as_dict(), "verdict": "paused"}
        return None

    @_atomic
    def _maybe_repair(self, say: str, warnings: list[str]) -> dict | None:
        """仅对之前开启的暂定记录认领修复措辞，不把当轮观察倒算成已修复。"""
        book = self._repair_book()
        active = book.active()
        if not active:
            return None

        for r in active:
            # 新观察尚未进入本轮生成上下文，不能倒算成针对它的修复。
            if r.turn_seq >= self._turn_seq:
                continue
            # 状态门槛：open = 还没试过，repairing = 试过一次但没回应。
            # 两种都允许再试，但 attempts >= 2 就停止，不把无反馈升级成禁区。
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

        明确的模型/语义证据可结算；没有证据则保留未知。
        关系信号独立记录，不覆盖预测真假；过期只结束保留。
        """
        out: list[dict] = []
        topic_verdicts = topic_verdicts or {}
        model_settle = model_settle or {}
        model_watch = model_watch or set()

        if self._last_user:
            observation = judge(extract_features(user_text),
                                baseline_formality=self._baseline_formality())
            if observation.relation_signal:
                self._probe["relation_signal"] = observation.relation_signal
                self.store.append("relation_observation", {
                    "turn_seq": self._turn_seq, "suspected_hurt": True,
                    "status": "tentative", "basis": "formality_shift",
                })
                self._last_formality_rise = (
                    formality_score(user_text) - self._baseline_formality())
                self._apply_outcome({}, SUSPECTED_HURT)

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
                semantic=model_item.outcome if model_item and model_item.evidence.strip() else None,
                similarity=self.sim,
                topic=tv.verdict if tv else None,
                topic_why=tv.why if tv else "",
                minimal_streak=self._minimal_streak.get(p["id"], 0),
            )

            # 模型明确说 watch -> 直接 WATCH，不要退回字面判定。
            # 丢掉这个意图的话，「只是附和没接话题」可能被判成 CONFIRMED，
            # 凭空提高命中率、污染校准账本。
            if p["id"] in model_watch:
                self._probe["settle"].append({"pid": p["id"], "verdict": WATCH,
                                              "settle": False,
                                              "why": "模型判 watch（再等等）"})
                continue

            verdict = judge(feat, baseline_formality=baseline_formality,
                            baseline_length=baseline_len)

            # 模型判了就以模型为准，证据也用模型写的
            if model_item is not None and feat.semantic and verdict.settle:
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

            if verdict.kind in (CONFIRMED, REFUTED) and self._is_delayed(p):
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
                "by": "semantic_judge" if tv and tv.verdict in ("returned", "left") else "heuristic",
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
        """保留关系修复；预测命中率不是用户认可度，不自动调人格。"""
        if outcome == SUSPECTED_HURT:
            # 同一轮只开一条修复 —— 用户被伤到一次，不是每条预言伤一次。
            if self._repair_opened_turn != self._turn_seq:
                self._repair_opened_turn = self._turn_seq
                self._open_repair(p)
            return

    @_atomic
    def _open_repair(self, p: dict) -> None:
        """观测到疑似受伤：记进未解决 + 开一条修复。

        关键：**受伤是「用户」的状态，不是「预言」的状态。**
        一轮里可能有好几条预言同时判成 SUSPECTED_HURT（说话的人只有一个，
        被伤到的也只有一个人）。所以要按「哪一轮」去重，不能按「哪条预言」。

        实测踩过：按 pending id 做 note 唯一性，结果一轮里 3 条预言同时
        受伤就开了 3 条修复，关系账本里堆了 3 条未解决；用户只被伤了一次。
        反过来用固定 note 又会让第二次受伤被静默吞掉。
        正解是 note 带**轮次**，不带 pending id。
        """
        source_turn = max(1, self._turn_seq - 1)
        note = f"可能被第 {source_turn} 轮那句话伤到，还没修复（观察于第 {self._turn_seq} 轮）"
        if note not in self.store.relation().get("unresolved", []):
            self.store.apply_relation_delta({"unresolved": [note]}, self.cfg.limits)

        # Prediction confidence is not evidence of hurt; a register shift stays tentative.
        conf = 0.0
        rise = self._last_formality_rise
        level = pick_level(confidence=conf, formality_rise=rise)

        book = self._repair_book()
        book.open(note, self._turn_seq, now_iso(), level)
        self._save_repair_book(book)
        self.store.append("repair_open", {"note": note, "level": level,
                                          "turn_seq": self._turn_seq,
                                          "source_turn_seq": source_turn,
                                          "status": "tentative", "limits": self.cfg.limits.as_dict(),
                                          "confidence": round(conf, 3),
                                          "pending_confidence": p.get("confidence"),
                                          "formality_rise": round(rise, 3)})

    # ------------------------------------------------------------ 辅助

    def _maybe_add_pending(self, parsed: ParsedTurn, turn_id: str) -> dict | None:
        tail = parsed.tail
        if not tail or not tail.pending or tail.pending.errors:
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

    @_atomic
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
    """收集回调输出；迭代包装器不保证实时性，实时输出直接使用 turn 的回调。"""
    box: dict[str, str] = {}

    def collect(piece: str) -> None:
        box["text"] = box.get("text", "") + piece

    engine.turn(user_text, on_delta=collect)
    for ch in box.get("text", ""):
        yield ch
