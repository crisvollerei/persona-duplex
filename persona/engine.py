"""单轮编排：组装 → 生成 → 解析 → 落账。

一次生成，产出「说出去的话」+「状态变更」。
观测在线写，结算下一轮填。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .config import DISPLACE_NONE, PersonaConfig
from .parse import ParsedTurn, parse
from .prompt import (
    build_context_block,
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
    ):
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.stream_llm = stream_llm
        self.sim = similarity
        self.judge = judge
        self.history: list[dict] = []
        self._last_user = ""
        self._last_say = ""
        self._topic: list[str] = []
        self._pending_topic: dict[str, list[str]] = {}
        self._turn_seq = 0

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

        # ---- 1. 修复反应判定。这个必须同步：它决定上下文里要不要放
        #        「你伤到对方了」的提示。但它只看语域，不调模型。
        repair_closed = self._judge_repair_response(user_text, warnings)

        # ---- 2. 把话题判定发射出去。注意这里只是发射，不等 ——
        #        它会在主模型生成的同时并行跑，生成完再收。
        pendings_snapshot = self.store.pendings()
        judge_handle = self._submit_topic_judge(user_text, pendings_snapshot)

        # ---- 3. 组装上下文。此时预言都还挂着 —— 模型需要在上下文里
        #        看见它们，才能在 TAIL 里给出自己的判定。
        relation = self.store.relation()
        pendings = pendings_snapshot
        judgments = self.store.judgments()

        sys_prompt = build_system_prompt(self.cfg)
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

        # ---- 4. 生成（流式或整段）。话题判定就在这段时间里跑完了。
        raw, parsed = self._generate(messages, on_delta)

        # ---- 5. 收判定。deadline 从这一轮开始算，所以如果生成已经
        #        用光了预算，这里会立刻返回字面兜底，绝不额外加延迟。
        topic_verdicts, topic_source = self._collect_topic_judge(
            judge_handle, user_text, pendings_snapshot, turn_started
        )
        if topic_source == "timeout":
            warnings.append("话题判定超时，退回字面判定")
        elif topic_source == "bigram":
            warnings.append("话题判定不可用，退回字面判定")

        # ---- 6. 结算（用刚收到的语义判定）
        settled = self._settle_with(user_text, warnings, topic_verdicts)

        warnings.extend(parsed.errors)
        if parsed.declared:
            warnings.append("位移被声明了（硬失败）：" + " / ".join(parsed.declared[:2]))
        if not parsed.is_consistent():
            warnings.append("HEAD 说有位移，但 direct 与 say 无法区分 —— 疑似装深沉")

        head = parsed.head
        if head and not self.cfg.can_displace(head.displace):
            warnings.append(f"角色 legibility={self.cfg.role.legibility_max} 不允许位移"
                            f"（模型写了 {head.displace}）")

        say = parsed.say or raw.strip()

        # ---- 4. 模型自己在 TAIL 里给的判定，补在启发式之后
        tail_settled = self._apply_tail_settle(parsed)
        if tail_settled:
            # 模型判掉了的，之前那些「挂起观察」的提示就过时了，撤掉
            done_ids = {s["id"] for s in tail_settled}
            warnings = [w for w in warnings
                        if not any(f"预言 {pid} 挂起" in w for pid in done_ids)]
            settled.extend(tail_settled)
        # ---- 5. 这一句是不是修复动作
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

        self.history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": say},
        ])
        self._last_user = user_text
        self._last_say = say
        self._topic.append(user_text)

        return TurnResult(
            turn_id=turn_id, say=say, parsed=parsed, stance=stance,
            displaced=displaced, settled=settled, pending_added=pending_added,
            relation_changed=relation_changed, warnings=warnings,
            repair_opened=repair_opened, repair_closed=repair_closed,
        )

    # ------------------------------------------------------------ 生成

    def _generate(self, messages, on_delta):
        """流式优先，失败退回整段。流式只是体验优化，不是正确性依赖。"""
        if on_delta is not None and self.stream_llm is not None:
            from .stream import StreamParser

            sp = StreamParser()
            try:
                for chunk in self.stream_llm(messages):
                    for piece in sp.feed(chunk):
                        on_delta(piece)
                parsed = sp.finish()
                return sp.say_text or "", parsed
            except Exception:
                # 流断了不能吞掉整轮 —— 退回整段重试一次
                pass

        raw = self.llm(messages)
        return raw, parse(raw)

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
        不是在照顾对方。
        """
        book = self._repair_book()
        active = book.active()
        if not active:
            return None

        for r in active:
            if r.state != "open":
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
                     topic_verdicts: dict | None = None) -> list[dict]:
        """拿用户这一句话，去核对上一轮的预言。"""
        out: list[dict] = []
        topic_verdicts = topic_verdicts or {}
        pendings = self.store.pendings()
        if not pendings:
            return out

        for p in self.store.sweep_expired(self.cfg.watch_ttl_hours):
            self.store.append("pending_expire", {"pending_id": p["id"],
                                                 "reason": "timeout"})
            out.append({**p, "outcome": "EXPIRED", "label": OUTCOME_LABEL["EXPIRED"]})

        semantic: dict[str, str] = {}
        baseline_len = self._baseline_length()

        for p in pendings:
            gap = _hours_since(p.get("created", ""))
            extra = list(self._pending_topic.get(p["id"], []))
            tv = topic_verdicts.get(p["id"])
            feat = extract_features(
                user_text,
                pending_claim=p.get("claim", ""),
                predicts=p.get("predicts", ""),
                prev_ai_say=self._last_say,
                prev_user_text=self._last_user,
                extra_refs=extra,
                gap_hours=gap,
                semantic=semantic.get(p["id"]),
                similarity=self.sim,
                topic=tv.verdict if tv else None,
                topic_why=tv.why if tv else "",
            )

            verdict = judge(feat, baseline_formality=self._baseline_formality(),
                            baseline_length=baseline_len)

            if verdict.kind == WATCH and gap > self.cfg.watch_ttl_hours / 3 \
                    and feat.mentions_topic:
                verdict = judge_delayed(feat, expired=False)

            if not verdict.settle:
                warnings.append(f"预言 {p['id']} 挂起观察：{verdict.reason}")
                continue

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
                self._apply_outcome(p, outcome)

        return out

    def _apply_tail_settle(self, parsed: ParsedTurn) -> list[dict]:
        """TAIL 里模型自己给的判定，当轮生效。主路径。"""
        if not parsed.tail or not parsed.tail.settle:
            return []
        live = {p["id"]: p for p in self.store.pendings()}
        out: list[dict] = []
        cur_seq = self._turn_seq
        for item in parsed.tail.settle:
            p = live.get(item.pending_id)
            if p is None:
                continue
            delayed = cur_seq - int(p.get("turn_seq", cur_seq)) >= 2
            outcome = item.outcome.upper()
            if delayed and outcome in ("CONFIRMED", "REFUTED"):
                outcome = f"{outcome}'"
            closed = self.store.close_pending(
                item.pending_id, outcome, item.evidence, delayed=delayed
            )
            if closed:
                self.store.append("pending_settle", {
                    "pending_id": item.pending_id,
                    "outcome": outcome,
                    "evidence": item.evidence,
                })
                out.append({**closed, "label": OUTCOME_LABEL.get(outcome, "校准账本")})
                self._apply_outcome(p, outcome)
                live.pop(item.pending_id, None)
        return out

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
            self._open_repair(p)
            return

        try:
            conf = float(p.get("confidence", 0.0))
        except (TypeError, ValueError):
            return
        if outcome in (CONFIRMED, CONFIRMED_DELAYED) and conf < 0.4:
            self.cfg.role.bias = min(0.4, self.cfg.role.bias + 0.005)
        elif outcome in (REFUTED, REFUTED_DELAYED) and conf > 0.7:
            self.cfg.role.bias = max(-0.4, self.cfg.role.bias - 0.005)

    def _open_repair(self, p: dict) -> None:
        """检测到受伤：记进未解决 + 开一条修复。"""
        note = "可能被上一轮某句话伤到，还没修复"
        if note not in self.store.relation().get("unresolved", []):
            self.store.apply_relation_delta({"unresolved": [note]}, self.cfg.limits)

        conf = float(p.get("confidence", 0.0) or 0.0)
        level = pick_level(confidence=max(conf, 0.5), formality_rise=0.3)

        book = self._repair_book()
        book.open(note, self._turn_seq, now_iso(), level)
        self._save_repair_book(book)
        self.store.append("repair_open", {"note": note, "level": level})

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
        """慢速跟踪用户的语域基线。这是「尺子」，要稳。"""
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
