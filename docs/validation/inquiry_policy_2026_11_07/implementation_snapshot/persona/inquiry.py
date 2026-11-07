"""Application-owned question slots and deterministic inquiry permissions.

Evidence presence is a mechanical check, not approval of a question's meaning.
The application reviews questions and confirms answers. Generated content never
creates a slot, grants permission, refunds a quota, or resolves a question here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta, timezone
import hashlib
import json
import re

from .policy import (FOCUS_KINDS, ContentFrame, Decision, PolicyConfig, Quote,
                     Source, _direct_span, _instant, _text, _tuple, _unit)


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _topic(value):
    _text(value, "topic")
    if value == "*":
        raise ValueError("'*' is reserved for global permissions")
    return value


@dataclass(frozen=True)
class QuestionSlot:
    id: str
    topic: str
    question: str
    kind: str
    evidence: Quote
    purpose: str

    def __post_init__(self):
        _text(self.id, "question id"); _topic(self.topic)
        _text(self.question, "question"); _text(self.purpose, "question purpose")
        if self.kind not in FOCUS_KINDS or not isinstance(self.evidence, Quote):
            raise ValueError("Invalid question kind or evidence")


def validate_slots(slots, user, sources=()):
    """Return normalized, uniquely identified slots with actual quoted evidence.

    The caller remains responsible for question relevance and non-advisory
    meaning. A quoted instruction cannot itself authorize a question.
    """
    slots = _tuple(slots, QuestionSlot, "question slots")
    sources = _tuple(sources, Source, "sources")
    _text(user, "user")
    if len({slot.id for slot in slots}) != len(slots):
        raise ValueError("Duplicate question ids")
    if len({source.id for source in sources}) != len(sources):
        raise ValueError("Duplicate source ids")
    actual = {"user": user, **{source.id: source.text for source in sources}}
    for slot in slots:
        evidence = slot.evidence
        if evidence.source_id not in actual or evidence.text not in actual[evidence.source_id]:
            raise ValueError("Question evidence is not in its actual source")
    return slots


@dataclass(frozen=True)
class InquiryConfig:
    initiative: float = .5
    max_per_topic: int = 2
    max_total: int = 6
    cooldown_turns: int = 1

    def __post_init__(self):
        _unit(self.initiative, "initiative")
        for name in ("max_per_topic", "max_total", "cooldown_turns"):
            _count(getattr(self, name), name)


@dataclass(frozen=True)
class QuestionChoice:
    """Experimental choice proposal, not permission or a state update.

    An application may use this to substitute a model's slot preference for the
    program draw. The shared executor must still check actual eligible ids.
    """
    slot_id: str | None

    def __post_init__(self):
        if self.slot_id is not None:
            _text(self.slot_id, "proposed question id")


@dataclass(frozen=True)
class QuestionRecord:
    slot: QuestionSlot
    status: str = "pending"
    answer: str = ""

    def __post_init__(self):
        if not isinstance(self.slot, QuestionSlot):
            raise ValueError("Question record requires a QuestionSlot")
        if self.status not in ("pending", "answered", "declined"):
            raise ValueError("Unknown question status")
        _text(self.answer, "answer", empty=self.status != "answered")
        if self.status != "answered" and self.answer:
            raise ValueError("Only an answered question can contain an answer")


@dataclass(frozen=True)
class InquiryPermission:
    scope: str
    mode: str
    expires_at: str | None = None

    def __post_init__(self):
        _text(self.scope, "permission scope")
        if self.mode not in ("allow", "deny"):
            raise ValueError("Invalid inquiry permission")
        if self.expires_at is not None:
            _instant(self.expires_at)


@dataclass(frozen=True)
class InquiryState:
    revision: int = 0
    permissions: tuple[InquiryPermission, ...] = ()
    records: tuple[QuestionRecord, ...] = ()

    def __post_init__(self):
        _count(self.revision, "inquiry revision")
        object.__setattr__(self, "permissions", _tuple(self.permissions, InquiryPermission, "inquiry permissions"))
        object.__setattr__(self, "records", _tuple(self.records, QuestionRecord, "question records"))
        if len({record.slot.id for record in self.records}) != len(self.records):
            raise ValueError("Duplicate question ids in state")

    def permission(self, topic, at=None):
        _topic(topic)
        for permission in reversed(self.permissions):
            if permission.scope not in ("*", topic):
                continue
            if permission.expires_at is not None:
                if at is None:
                    raise ValueError("An explicit time is required for expiring inquiry state")
                if _instant(at) >= _instant(permission.expires_at):
                    continue
            return permission.mode
        return "unspecified"

    @classmethod
    def from_dict(cls, obj):
        if not isinstance(obj, dict) or set(obj) != {"revision", "permissions", "records"}:
            raise ValueError("Invalid inquiry state fields")
        if not isinstance(obj["permissions"], (list, tuple)) or not isinstance(obj["records"], (list, tuple)):
            raise ValueError("Inquiry state collections must be arrays")
        permissions = tuple(InquiryPermission(**row) for row in obj["permissions"])
        records = []
        for row in obj["records"]:
            if not isinstance(row, dict) or set(row) != {"slot", "status", "answer"}:
                raise ValueError("Invalid question record fields")
            slot = dict(row["slot"])
            slot["evidence"] = Quote(**slot["evidence"])
            records.append(QuestionRecord(QuestionSlot(**slot), row["status"], row["answer"]))
        return cls(obj["revision"], permissions, tuple(records))


def delivered(state, slot):
    """Record one confirmed delivery. Retrying or redefining the id is rejected."""
    if not isinstance(state, InquiryState) or not isinstance(slot, QuestionSlot):
        raise TypeError("Delivery requires InquiryState and QuestionSlot")
    if any(record.slot.id == slot.id for record in state.records):
        raise ValueError("Question id was already delivered")
    return replace(state, revision=state.revision + 1, records=(*state.records, QuestionRecord(slot)))


def resolve(state, id, topic, operation, user, quote):
    """Application-confirmed answer or refusal; no inference from silence/text tone.

    Only the exact answer span is stored as an answer. A refusal has no answer;
    callers should retain its evidence in their event log. Quotas are not reset.
    """
    if not isinstance(state, InquiryState):
        raise TypeError("Resolution requires InquiryState")
    _text(id, "question id"); _topic(topic); _text(user, "user"); _text(quote, "resolution quote")
    if operation not in ("answer", "decline"):
        raise ValueError("Unknown question resolution")
    if not _direct_span(quote, user):
        raise ValueError("Question resolution requires direct user evidence")
    records = list(state.records)
    found = next((i for i, record in enumerate(records) if record.slot.id == id), None)
    if found is None or records[found].slot.topic != topic or records[found].status != "pending":
        raise ValueError("No pending question for that id and topic")
    records[found] = replace(records[found], status="answered" if operation == "answer" else "declined",
                             answer=quote if operation == "answer" else "")
    return replace(state, revision=state.revision + 1, records=tuple(records))


_CONTROL = re.compile(
    r"(?P<time>今天|今晚|本轮|这轮|这一轮|这次|现在|暂时|目前|以后|今后)?"
    r"(?P<global>所有话题都|任何话题都|不管聊什么都)?"
    r"(?P<operation>(?:不要|别|不用|不必|不需要)(?:再)?(?:追问|问我|问)(?:了|吧)?|"
    r"(?:可以|允许)(?:继续)?追问(?:了|吧)?|继续(?:追问|问)(?:吧)?)"
)


def parse_control(user, topic, at):
    """Parse only a complete supported command, never a model's clipped quote."""
    _text(user, "user"); _topic(topic)
    if "?" in user or "？" in user:
        return None
    text = re.sub(r"\s+", "", user).rstrip("。.!！")
    match = _CONTROL.fullmatch(text)
    if match is None:
        return None
    current = _instant(at).astimezone(timezone(timedelta(hours=8)))
    transient = match["time"] in ("本轮", "这轮", "这一轮", "这次", "现在", "暂时", "目前")
    until = ((current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
             if match["time"] in ("今天", "今晚") else None)
    mode = "allow" if match["operation"].startswith(("可以", "允许", "继续")) else "deny"
    return dict(mode=mode, scope="*" if match["global"] else topic, quote=user,
                expires_at=until, transient=transient)


def apply_control(state, event):
    """Apply a parsed control; callers persist only non-transient events."""
    if not isinstance(state, InquiryState):
        raise TypeError("Control requires InquiryState")
    if not isinstance(event, dict) or set(event) != {"mode", "scope", "quote", "expires_at", "transient"}:
        raise ValueError("Invalid inquiry control event")
    _text(event["quote"], "control quote")
    if type(event["transient"]) is not bool or (event["transient"] and event["expires_at"] is not None):
        raise ValueError("Invalid transient inquiry control")
    permission = InquiryPermission(event["scope"], event["mode"], event["expires_at"])
    return replace(state, revision=state.revision + 1, permissions=(*state.permissions, permission))


def select_question(frame, base_decision, policy_config, inquiry_config, state, slots,
                    topic, key, topic_turns_since_question, *, at=None):
    """Choose one already reviewed information slot without changing content facts.

    Permission and delivery limits mask candidates before the initiative draw.
    A same-topic pending question prevents another question regardless of elapsed
    turns. Supplying reviewed slots and validating their evidence is a caller duty.
    """
    if not isinstance(frame, ContentFrame) or not isinstance(base_decision, Decision):
        raise TypeError("Inquiry selection requires content and a base decision")
    if not isinstance(policy_config, PolicyConfig) or not isinstance(inquiry_config, InquiryConfig):
        raise TypeError("Inquiry selection requires explicit configurations")
    if not isinstance(state, InquiryState):
        raise TypeError("Inquiry selection requires InquiryState")
    _topic(topic); _text(key, "inquiry key")
    if topic_turns_since_question is not None:
        _count(topic_turns_since_question, "topic turns since question")
    slots = _tuple(slots, QuestionSlot, "question slots")
    if len({slot.id for slot in slots}) != len(slots):
        raise ValueError("Duplicate question ids")
    known = {record.slot.id: record.slot for record in state.records}
    if any(slot.id in known and slot != known[slot.id] for slot in slots):
        raise ValueError("A delivered question id cannot be redefined")
    encoded = json.dumps([policy_config.seed, "inquiry", key], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # Use 53 bits so even the largest hash cannot round to 1.0.
    draw = (int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") >> 11) / 2**53
    result = dict(selected_id=None, eligible_ids=[], reason="", draw=draw)

    def blocked(reason):
        return {**result, "reason": reason}

    if base_decision.action != "respond" or frame.request != "open":
        return blocked("base_action_not_open_response")
    if state.permission(topic, at=at) == "deny":
        return blocked("inquiry_denied")
    if any(record.slot.topic == topic and record.status == "pending" for record in state.records):
        return blocked("pending_question")
    if sum(record.slot.topic == topic for record in state.records) >= inquiry_config.max_per_topic:
        return blocked("topic_question_limit")
    if len(state.records) >= inquiry_config.max_total:
        return blocked("total_question_limit")
    if topic_turns_since_question is not None and topic_turns_since_question < inquiry_config.cooldown_turns:
        return blocked("question_cooldown")
    eligible = [slot for slot in slots if slot.topic == topic and slot.id not in known]
    if not eligible:
        return blocked("no_unasked_question_slot")
    result["eligible_ids"] = [slot.id for slot in eligible]
    if draw >= inquiry_config.initiative:
        return {**result, "reason": "initiative_draw_declined"}
    selected = max(eligible, key=lambda slot: getattr(policy_config, slot.kind))
    return {**result, "selected_id": selected.id, "reason": "selected_inquiry_slot"}
