"""Executable dialogue choices, separate from factual inference and realization.

Evidence spans are checked mechanically, not certified semantically. Model-made
boundary proposals pass a conservative wording gate; confirmed commitments are
application-owned events. No model output can directly overwrite policy state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re

FOCUS_KINDS = ("practical", "feeling", "detail")
REQUESTS = ("open", "listen", "answer", "advice", "resume")


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _tuple(values, kind, name):
    if not isinstance(values, (list, tuple)) or any(not isinstance(v, kind) for v in values):
        raise ValueError(f"{name} has invalid items")
    return tuple(values)


def _outside_quotes(text):
    return re.sub(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"|‘[^’]*’', "", text)


def _direct_span(quote, user):
    """Reject quoted/hypothetical proposals; presence alone is not meaning."""
    outside = _outside_quotes(user)
    return bool(quote and quote in user and quote in outside and not re.search(
        r"如果|假如|假设|倘若|要是|举例|示例|转述|(?:朋友|同事|别人|有人|他|她)(?:说|表示)|\b(?:if|suppose|quoted)\b",
        outside, re.I))


def _command(text):
    return re.sub(r"\s+", "", text).strip("。.!！?？")


def _positive_request(user, request):
    """A deliberately small full-command grammar, not general intent inference."""
    text = _command(user)
    if request == "advice":
        return bool(re.fullmatch(
            r"(?:现在|这次|本轮|那么|那|好|还是|麻烦|请|你|可以|能不能|能)*"
            r"(?:帮我(?:想(?:想)?办法|出(?:个)?主意|想想怎么办)|"
            r"给我(?:一些|一点|一条|两条|几条|些|点)?建议)(?:吧|吗|呢)?", text))
    return bool(re.fullmatch(
        r"(?:现在|那|请|好|我们)?(?:继续|接着)(?:上次|之前|刚才)?(?:的)?"
        r"(?:约定|任务|工作|简历|改标题|改提纲)?(?:吧|好吗)?", text))


@dataclass(frozen=True)
class Source:
    id: str
    text: str

    def __post_init__(self):
        _text(self.id, "source id"); _text(self.text, "source text")
        if self.id == "user":
            raise ValueError("source id 'user' is reserved")


@dataclass(frozen=True)
class Quote:
    source_id: str
    text: str

    def __post_init__(self):
        _text(self.source_id, "quote source"); _text(self.text, "quote text")


@dataclass(frozen=True)
class Focus:
    id: str
    kind: str
    evidence: Quote

    def __post_init__(self):
        _text(self.id, "focus id")
        if self.kind not in FOCUS_KINDS or not isinstance(self.evidence, Quote):
            raise ValueError("Invalid focus kind or evidence")


@dataclass(frozen=True)
class Boundary:
    advice: str
    quote: str
    scope: str = "topic"

    def __post_init__(self):
        if self.advice not in ("allow", "deny") or self.scope not in ("topic", "session", "turn"):
            raise ValueError("Invalid advice boundary")
        _text(self.quote, "boundary quote")


@dataclass(frozen=True)
class ContentFrame:
    request: str = "open"
    request_quote: str = ""
    resume_id: str | None = None
    answer: str = ""
    facts: tuple[Quote, ...] = ()
    unknowns: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    focuses: tuple[Focus, ...] = ()
    boundaries: tuple[Boundary, ...] = ()

    def __post_init__(self):
        if self.request not in REQUESTS:
            raise ValueError("Unknown request")
        _text(self.request_quote, "request_quote", empty=True)
        if self.resume_id is not None:
            _text(self.resume_id, "resume_id")
            if self.request != "resume":
                raise ValueError("resume_id only applies to a resume request")
        _text(self.answer, "answer", empty=True)
        for name, kind in (("facts", Quote), ("unknowns", str), ("required", str),
                           ("focuses", Focus), ("boundaries", Boundary)):
            object.__setattr__(self, name, _tuple(getattr(self, name), kind, name))
        for value in (*self.unknowns, *self.required):
            _text(value, "content item")
        if len({f.id for f in self.focuses}) != len(self.focuses):
            raise ValueError("Duplicate focus ids")
        if len(self.boundaries) > 1:
            raise ValueError("At most one unambiguous boundary per input")

    def validate_evidence(self, user, sources=(), *, confirmed_request=False):
        _text(user, "user")
        sources = _tuple(sources, Source, "sources")
        if len({s.id for s in sources}) != len(sources):
            raise ValueError("Duplicate source ids")
        texts = {"user": user, **{s.id: s.text for s in sources}}
        for quote in (*self.facts, *(f.evidence for f in self.focuses)):
            if quote.source_id not in texts or quote.text not in texts[quote.source_id]:
                raise ValueError("Evidence quote is not in its actual source")
        if self.request != "open" and not (self.request_quote and self.request_quote in _outside_quotes(user)):
            raise ValueError("A non-open request needs direct user evidence")
        if self.request in ("advice", "resume") and not confirmed_request and not _positive_request(user, self.request):
            raise ValueError("Privileged request needs a supported full command or application confirmation")
        for boundary in self.boundaries:
            if boundary.quote not in user:
                raise ValueError("Boundary evidence is not in the current user input")

    @classmethod
    def from_json(cls, text):
        def pairs(items):
            out = {}
            for key, value in items:
                if key in out:
                    raise ValueError("Duplicate JSON key")
                out[key] = value
            return out
        def bad(value):
            raise ValueError("Non-finite JSON value")
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=bad)
        if not isinstance(obj, dict):
            raise ValueError("Content frame must be a JSON object")
        values = dict(obj)
        for name in ("facts", "focuses", "boundaries", "unknowns", "required"):
            if name in values and not isinstance(values[name], list):
                raise ValueError(f"{name} must be an array")
        values["facts"] = tuple(Quote(**v) for v in values.get("facts", []))
        focuses = []
        for value in values.get("focuses", []):
            if not isinstance(value, dict) or set(value) != {"id", "kind", "evidence"}:
                raise ValueError("Invalid focus fields")
            focuses.append(Focus(value["id"], value["kind"], Quote(**value["evidence"])))
        values["focuses"] = tuple(focuses)
        values["boundaries"] = tuple(Boundary(**v) for v in values.get("boundaries", []))
        return cls(**values)


@dataclass(frozen=True)
class PolicyConfig:
    intervention: float = 0.5
    practical: float = 0.5
    feeling: float = 0.5
    detail: float = 0.5
    seed: str = "persona-policy-v1"

    def __post_init__(self):
        for name in ("intervention", *FOCUS_KINDS):
            _unit(getattr(self, name), name)
        _text(self.seed, "seed")


@dataclass(frozen=True)
class Commitment:
    id: str
    text: str
    topic: str
    quote: str
    status: str = "open"

    def __post_init__(self):
        for name in ("id", "text", "topic", "quote"):
            _text(getattr(self, name), name)
        if self.status not in ("open", "completed", "cancelled"):
            raise ValueError("Unknown commitment status")


@dataclass(frozen=True)
class Permission:
    scope: str
    advice: str
    expires_at: str | None = None

    def __post_init__(self):
        _text(self.scope, "scope")
        if self.advice not in ("allow", "deny"):
            raise ValueError("Invalid permission")
        if self.expires_at is not None:
            _instant(self.expires_at)


def _instant(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Policy time must include a timezone")
    return parsed


@dataclass(frozen=True)
class PolicyState:
    revision: int = 0
    permissions: tuple[Permission, ...] = ()
    commitments: tuple[Commitment, ...] = ()

    def __post_init__(self):
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("Invalid state revision")
        object.__setattr__(self, "permissions", _tuple(self.permissions, Permission, "permissions"))
        object.__setattr__(self, "commitments", _tuple(self.commitments, Commitment, "commitments"))
        if len({c.id for c in self.commitments}) != len(self.commitments):
            raise ValueError("Duplicate commitments")

    def advice_permission(self, topic, *, at=None):
        for value in reversed(self.permissions):
            if value.scope not in ("*", topic):
                continue
            if value.expires_at is not None:
                if at is None:
                    raise ValueError("An explicit time is required for expiring state")
                if _instant(at) >= _instant(value.expires_at):
                    continue
            return value.advice
        return "unspecified"

    def for_topic(self, topic):
        return tuple(c for c in self.commitments if c.topic == topic and c.status == "open")

    @classmethod
    def from_dict(cls, obj):
        return cls(obj["revision"], tuple(Permission(**p) for p in obj["permissions"]),
                   tuple(Commitment(**c) for c in obj["commitments"]))


_ADVICE_CONTROL = re.compile(
    r"(?P<time>今天|今晚|本轮|这轮|这一轮|这次|现在|暂时|目前|以后|今后)?"
    r"(?P<global>所有话题都|任何话题都|不管聊什么都)?"
    r"(?P<operation>不要|别|不需要|不用|不必|可以|允许)(?:再)?"
    r"(?:给(?:我)?建议|替我出主意|帮我想办法)(?:了|吧)?"
)


def parse_advice_control(user, topic, *, at):
    """Recognize only a supported complete command, independently of a model.

    A question, quote, hypothesis or mixed sentence is not a control. The same
    grammar gates model proposals; a proposal's clipped span never grants scope.
    """
    _text(user, "user"); _text(topic, "topic")
    if topic == "*":
        raise ValueError("'*' is reserved for global boundaries")
    if "?" in user or "？" in user:
        return None
    match = _ADVICE_CONTROL.fullmatch(_command(user))
    if match is None:
        return None
    current = _instant(at).astimezone(timezone(timedelta(hours=8)))
    transient = match["time"] in ("本轮", "这轮", "这一轮", "这次", "现在", "暂时", "目前")
    until = ((current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
             if not transient and match["time"] in ("今天", "今晚") else None)
    return {"scope": "*" if match["global"] else topic,
            "advice": "allow" if match["operation"] in ("可以", "允许") else "deny",
            "quote": user, "expires_at": until, "transient": transient}


def boundary_changes(frame, user, topic, *, at):
    control = parse_advice_control(user, topic, at=at)
    accepted, rejected = [], []
    for boundary in frame.boundaries:
        if control is not None and boundary.quote in user and boundary.advice == control["advice"]:
            accepted.append({**control, "quote": boundary.quote})
            continue
        rejected.append({**asdict(boundary), "reason": "unsupported_or_ambiguous_full_control_command"})
    return accepted, rejected


def apply_boundary(state, event):
    value = Permission(event["scope"], event["advice"], event.get("expires_at"))
    return replace(state, revision=state.revision + 1, permissions=(*state.permissions, value))


def update_commitment(state, *, operation, id, quote, topic, text=""):
    """Application-confirmed event. Never called from generated frame or SAY."""
    _text(id, "commitment id"); _text(quote, "confirmation evidence"); _text(topic, "topic")
    items = list(state.commitments)
    found = next((i for i, c in enumerate(items) if c.id == id), None)
    if operation == "confirm":
        _text(text, "commitment text")
        if found is not None:
            raise ValueError("Commitment id already exists")
        items.append(Commitment(id, text, topic, quote))
    elif operation in ("complete", "cancel"):
        if found is None or items[found].topic != topic or items[found].status != "open":
            raise ValueError("No open commitment for that id and topic")
        items[found] = replace(items[found], status="completed" if operation == "complete" else "cancelled")
    else:
        raise ValueError("Unknown commitment operation")
    return replace(state, revision=state.revision + 1, commitments=tuple(items))


@dataclass(frozen=True)
class Decision:
    action: str
    allowed_actions: tuple[str, ...]
    focus_id: str | None
    focus_kind: str | None
    commitment_id: str | None
    advice_permission: str
    state_revision: int
    draw: float
    reason: str


def decide(frame, state, config, *, topic, key, at=None):
    """Deterministic given frame, state, parameters and draw key; no model call."""
    _text(topic, "topic"); _text(key, "decision key")
    raw = hashlib.sha256((config.seed + "\0" + key).encode()).digest()
    draw = int.from_bytes(raw[:8], "big") / 2**64
    permission = state.advice_permission(topic, at=at)
    commitment = None
    if frame.request == "advice":
        action, allowed, reason = "advise", ("advise",), "explicit_advice_request_this_turn"
    elif frame.request == "answer":
        action, allowed, reason = "answer", ("answer",), "answer_requested_content"
    elif frame.request == "listen":
        action, allowed, reason = "listen", ("listen",), "explicit_listening_request"
    elif frame.request == "resume":
        available = state.for_topic(topic)
        candidates = [c for c in available if frame.resume_id is None or c.id == frame.resume_id]
        if len(available) > 1:
            candidates = [c for c in candidates if c.id in frame.request_quote or c.text in frame.request_quote]
        if len(candidates) == 1:
            commitment = candidates[0]
            action, allowed, reason = "continue", ("continue",), "resume_confirmed_open_commitment"
        else:
            action, allowed, reason = "clarify", ("clarify",), "resume_target_not_unique_or_unavailable"
    else:
        allowed = ("respond",) if permission == "deny" else ("respond", "advise")
        action = "advise" if "advise" in allowed and draw < config.intervention else "respond"
        reason = "advice_masked_by_user_boundary" if permission == "deny" else "configured_intervention_draw"
    focus = max(frame.focuses, key=lambda f: (getattr(config, f.kind), -frame.focuses.index(f))) if frame.focuses else None
    return Decision(action, allowed, focus.id if focus else None, focus.kind if focus else None,
                    commitment.id if commitment else None, permission, state.revision, draw, reason)
