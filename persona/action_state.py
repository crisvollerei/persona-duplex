"""Turn-local projection of reported state; it neither executes nor verifies actions."""
from __future__ import annotations

from dataclasses import dataclass
import json

from .actions import ActionContext


_SOURCE_FIELDS = ("planned_quote", "before_quote", "evidence_quote", "result_quote", "current_quote")
_MANUAL = """[调用方提供的本轮现实动作状态]
只在同一 action_id 内推导。planned_quote 是预期变化，before_quote 始终只代表先前观察，evidence_quote 是本轮有关进展的来源。缺失的字段不给事实。
pending 有明确的未执行依据；只有该来源明确报告现态的 current_quote 才可作为现态，否则具体值未知。confirmed 有明确的完成依据，报告该来源里的 result_quote，即使与计划不同。unknown 的现态和完成情况均不明，不要猜已完成或未完成。
写交接稿、摘要或回答现况时，只写有依据的现态；拟稿不改变现实。若要求未来才能使用的稿件，把前提写清。状态不能供其他动作借用，故事中的动作也不使现实动作完成。按角色自然完成当轮任务，不必逐项解释字段。"""


@dataclass(frozen=True)
class ActionState:
    """Caller classification of a real action, with literal spans of the current user turn."""

    action_quote: str
    status: str
    evidence_quote: str = ""
    planned_quote: str = ""
    before_quote: str = ""
    result_quote: str = ""
    current_quote: str = ""

    def __post_init__(self):
        if not isinstance(self.action_quote, str) or not self.action_quote.strip():
            raise ValueError("state needs the literal quote of its action")
        if self.status not in ("pending", "confirmed", "unknown"):
            raise ValueError("expected pending, confirmed or unknown action status")
        for name in _SOURCE_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or (value and not value.strip()):
                raise ValueError(f"{name} must be a literal quote or empty")
        if self.status != "unknown" and not self.evidence_quote:
            raise ValueError("pending or confirmed state needs explicit evidence")
        if self.status != "unknown" and any(self.evidence_quote in s for s in
                                            (self.action_quote, self.planned_quote)):
            raise ValueError("a request or plan alone is not execution evidence")
        if self.current_quote and (self.status != "pending" or self.current_quote not in self.evidence_quote):
            raise ValueError("current_quote needs an explicit observation in the pending evidence")
        if self.status == "confirmed":
            if not self.result_quote or self.result_quote not in self.evidence_quote:
                raise ValueError("confirmed state needs a result quoted from its evidence")
        elif self.result_quote:
            raise ValueError("only confirmed state may carry a reported result")


@dataclass(frozen=True)
class ActionStateContext(ActionContext):
    """Opt-in extension of ActionContext. Keep evidence on separate, real action IDs."""

    states: tuple[ActionState, ...]

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.states, (tuple, list)) or not self.states:
            raise ValueError("states must be a nonempty ordered list or tuple")
        states = tuple(self.states)
        if not all(isinstance(s, ActionState) for s in states):
            raise TypeError("states must contain ActionState instances")
        if len({s.action_quote for s in states}) != len(states):
            raise ValueError("conflicting or duplicate state for one action")
        physical = {a.quote for a in self.actions if a.kind == "user_physical"}
        if any(s.action_quote not in physical for s in states):
            raise ValueError("a real state must refer to an annotated user_physical action")
        # A known line of fiction or requested wording may name the desired outcome,
        # but that span cannot double as evidence of an observed real state.
        nonphysical = [span for a in self.actions if a.kind != "user_physical"
                       for span in (a.quote, a.frame_quote) if span]
        for state in states:
            for span in (state.evidence_quote, state.before_quote, state.result_quote, state.current_quote):
                if span and any(span in excluded for excluded in nonphysical):
                    raise ValueError("use a distinct real source outside the story or text-task quote")
        evidence = [s.evidence_quote for s in states if s.evidence_quote]
        if len(set(evidence)) != len(evidence):
            raise ValueError("each reported action needs distinguishable evidence")
        object.__setattr__(self, "states", states)

    def validate_evidence(self, user_text: str) -> None:
        super().validate_evidence(user_text)
        for state in self.states:
            for name in _SOURCE_FIELDS:
                span = getattr(state, name)
                if span and (user_text.find(span) < 0 or user_text.find(span) != user_text.rfind(span)):
                    raise ValueError(f"{name} must be a unique literal quote in the current user text")

    def _state_rows(self):
        ids = {a.quote: f"a{i}" for i, a in enumerate(self.actions, 1)}
        return [dict(action_id=ids[s.action_quote], status=s.status,
                     **{name: getattr(s, name) for name in _SOURCE_FIELDS if getattr(s, name)})
                for s in self.states]

    @staticmethod
    def _project(rows):
        """Return source references, not a fresh assertion about the world."""
        projected = []
        for row in rows:
            ref = row["action_id"] + "."
            status = row["status"]
            field = {"confirmed": "result_quote", "pending": "current_quote", "unknown": None}[status]
            projected.append(dict(action_id=row["action_id"], completion={
                "confirmed": "reported_complete", "pending": "reported_not_done", "unknown": "unknown"}[status],
                current_ref=ref + field if field in row else None,
                evidence_ref=ref + "evidence_quote" if "evidence_quote" in row else None,
                last_known_ref=ref + "before_quote" if "before_quote" in row else None,
                plan_ref=ref + "planned_quote" if "planned_quote" in row else None))
        return projected

    def record(self):
        record = super().record()
        rows = self._state_rows()
        record["action_state"] = dict(rules_version=1, source="caller_annotation", rows=rows,
                                      derivation="within_action_evidence_refs", projection=self._project(rows),
                                      output_validation="not_assessed")
        return record

    def manual_prompt(self):
        # Keep ownership formatting identical for a contrast about the STATE component.
        rows = self._state_rows()
        return "\n\n".join((super().prompt(), _MANUAL, json.dumps(rows, ensure_ascii=False)))

    def prompt(self):
        refs = self._project(self._state_rows())
        return (self.manual_prompt() + "\n[按上述规则推导的本轮引用]\n"
                + "current_ref 为 null 表示无法确定现态；plan_ref 不是已完成事实。\n"
                + json.dumps(refs, ensure_ascii=False))
