"""Question authority, actual-delivery budgets and deterministic selection."""
from dataclasses import asdict, replace
import json
import unittest

from persona.inquiry import (InquiryConfig, InquiryPermission, InquiryState,
                            QuestionChoice, QuestionRecord, QuestionSlot, apply_control, delivered,
                            parse_control, resolve, select_question, validate_slots)
from persona.policy import (ContentFrame, PolicyConfig, PolicyState, Quote, Source,
                            decide)

NOW = "2026-11-07T21:00:00+08:00"


def slot(id="deadline", *, topic="work", kind="practical", source="user", text="报告还没写完"):
    return QuestionSlot(id, topic, "报告最晚何时需要交？", kind, Quote(source, text), "确认排期所需的截止时间")


def selection(slots=None, *, state=None, request="open", inquiry=None, policy=None,
              topic="work", key="1", elapsed=None, at=NOW):
    frame = ContentFrame(request=request, request_quote="当前原话" if request != "open" else "")
    policy = policy or PolicyConfig(intervention=0)
    base = decide(frame, PolicyState(), policy, topic=topic, key=key)
    return select_question(frame, base, policy, inquiry or InquiryConfig(initiative=1),
                           state or InquiryState(), (slot(),) if slots is None else slots,
                           topic, key, elapsed, at=at)


class TestQuestionData(unittest.TestCase):
    def test_slots_require_exact_named_evidence_and_do_not_invent_it(self):
        actual = slot(source="task")
        self.assertEqual(validate_slots([actual], "我还没想好", [Source("task", "报告还没写完，格式未知")]), (actual,))
        for invalid in (slot(), slot(source="missing"), slot(source="task", text="报告已经完成")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_slots([invalid], "我还没想好", [Source("task", "报告还没写完，格式未知")])

    def test_duplicate_ids_sources_and_invalid_fields_are_rejected(self):
        with self.assertRaises(ValueError): validate_slots([slot(), slot()], "报告还没写完")
        with self.assertRaises(ValueError):
            validate_slots([], "输入", [Source("x", "一"), Source("x", "二")])
        for change in ({"id":""}, {"topic":"*"}, {"question":""}, {"purpose":" "},
                       {"kind":"mind_reading"}, {"evidence": {"source_id":"user", "text":"原话"}}):
            with self.subTest(change=change), self.assertRaises(ValueError): replace(slot(), **change)

    def test_record_only_stores_an_answer_when_explicitly_answered(self):
        for kwargs in ({"status":"completed"}, {"answer":"明天"}, {"status":"declined", "answer":"不想说"},
                       {"status":"answered"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): QuestionRecord(slot(), **kwargs)
        self.assertEqual(QuestionRecord(slot(), "answered", "明天").answer, "明天")

    def test_config_rejects_nonfinite_bool_negative_and_fractional_counts(self):
        for value in (True, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError): InquiryConfig(initiative=value)
        for name in ("max_per_topic", "max_total", "cooldown_turns"):
            for value in (True, -1, .5, "2"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError): InquiryConfig(**{name:value})
        self.assertEqual(InquiryConfig(max_per_topic=0, max_total=0, cooldown_turns=0).max_total, 0)

    def test_choice_only_proposes_an_id_without_permission_or_state_authority(self):
        self.assertEqual(QuestionChoice(None).slot_id, None)
        self.assertEqual(QuestionChoice("deadline").slot_id, "deadline")
        for value in ("", " ", True, 1, {"mode":"allow"}):
            with self.subTest(value=value), self.assertRaises(ValueError): QuestionChoice(value)

    def test_state_roundtrip_preserves_quotas_permissions_and_pending_status(self):
        state = apply_control(InquiryState(), parse_control("今晚可以追问", "work", NOW))
        state = delivered(state, slot())
        restored = InquiryState.from_dict(json.loads(json.dumps(asdict(state))))
        self.assertEqual(restored, state)
        self.assertEqual(restored.permission("work", NOW), "allow")
        self.assertEqual(selection(state=restored)["reason"], "pending_question")
        with self.assertRaises(ValueError): InquiryState(records=(state.records[0], state.records[0]))
        with self.assertRaises(ValueError): InquiryState.from_dict({**asdict(state), "reset_budget":True})
        with self.assertRaises(ValueError): InquiryState(revision=True)


class TestInquiryControls(unittest.TestCase):
    def test_only_supported_complete_commands_change_permission(self):
        for text in ("不要追问", "别再问了", "不必问我"):
            self.assertEqual(parse_control(text, "work", NOW)["mode"], "deny")
        for text in ("可以追问", "继续问吧"):
            self.assertEqual(parse_control(text, "work", NOW)["mode"], "allow")
        for text in ('他说“不要追问”', '"可以追问"', "如果可以追问", "不要追问？", "可以追问?",
                     "可以追问吗", "我没有说可以追问", "今晚不要追问，明天再说", "以后在工作上不要追问",
                     "可以追问；不要给建议", "下周不要追问", "？不要追问", "不要追问?!"):
            with self.subTest(text=text): self.assertIsNone(parse_control(text, "work", NOW))

    def test_temporal_words_never_authorize_global_scope(self):
        local = parse_control("以后可以追问", "work", NOW)
        global_event = parse_control("以后所有话题都不要追问", "work", NOW)
        self.assertEqual(local["scope"], "work")
        self.assertEqual(global_event["scope"], "*")
        state = apply_control(apply_control(InquiryState(), global_event), local)
        self.assertEqual(state.permission("work"), "allow")
        self.assertEqual(state.permission("social"), "deny")

    def test_midnight_expiry_uses_utc8_and_falls_back_to_older_applicable_permission(self):
        state = apply_control(InquiryState(), parse_control("以后可以追问", "work", NOW))
        event = parse_control("今晚不要追问", "work", "2026-11-07T13:00:00+00:00")
        self.assertEqual(event["expires_at"], "2026-11-08T00:00:00+08:00")
        state = apply_control(state, event)
        self.assertEqual(state.permission("work", NOW), "deny")
        self.assertEqual(state.permission("work", "2026-11-07T16:00:00+00:00"), "allow")
        self.assertEqual(state.permission("social"), "unspecified")
        with self.assertRaises(ValueError): state.permission("work")
        with self.assertRaises(ValueError): InquiryPermission("work", "deny", "2026-11-08T00:00:00")

    def test_current_turn_event_is_explicit_and_does_not_mutate_persistent_state(self):
        persisted = InquiryState()
        event = parse_control("这轮不要追问", "work", NOW)
        self.assertTrue(event["transient"])
        transient = apply_control(persisted, event)
        self.assertEqual(selection(state=transient)["eligible_ids"], [])
        self.assertEqual(selection(state=persisted)["selected_id"], "deadline")
        self.assertEqual(persisted.revision, 0)
        with self.assertRaises(ValueError): apply_control(persisted, {**event, "reset_budget":True})


class TestQuestionAuthority(unittest.TestCase):
    def test_selection_does_not_consume_delivery_quota_or_change_state(self):
        state = InquiryState()
        first = selection(state=state)
        self.assertEqual(first["selected_id"], "deadline")
        self.assertEqual(first, selection(state=state))
        self.assertEqual(state, InquiryState())
        after = delivered(state, slot())
        self.assertEqual(after.revision, 1)
        self.assertEqual(selection(state=after)["reason"], "pending_question")
        self.assertEqual(state, InquiryState())

    def test_delivery_rejects_duplicate_id_even_if_same_or_reworded(self):
        state = delivered(InquiryState(), slot())
        for item in (slot(), replace(slot(), question="你会立刻辞职吗？"), slot(topic="social")):
            with self.subTest(item=item), self.assertRaises(ValueError): delivered(state, item)
        with self.assertRaises(ValueError): selection([replace(slot(), purpose="换一个用途")], state=state)

    def test_answer_requires_direct_evidence_pending_id_and_same_topic(self):
        state = delivered(InquiryState(), slot())
        for user, quote in (("不知道", "明天"), ('朋友说“明天”', "明天"), ("假如是明天", "明天")):
            with self.subTest(user=user), self.assertRaises(ValueError):
                resolve(state, "deadline", "work", "answer", user, quote)
        for id, topic, op in (("other", "work", "answer"), ("deadline", "social", "answer"), ("deadline", "work", "reset")):
            with self.subTest(id=id, topic=topic, op=op), self.assertRaises(ValueError):
                resolve(state, id, topic, op, "明天下午三点", "明天下午三点")
        after = resolve(state, "deadline", "work", "answer", "需要明天下午三点交", "明天下午三点")
        self.assertEqual(after.records[0].answer, "明天下午三点")
        self.assertEqual(after.records[0].status, "answered")
        self.assertEqual(state.records[0].status, "pending")
        with self.assertRaises(ValueError): resolve(after, "deadline", "work", "answer", "后天", "后天")

    def test_answer_and_decline_never_refund_quota_or_reask_same_id(self):
        for operation, user in (("answer", "明天"), ("decline", "不想回答")):
            with self.subTest(operation=operation):
                state = resolve(delivered(InquiryState(), slot()), "deadline", "work", operation, user, user)
                candidate = slot("format")
                one = selection([candidate], state=state, inquiry=InquiryConfig(initiative=1, max_per_topic=1), elapsed=99)
                self.assertEqual(one["reason"], "topic_question_limit")
                two = selection([slot(), candidate], state=state, elapsed=99)
                self.assertEqual(two["eligible_ids"], ["format"])
                self.assertEqual(two["selected_id"], "format")
                self.assertEqual(state.records[0].answer, user if operation == "answer" else "")

    def test_silence_and_unrelated_text_do_not_resolve_a_pending_question(self):
        state = delivered(InquiryState(), slot())
        for key in ("2", "3", "100"):
            choice = selection([slot("other")], state=state, key=key, elapsed=100)
            self.assertEqual(choice["reason"], "pending_question")
            self.assertEqual(choice["eligible_ids"], [])
        self.assertEqual(state.records[0].status, "pending")


class TestInquiryChoice(unittest.TestCase):
    def test_action_and_listening_priority_cannot_be_overridden_by_initiative(self):
        for request in ("listen", "answer", "advice", "resume"):
            with self.subTest(request=request):
                self.assertEqual(selection(request=request)["eligible_ids"], [])
                self.assertIsNone(selection(request=request)["selected_id"])
        choice = selection(policy=PolicyConfig(intervention=1))
        self.assertEqual(choice["reason"], "base_action_not_open_response")

    def test_question_permission_is_distinct_from_advice_permission(self):
        state = apply_control(InquiryState(), parse_control("不要追问", "work", NOW))
        self.assertEqual(selection(state=state)["reason"], "inquiry_denied")
        self.assertEqual(selection(state=InquiryState())["selected_id"], "deadline")
        # An allow event changes permission only; it cannot clear a pending question.
        after = apply_control(delivered(state, slot()), parse_control("可以追问", "work", NOW))
        self.assertEqual(selection([slot("format")], state=after, elapsed=100)["reason"], "pending_question")

    def test_global_and_topic_limits_include_resolved_questions_across_restart(self):
        state = delivered(InquiryState(), slot(topic="other"))
        state = resolve(state, "deadline", "other", "answer", "明天", "明天")
        state = InquiryState.from_dict(json.loads(json.dumps(asdict(state))))
        result = selection([slot("next")], state=state, inquiry=InquiryConfig(initiative=1, max_total=1))
        self.assertEqual(result["reason"], "total_question_limit")
        self.assertEqual(result["eligible_ids"], [])
        for kwargs in ({"max_total":0}, {"max_per_topic":0}):
            self.assertIsNone(selection(inquiry=InquiryConfig(initiative=1, **kwargs))["selected_id"])

    def test_cooldown_counts_confirmed_topic_turns_and_is_not_bypassed_by_other_topics(self):
        inquiry = InquiryConfig(initiative=1, cooldown_turns=2)
        self.assertEqual(selection(inquiry=inquiry, elapsed=None)["selected_id"], "deadline")
        for elapsed in (0, 1):
            self.assertEqual(selection(inquiry=inquiry, elapsed=elapsed)["reason"], "question_cooldown")
        self.assertEqual(selection(inquiry=inquiry, elapsed=2)["selected_id"], "deadline")
        for elapsed in (-1, True, .5):
            with self.assertRaises(ValueError): selection(elapsed=elapsed)

    def test_personality_selects_information_kind_and_ties_preserve_reviewed_order(self):
        tasks = [slot("detail", kind="detail"), slot("deadline"), slot("feeling", kind="feeling")]
        for expected, policy in (("deadline", PolicyConfig(intervention=0, practical=1, feeling=0, detail=0)),
                                 ("feeling", PolicyConfig(intervention=0, practical=0, feeling=1, detail=0)),
                                 ("detail", PolicyConfig(intervention=0, practical=.5, feeling=.5, detail=.5))):
            with self.subTest(expected=expected):
                self.assertEqual(selection(tasks, policy=policy)["selected_id"], expected)
        self.assertEqual(tasks[0].id, "detail")

    def test_initiative_draw_is_stable_and_does_not_erase_eligible_candidates(self):
        for key in map(str, range(40)):
            high = selection(key=key, inquiry=InquiryConfig(initiative=1))
            low = selection(key=key, inquiry=InquiryConfig(initiative=0))
            self.assertEqual(high["selected_id"], "deadline")
            self.assertIsNone(low["selected_id"])
            self.assertEqual(high["eligible_ids"], low["eligible_ids"])
            self.assertEqual(high["draw"], low["draw"])
            self.assertGreaterEqual(high["draw"], 0)
            self.assertLess(high["draw"], 1)
            self.assertEqual(high, selection(key=key, inquiry=InquiryConfig(initiative=1)))

    def test_only_same_topic_unasked_slots_are_candidates(self):
        result = selection([slot("wrong", topic="other"), slot("correct")])
        self.assertEqual(result["eligible_ids"], ["correct"])
        self.assertEqual(result["selected_id"], "correct")
        self.assertEqual(selection([])["reason"], "no_unasked_question_slot")

    def test_selection_requires_explicit_clock_for_expiring_state(self):
        state = apply_control(InquiryState(), parse_control("今晚不要追问", "work", NOW))
        with self.assertRaises(ValueError): selection(state=state, at=None)
        self.assertEqual(selection(state=state)["reason"], "inquiry_denied")
        next_day = selection(state=state, at="2026-11-08T00:00:00+08:00")
        self.assertEqual(next_day["selected_id"], "deadline")


if __name__ == "__main__":
    unittest.main()
