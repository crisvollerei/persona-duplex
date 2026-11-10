"""Memory equivalence must preserve changed content and source authority."""
import json
from pathlib import Path
import tempfile
import unittest

from persona.config import PersonaConfig
from persona.consolidate import Consolidator
from persona.llm import make_reply
from persona.memory_text import equivalent_memory_text, memory_text_key
from persona.session import Session, SessionConfig
from persona.store import Store, rebuild_views

ROOT = Path(__file__).resolve().parent.parent


class MemoryTextTests(unittest.TestCase):
    def test_typography_normalization_is_limited(self):
        self.assertTrue(equivalent_memory_text(
            " 用户 希望专注25 分钟，结束后休息。 ", "用户希望专注25分钟,结束后休息"))
        for left, right in (
                ("用户专注25分钟", "用户专注35分钟"),
                ("用户使用ReaderPro", "用户使用ReaderLite"),
                ("用户喜欢喝茶", "用户喜欢喝茶但只在周末喝茶"),
                ("用户喜欢喝茶", "用户不喜欢喝茶"),
                ("用户专注1小时", "用户专注1分钟"),
                ("版本1.5", "版本15"),
                ("语言C++", "语言C"),
                ("名称foo-bar", "名称foobar"),
                ("名称Reader Pro", "名称ReaderPro"),
                ("编号1 2", "编号12"),
                ("温度-5度", "温度5度"),
                ("日期2026-11-10", "日期2026-01-11"),
                ('文件名“我的 草稿”', '文件名“我的草稿”')):
            with self.subTest(left=left, right=right):
                self.assertFalse(equivalent_memory_text(left, right))
        self.assertFalse(equivalent_memory_text("", ""))

    def test_equivalence_does_not_claim_semantic_paraphrase(self):
        self.assertFalse(equivalent_memory_text("用户在意别人的评价", "用户重视别人对自己的看法"))
        self.assertNotEqual(memory_text_key("设置ID:AB"), memory_text_key("设置ID:ab"))


class MemoryEquivalenceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="persona-memory-equivalence-")
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name))

    def seed(self, text, quote=None, *, source="explicit", **metadata):
        quote = quote or "我明确采用：" + text
        self.store.append("turn", {"turn_id": "u_old", "user": quote})
        return self.store.add_judgment(text, .95, ["u_old"], source=source,
            evidence_quotes=[{"turn_id": "u_old", "quote": quote, "evidence_scope": "specified"}], **metadata)

    def run_batch(self, item, user, *, turn_id="u_new", memory_scope=None):
        self.store.append("turn", {"turn_id": turn_id, "user": user, "memory_scope": memory_scope})
        self.store.push_buffer({"turn_id": turn_id, "user": user, "need": "只想被听",
                                "memory_scope": memory_scope})
        calls = []
        def model(prompt):
            calls.append(prompt)
            return json.dumps([item], ensure_ascii=False) if len(calls) == 1 else "本次记录仅依据用户当前原话。"
        result = Consolidator(self.store, llm=model, memory_scope=memory_scope).run_once()
        return result, calls

    def test_changed_number_survives_both_dedup_stages_and_reopen(self):
        old = self.seed("用户希望每次专注25分钟", "我明确采用每次专注25分钟。")
        user = "我明确更正，现在每次专注35分钟，旧25分钟作废。"
        result, calls = self.run_batch({"judgment": "用户希望每次专注35分钟", "confidence": .4,
                                      "evidence_ids": ["u_new"]}, user)
        self.assertEqual(result["judgments"], 1)
        self.assertEqual(len(calls), 2)
        records = self.store.judgments(limit=None)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], old["id"])
        self.assertEqual(records[0]["evidence_quotes"], old["evidence_quotes"])
        self.assertEqual(records[1]["evidence_quotes"][0]["quote"], user)
        self.assertIsNone(records[1]["supersedes"])
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])
        self.assertEqual(Store(self.store.root).judgments(limit=None), records)
        rebuild_views(self.store)
        self.assertEqual(self.store.judgments(limit=None), records)

    def test_names_conditions_negation_and_units_do_not_support_old_text(self):
        pairs = (
            ("用户偏好使用ReaderPro作为阅读器", "用户偏好使用ReaderLite作为阅读器"),
            ("用户喜欢喝茶", "用户喜欢喝茶但只在周末喝茶"),
            ("用户喜欢咖啡", "用户不喜欢咖啡"),
            ("用户每次练习1小时", "用户每次练习1分钟"),
        )
        for index, (before, after) in enumerate(pairs):
            with self.subTest(before=before):
                store = Store(Path(self.temp.name) / str(index))
                old_store = self.store
                self.store = store
                try:
                    old = self.seed(before)
                    result, _ = self.run_batch({"judgment": after, "confidence": .7,
                                               "evidence_ids": ["u_new"]}, "我明确采用：" + after)
                    rows = store.judgments(limit=None)
                    self.assertEqual(result["judgments"], 1)
                    self.assertEqual([row["judgment"] for row in rows], [before, after])
                    self.assertEqual(rows[0]["evidence_quotes"], old["evidence_quotes"])
                    self.assertIsNone(rows[1]["supersedes"])
                finally:
                    self.store = old_store

    def test_typographic_duplicate_only_refreshes_sources(self):
        old = self.seed("用户希望每次专注25分钟，结束后休息。")
        item = {"judgment": " 用户 希望每次专注25 分钟,结束后休息 ", "confidence": .2,
                "evidence_ids": ["u_new"]}
        result, calls = self.run_batch(item, "我继续采用每次专注25分钟，结束后休息。")
        self.assertEqual(result["judgments"], 0)
        self.assertFalse(result["self_story"])
        self.assertEqual(len(calls), 1)
        current, = Store(self.store.root).judgments(limit=None)
        self.assertEqual(current["id"], old["id"])
        self.assertEqual(current["confidence"], .95)
        self.assertEqual(len(current["evidence_quotes"]), 2)

    def test_unsupported_model_proposal_is_stored_without_story_or_foreground(self):
        item = {"judgment": "用户偏好ReaderLite", "confidence": .8, "evidence_ids": []}
        result, calls = self.run_batch(item, "我今天只是路过。")
        self.assertEqual(result["judgments"], 1)
        self.assertFalse(result["self_story"])
        self.assertEqual(len(calls), 1)
        row, = self.store.judgments(limit=None)
        self.assertEqual(row["evidence_scope"], "unsupported")
        self.assertEqual(row["evidence_ids"], [])
        self.assertNotIn("evidence_quotes", row)
        self.assertEqual(self.store.user_judgments(limit=None), [])

    def test_supported_candidate_does_not_merge_into_unsupported_record(self):
        item = {"judgment": "用户偏好ReaderLite", "confidence": .8, "evidence_ids": []}
        self.run_batch(item, "我只是路过。")
        result, _ = self.run_batch({**item, "evidence_ids": ["u_confirm"]},
                                  "我明确偏好ReaderLite作为阅读器。", turn_id="u_confirm")
        self.assertEqual(result["judgments"], 1)
        rows = self.store.user_judgments(limit=None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evidence_scope"], "specified")
        self.assertEqual(len(self.store.judgments(limit=None)), 2)

    def test_unsupported_candidate_cannot_borrow_application_authority(self):
        old = self.store.add_judgment("用户偏好ReaderLite", .9)
        self.run_batch({"judgment": old["judgment"], "confidence": .8, "evidence_ids": []}, "只是路过。")
        self.assertEqual([row["id"] for row in self.store.user_judgments(limit=None)], [old["id"]])
        self.assertEqual(len(self.store.judgments(limit=None)), 2)

    def test_unscoped_candidate_does_not_refresh_restricted_memory(self):
        for name, value in (("memory_scope", "work"),
                            ("valid_from", "2026-01-01T00:00:00+08:00"),
                            ("expires_at", "2030-01-01T00:00:00+08:00")):
            with self.subTest(name=name):
                store = Store(Path(self.temp.name) / name)
                previous = self.store
                self.store = store
                try:
                    old = store.add_judgment("用户偏好ReaderLite", .9, **{name: value})
                    result, _ = self.run_batch({"judgment": old["judgment"], "confidence": .6,
                                               "evidence_ids": ["u_new"]}, "我明确采用ReaderLite。")
                    self.assertEqual(result["judgments"], 1)
                    rows = store.judgments(limit=None)
                    self.assertEqual(len(rows), 2)
                    self.assertNotIn("evidence_quotes", rows[0])
                    self.assertEqual(rows[0][name], value)
                    self.assertNotIn(name, rows[1])
                finally:
                    self.store = previous

    def test_same_scope_duplicate_keeps_scope_and_refreshes_sources(self):
        old = self.seed("用户偏好ReaderLite", memory_scope="work")
        result, calls = self.run_batch({"judgment": old["judgment"], "confidence": .2,
                                       "evidence_ids": ["u_new"]},
                                      "我继续使用ReaderLite。", memory_scope="work")
        self.assertEqual(result["judgments"], 0)
        self.assertEqual(len(calls), 1)
        current, = self.store.judgments(limit=None)
        self.assertEqual(current["id"], old["id"])
        self.assertEqual(current["memory_scope"], "work")
        self.assertEqual(current["confidence"], .95)
        self.assertEqual(len(current["evidence_quotes"]), 2)

    def test_scoped_consolidation_reaches_frontend_and_survives_restart(self):
        user = "我明确偏好ReaderLite作为工作阅读器。"
        result, _ = self.run_batch({"judgment": "用户工作阅读器为ReaderLite", "confidence": .7,
                                   "evidence_ids": ["u_new"], "memory_scope": "forged_scope"},
                                  user, memory_scope="work")
        self.assertEqual(result["judgments"], 1)
        record, = self.store.judgments(limit=None)
        self.assertEqual(record["memory_scope"], "work")
        reopened = Store(self.store.root)
        self.assertEqual(reopened.recall_judgments("工作阅读器"), [])
        self.assertEqual([r["id"] for r in reopened.recall_judgments("工作阅读器", scope="work")], [record["id"]])
        seen = []
        def model(messages):
            seen.append(messages)
            return make_reply("need: 只是想说说话\nconfidence: 0.8\ndirect: ReaderLite\nbaseline: ReaderLite\ndisplace: none",
                              "ReaderLite", "settle: -\ndelta: -\npending: -")
        cfg = PersonaConfig.load(ROOT / "personas/default.toml")
        with Session(cfg, reopened, model, options=SessionConfig(memory_mode="frozen", memory_scope="work")) as session:
            self.assertEqual(session.say("我的工作阅读器是什么？").say, "ReaderLite")
        self.assertEqual(len(seen), 1)
        self.assertIn(user, str(seen[0]))

    def test_unsupported_scoped_proposal_stays_out_of_recall(self):
        result, calls = self.run_batch({"judgment": "用户工作阅读器为ReaderLite", "confidence": .7,
                                       "evidence_ids": []}, "我只是路过。", memory_scope="work")
        self.assertEqual(result["judgments"], 1)
        self.assertEqual(len(calls), 1)
        record, = self.store.judgments(limit=None)
        self.assertEqual(record["memory_scope"], "work")
        self.assertEqual(record["evidence_scope"], "unsupported")
        self.assertEqual(self.store.recall_judgments("工作阅读器", scope="work"), [])

    def test_session_scope_changes_preserve_original_work_backlog(self):
        cfg = PersonaConfig.load(ROOT / "personas/default.toml")
        user = "我明确采用ReaderLite作为工作阅读器。"
        def frontend(messages):
            return make_reply("need: 只想被听\nconfidence: 0.8\ndirect: 好\nbaseline: 好\ndisplace: none",
                              "好", "settle: -\ndelta: -\npending: -")
        with Session(cfg, self.store, frontend, options=SessionConfig(
                memory_scope="work", consolidate_on_exit=False)) as session:
            turn = session.say(user)
        queued, = [json.loads(path.read_text(encoding="utf-8"))
                   for path in self.store.buffer_dir.glob("*.json")]
        self.assertEqual(queued["memory_scope"], "work")
        calls = []
        def memory(prompt):
            calls.append(prompt)
            return (json.dumps([{"judgment": "用户工作阅读器为ReaderLite", "confidence": .7,
                                 "evidence_ids": [turn.turn_id]}], ensure_ascii=False)
                    if len(calls) == 1 else "本次记录仅依据用户原话。")
        self.store = Store(self.store.root)
        with Session(cfg, self.store, frontend, consolidate_llm=memory,
                options=SessionConfig(memory_scope="personal", consolidate_on_exit=False)) as session:
            self.assertEqual(session.consolidate_now()["reason"], "no events")
        self.assertEqual(calls, [])
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))), 1)
        self.assertEqual(self.store.judgments(limit=None), [])
        with Session(cfg, self.store, frontend, consolidate_llm=memory,
                options=SessionConfig(memory_scope="work", consolidate_on_exit=False)) as session:
            self.assertEqual(session.consolidate_now()["judgments"], 1)
        record, = Store(self.store.root).judgments(limit=None)
        self.assertEqual(record["memory_scope"], "work")
        self.assertEqual(record["evidence_ids"], [turn.turn_id])
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])

    def test_mixed_scopes_and_legacy_batches_keep_identity_and_evidence(self):
        for scope, count in (("work", 12), ("personal", 3), (None, 1)):
            for index in range(count):
                turn_id = f"{scope}_{index}"
                user = f"我在{scope or '默认'}场景采用设置{index}。"
                self.store.append("turn", {"turn_id": turn_id, "user": user})
                event = {"turn_id": turn_id, "user": user, "need": "只想被听"}
                if scope is not None:
                    event["memory_scope"] = scope
                self.store.push_buffer(event)
        consumed = []
        for scope, activity, expected_count in (("personal", "medium", 3), (None, "sparse", 1),
                                                 ("work", "dense", 12)):
            calls = []
            def memory(prompt):
                calls.append(prompt)
                return "[]"
            consolidator = Consolidator(self.store, llm=memory, memory_scope=scope)
            self.assertEqual(consolidator._pending_count(), expected_count)
            result = consolidator.run_once()
            self.assertEqual(result["activity"], activity)
            batch = [row for row in self.store.read_ledger() if row["kind"] == "memory_batch_commit"][-1]
            self.assertEqual(len(batch["events"]), expected_count)
            self.assertTrue(all(event.get("memory_scope") == scope for event in batch["events"]))
            consumed.extend(batch["event_ids"])
            self.assertEqual(len(calls), 1)
            self.store = Store(self.store.root)
        self.assertEqual(len(consumed), len(set(consumed)))
        self.assertEqual(len(consumed), 16)
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")), [])
        self.assertEqual(list(self.store.processing_dir.glob("*.json")), [])

    def test_invalid_consolidation_scope_is_rejected(self):
        for scope in ("", " ", 1, True):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                Consolidator(self.store, memory_scope=scope)

    def test_scoped_memory_operation_cannot_rewrite_global_background(self):
        old = self.seed("用户默认每次专注25分钟", "我明确采用每次专注25分钟。")
        user = "我明确更正工作中的专注设置，现在改为35分钟，其他场景不变。"
        item = {"judgment": "用户每次专注35分钟", "confidence": .7, "evidence_ids": ["u_new"],
                "memory_operation": {"kind": "revise", "target_id": old["id"], "scope": "user_preference",
                    "basis": "explicit_user_statement", "evidence_ids": ["u_new"],
                    "statement": user, "target_anchor": "专注"}}
        result, calls = self.run_batch(item, user, memory_scope="work")
        self.assertFalse(result["memory_operations"][0]["effective"])
        self.assertTrue(result["memory_operations"][0]["reasons"])
        self.assertEqual(len(calls), 1)
        self.assertEqual([j["id"] for j in self.store.recall_judgments("专注", None)], [old["id"]])
        prompt = calls[0]
        self.assertIn("仅背景，不可作为更正/撤销目标", prompt)
        self.assertIn('"memory_scope": null', prompt)

    def test_global_memory_operation_cannot_guess_a_hidden_scoped_target(self):
        old = self.seed("用户工作专注25分钟", "我明确采用工作专注25分钟。", memory_scope="work")
        user = "我明确更正专注设置，现在改为35分钟。"
        item = {"judgment": "用户专注35分钟", "confidence": .7, "evidence_ids": ["u_new"],
                "memory_operation": {"kind": "revise", "target_id": old["id"], "scope": "user_preference",
                    "basis": "explicit_user_statement", "evidence_ids": ["u_new"],
                    "statement": user, "target_anchor": "专注"}}
        result, _ = self.run_batch(item, user)
        self.assertFalse(result["memory_operations"][0]["effective"])
        self.assertEqual([j["id"] for j in self.store.recall_judgments("专注", None, scope="work")], [old["id"]])

    def test_same_scope_revision_inherits_application_scope(self):
        old = self.seed("用户工作专注25分钟", "我明确采用工作专注25分钟。", memory_scope="work")
        user = "我明确更正工作专注设置，现在改为35分钟。"
        item = {"judgment": "用户工作专注35分钟", "confidence": .2, "evidence_ids": ["u_new"],
                "memory_operation": {"kind": "revise", "target_id": old["id"], "scope": "user_preference",
                    "basis": "explicit_user_statement", "evidence_ids": ["u_new"],
                    "statement": user, "target_anchor": "专注"}}
        result, _ = self.run_batch(item, user, memory_scope="work")
        self.assertTrue(result["memory_operations"][0]["effective"])
        self.assertEqual(self.store.recall_judgments("专注", None), [])
        active, = Store(self.store.root).recall_judgments("专注", None, scope="work")
        self.assertEqual(active["memory_scope"], "work")
        self.assertEqual(active["judgment"], item["judgment"])

    def test_explicit_revision_and_retraction_remain_effective_after_restart(self):
        old = self.seed("用户每次专注25分钟", "我明确采用每次专注25分钟。")
        user = "我明确更正专注设置，现在改为35分钟，旧25分钟作废。"
        operation = {"kind": "revise", "target_id": old["id"], "scope": "user_preference",
                     "basis": "explicit_user_statement", "statement": user,
                     "target_anchor": "专注", "evidence_ids": ["u_new"]}
        result, calls = self.run_batch({"judgment": "用户每次专注35分钟", "confidence": .2,
                                       "evidence_ids": ["u_new"], "memory_operation": operation}, user)
        self.assertTrue(result["memory_operations"][0]["effective"])
        self.assertEqual(len(calls), 1)
        self.store = Store(self.store.root)
        current, = self.store.user_judgments(limit=None)
        self.assertEqual(current["judgment"], "用户每次专注35分钟")
        self.assertEqual(current["confidence"], .2)
        self.assertEqual(current["supersedes"], old["id"])
        user = "我明确撤销专注设置，新设置尚未决定。"
        operation.update(kind="retract", target_id=current["id"], statement=user, evidence_ids=["u_cancel"])
        result, calls = self.run_batch({"memory_operation": operation, "evidence_ids": ["u_cancel"]},
                                      user, turn_id="u_cancel")
        self.assertTrue(result["memory_operations"][0]["effective"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(Store(self.store.root).user_judgments(limit=None), [])

    def test_consolidation_retrieves_an_older_relevant_operation_target(self):
        old = self.seed("用户每次专注25分钟", "我明确采用每次专注25分钟。")
        for index in range(50):
            self.store.add_judgment(f"用户的第{index}项园艺习惯", .7)
        event = {"turn_id": "u_new", "user": "我明确更正专注设置，现在改为35分钟。"}
        prompt = Consolidator(self.store)._consolidation_prompt([event])
        historical = prompt.split("本批当前用户事件", 1)[0]
        self.assertIn("[" + old["id"] + "]", historical)
        self.assertIn(old["evidence_quotes"][0]["quote"], historical)
        self.assertIn("当前证据ID清单以外", prompt)

    def test_background_target_budget_can_retain_a_long_original_source(self):
        user = "我明确采用每次专注25分钟。" + "这是我之前给出的相关说明。" * 500
        old = self.seed("用户每次专注25分钟", user)
        event = {"turn_id": "u_new", "user": "我明确更正专注设置，现在改为35分钟。"}
        prompt = Consolidator(self.store)._consolidation_prompt([event])
        historical = prompt.split("本批当前用户事件", 1)[0]
        self.assertIn("[" + old["id"] + "]", historical)
        self.assertIn(user, historical)


if __name__ == "__main__":
    unittest.main()
