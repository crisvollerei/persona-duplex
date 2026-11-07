"""Evidence-bound preference changes are distinct from model confidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from persona.consolidate import Consolidator
from persona.store import Store, rebuild_views


class MemoryOperationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name))
        old_user = "我明确采用的固定专注时长为19分钟。"
        self.store.append("turn", {"turn_id":"u_old", "user":old_user})
        self.old = self.store.add_judgment("用户自述固定专注19分钟。", .95, ["u_old"],
            evidence_quotes=[{"turn_id":"u_old", "quote":old_user, "evidence_scope":"specified"}])
        self.user = "我明确更正个人设置，现在改为专注23分钟，旧19分钟作废。"
        self.store.append("turn", {"turn_id": "u_new", "user": self.user})

    def proposal(self, kind="revise", **changes):
        return {"kind": kind, "target_id": self.old["id"], "scope": "user_preference",
                "basis": "explicit_user_statement", "evidence_ids": ["u_new"],
                "statement": self.user, "target_anchor":"专注", **changes}

    def apply(self, operation=None, **changes):
        arguments = {"judgment": "用户自述现在专注23分钟。", "confidence": .2,
                     "evidence_ids": ["u_new"], "evidence_quotes": [
                         {"turn_id": "u_new", "quote": self.user, "evidence_scope": "specified"}]}
        arguments.update(changes)
        return self.store.apply_memory_operation(operation or self.proposal(), **arguments)

    def replace_user(self, text):
        self.user = text
        self.store.append("turn", {"turn_id": "u_updated", "user": text})
        return self.proposal(evidence_ids=["u_updated"], statement=text)

    def test_verified_revision_does_not_inflate_confidence(self):
        result = self.apply()
        self.assertTrue(result["effective"])
        self.assertEqual(result["replacement"]["confidence"], .2)
        active, = self.store.user_judgments(limit=None)
        self.assertEqual(active["id"], result["replacement_id"])
        self.assertEqual(active["supersedes"], self.old["id"])
        self.assertEqual(result["semantic_status"], "model_proposal")

    def test_legacy_direct_api_keeps_compatibility(self):
        updated = self.store.add_judgment("旧兼容调用的低把握新推断", .2, supersedes=self.old["id"])
        self.assertIsNone(updated["supersedes"])
        self.assertEqual(len(self.store.user_judgments(limit=None)), 2)

    def test_retract_creates_no_replacement_preference(self):
        proposal = self.replace_user("我的旧专注设置现在作废，新设置尚未决定。")
        proposal["kind"] = "retract"
        result = self.apply(proposal, judgment="", evidence_ids=["u_updated"], evidence_quotes=[
            {"turn_id":"u_updated", "quote":self.user, "evidence_scope":"specified"}])
        self.assertTrue(result["effective"])
        self.assertIsNone(result["replacement"])
        self.assertEqual(self.store.user_judgments(limit=None), [])
        self.assertEqual(sum(row["kind"] == "judgment_add" for row in self.store.read_ledger()), 1)

    def test_retraction_and_revision_survive_actual_rebuild_and_reopen(self):
        result = self.apply()
        expected = self.store.user_judgments(limit=None)
        self.store.save_view("judgments", {"items": []})
        rebuild_views(self.store)
        self.assertEqual(self.store.load_view("judgments")["items"], expected)
        reopened = Store(self.store.root)
        self.assertEqual(reopened.user_judgments(limit=None), expected)
        self.assertEqual(result["replacement_id"], expected[0]["id"])

    def test_basis_flag_without_user_source_cannot_authorize_change(self):
        result = self.apply(evidence_ids=["missing"], evidence_quotes=[
            {"turn_id":"missing", "quote":self.user,"evidence_scope":"specified"}],
            operation=self.proposal(evidence_ids=["missing"]))
        self.assertFalse(result["effective"])
        self.assertIn("source_user_turn_not_found", result["reasons"])
        self.assertEqual([row["id"] for row in self.store.user_judgments()], [self.old["id"]])

    def test_model_text_or_shortened_quote_cannot_become_user_evidence(self):
        for quote in ("我喜欢23分钟。", "现在改为专注23分钟"):
            with self.subTest(quote=quote):
                result = self.apply(evidence_quotes=[
                    {"turn_id":"u_new","quote":quote,"evidence_scope":"specified"}])
                self.assertFalse(result["effective"])
                self.assertIn("quote_not_full_original_user_message", result["reasons"])

    def test_correct_quote_cannot_hide_a_second_fabricated_quote_for_same_id(self):
        result = self.apply(evidence_quotes=[
            {"turn_id":"u_new","quote":"伪造的用户信息","evidence_scope":"specified"},
            {"turn_id":"u_new","quote":self.user,"evidence_scope":"specified"}])
        self.assertFalse(result["effective"])
        self.assertIn("ambiguous_operation_quotes", result["reasons"])

    def test_extra_candidate_reference_must_also_have_its_real_user_source(self):
        result = self.apply(evidence_ids=["u_new", "invented_extra"], evidence_quotes=[
            {"turn_id":"u_new","quote":self.user,"evidence_scope":"specified"},
            {"turn_id":"invented_extra","quote":"未发生的用户陈述","evidence_scope":"specified"}])
        self.assertFalse(result["effective"])
        self.assertIn("source_user_turn_not_found",result["reasons"])

    def test_evidence_before_old_memory_is_not_a_current_correction(self):
        other = Store(Path(self.temp.name)/"older")
        other.append("turn", {"turn_id":"u_new","user":self.user})
        old = other.add_judgment("后来记录的19分钟偏好", .95)
        result = other.apply_memory_operation(self.proposal(target_id=old["id"]), judgment="23分钟",confidence=.2,
            evidence_ids=["u_new"], evidence_quotes=[{"turn_id":"u_new","quote":self.user,"evidence_scope":"specified"}])
        self.assertFalse(result["effective"])
        self.assertIn("evidence_not_after_target",result["reasons"])

    def test_rejection_never_appends_replacement_judgment(self):
        result = self.apply(self.proposal(basis="model_inference"), confidence=.99)
        self.assertFalse(result["effective"])
        self.assertEqual(len(self.store.user_judgments()),1)
        operation, = [r for r in self.store.read_ledger() if r["kind"]=="judgment_operation"]
        self.assertFalse(operation["effective"])
        self.assertIsNone(operation["replacement_id"])

    def test_operation_scope_does_not_turn_self_reports_into_external_facts(self):
        result = self.apply(self.proposal(scope="external_fact"))
        self.assertFalse(result["effective"])
        self.assertIn("unsupported_scope",result["reasons"])

    def test_guesses_cannot_target_diagnostic_or_nonexistent_memory(self):
        diagnostic = self.store.add_judgment("重复模型读法",.9,source="heuristic",evidence_scope="need_count")
        for target in (diagnostic["id"],"j_missing"):
            with self.subTest(target=target):
                self.assertFalse(self.apply(self.proposal(target_id=target))["effective"])

    def test_quoted_hypothetical_third_party_and_tentative_updates_rejected(self):
        texts = [
            "朋友说：‘我明确更正，现在改为专注23分钟。’这不是我的设置。",
            "如果我改为专注23分钟，会不会更合适？",
            "我正在考虑改为专注23分钟，还没决定。",
            "我没有更正为23分钟，旧设置仍然有效。",
            "这里是示例字符串：我明确更正，现在改为23分钟。请不要当成我的偏好。",
        ]
        for text in texts:
            with self.subTest(text=text):
                operation = self.replace_user(text)
                result = self.apply(operation,evidence_ids=["u_updated"],evidence_quotes=[
                    {"turn_id":"u_updated","quote":text,"evidence_scope":"specified"}])
                self.assertFalse(result["effective"])
        self.assertEqual(len(self.store.user_judgments()),1)

    def test_real_self_update_may_contain_a_quoted_new_name(self):
        old_user = "我固定的启动文件名为draft_cedar。"
        self.store.append("turn", {"turn_id":"u_file_old", "user":old_user})
        target = self.store.add_judgment("用户自述启动文件名为draft_cedar。",.95,["u_file_old"],
            evidence_quotes=[{"turn_id":"u_file_old","quote":old_user,"evidence_scope":"specified"}])
        operation = self.replace_user("我明确更正启动文件名，现在改为“draft_iris”。")
        operation.update(target_id=target["id"], target_anchor="启动文件名")
        result = self.apply(operation,judgment="用户自述启动文件名为draft_iris。",evidence_ids=["u_updated"],
            evidence_quotes=[{"turn_id":"u_updated","quote":self.user,"evidence_scope":"specified"}])
        self.assertTrue(result["effective"])

    def test_duplicate_operation_cannot_replace_an_inactive_target_again(self):
        self.assertTrue(self.apply()["effective"])
        second = self.apply()
        self.assertFalse(second["effective"])
        self.assertIn("target_not_active",second["reasons"])
        self.assertEqual(len(self.store.user_judgments()),1)

    def test_failed_atomic_operation_does_not_leave_replacement_or_retraction(self):
        original = self.store.append
        def fail(kind, payload):
            if kind == "judgment_operation": raise OSError("simulated local commit failure")
            return original(kind,payload)
        with patch.object(self.store,"append",side_effect=fail):
            with self.assertRaises(OSError): self.apply()
        self.assertEqual([row["id"] for row in self.store.user_judgments()], [self.old["id"]])

    def test_consolidator_uses_existing_call_and_actual_batch_source(self):
        event = {"turn_id":"u_buffer","user":"我明确更正，现在改为专注23分钟。","need":"只想被听"}
        self.store.push_buffer(event)
        proposal = self.proposal(evidence_ids=["u_buffer"],statement=event["user"])
        calls=[]
        def model(prompt):
            calls.append(prompt)
            return json.dumps([{"judgment":"用户自述23分钟。","confidence":.1,"evidence_ids":["u_buffer"],
                                "memory_operation":proposal}],ensure_ascii=False)
        result=Consolidator(self.store,llm=model).run_once()
        self.assertEqual(len(calls),1)
        self.assertTrue(result["memory_operations"][0]["effective"])
        self.assertEqual(len(self.store.user_judgments()),1)
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")),[])

    def test_one_rejected_operation_does_not_discard_a_legal_batch(self):
        event={"turn_id":"u_buffer","user":"我明确采用每周例会这个长期约定。","need":"只想被听"}
        self.store.push_buffer(event)
        model=lambda prompt: json.dumps([
            {"memory_operation":{"kind":"retract","target_id":"missing"}},
            {"judgment":"用户自述有每周例会约定。","confidence":.8,"evidence_ids":["u_buffer"]}],ensure_ascii=False)
        result=Consolidator(self.store,llm=model).run_once()
        self.assertFalse(result["memory_operations"][0]["effective"])
        self.assertEqual(len(self.store.user_judgments()),2)
        self.assertEqual(list(self.store.buffer_dir.glob("*.json")),[])

    def test_model_legacy_supersedes_without_explicit_source_cannot_bypass_checks(self):
        self.store.push_buffer({"turn_id":"u_buffer","user":"今天只是有点困。","need":"只想被听"})
        model=lambda prompt: json.dumps([{"judgment":"用户改为23分钟。","confidence":.99,
            "supersedes":self.old["id"],"evidence_ids":["u_buffer"]}],ensure_ascii=False)
        result=Consolidator(self.store,llm=model).run_once()
        self.assertFalse(result["memory_operations"][0]["effective"])
        self.assertEqual([row["id"] for row in self.store.user_judgments()],[self.old["id"]])

    def test_legacy_source_backed_model_revision_uses_same_evidence_gate(self):
        self.store.push_buffer({"turn_id":"u_buffer","user":"我明确更正，现在改为专注23分钟。","need":"只想被听"})
        model=lambda prompt: json.dumps([{"judgment":"用户自述改为23分钟。","confidence":.1,
            "supersedes":self.old["id"],"target_anchor":"专注","evidence_ids":["u_buffer"]}],ensure_ascii=False)
        result=Consolidator(self.store,llm=model).run_once()
        operation, = result["memory_operations"]
        self.assertTrue(operation["effective"])
        self.assertEqual(operation["operation"]["evidence_origin"],"legacy_explicit_wording_adapter")
        self.assertEqual(len(self.store.user_judgments()),1)

    def test_unrelated_target_cannot_be_replaced_by_a_verified_new_topic(self):
        user = "我偏好的热饮是茉莉茶。"
        self.store.append("turn", {"turn_id":"u_drink", "user":user})
        drink = self.store.add_judgment("用户偏好茉莉茶。",.95,["u_drink"],
            evidence_quotes=[{"turn_id":"u_drink","quote":user,"evidence_scope":"specified"}])
        # A current source is necessary too: put it after the target record.
        self.store.append("turn", {"turn_id":"u_new_after_drink", "user":self.user})
        for confidence in (.2,.55,.95):
            with self.subTest(confidence=confidence):
                operation=self.proposal(target_id=drink["id"],evidence_ids=["u_new_after_drink"])
                result=self.apply(operation, confidence=confidence, evidence_ids=["u_new_after_drink"],
                    evidence_quotes=[{"turn_id":"u_new_after_drink","quote":self.user,"evidence_scope":"specified"}])
                self.assertFalse(result["effective"])
                self.assertIn("target_anchor_not_in_target_source",result["reasons"])
        self.assertIn(drink["id"],[item["id"] for item in self.store.user_judgments(limit=None)])

    def test_generic_or_missing_anchor_cannot_authorize_mutation(self):
        for anchor in (None,"我","个人设置","固定偏好","时长","19分钟",""):
            with self.subTest(anchor=anchor):
                result=self.apply(self.proposal(target_anchor=anchor))
                self.assertFalse(result["effective"])
                self.assertIn("missing_or_generic_target_anchor",result["reasons"])

    def test_anchor_must_appear_in_current_statement_not_only_old_source(self):
        self.assertFalse(self.apply(self.proposal(target_anchor="茉莉茶"))["effective"])

    def test_target_without_complete_original_user_quote_stays_active(self):
        target=self.store.add_judgment("用户旧的专注偏好19分钟。",.95)
        self.store.append("turn", {"turn_id":"u_late", "user":self.user})
        result=self.apply(self.proposal(target_id=target["id"],evidence_ids=["u_late"]),
            evidence_ids=["u_late"],evidence_quotes=[{"turn_id":"u_late","quote":self.user,"evidence_scope":"specified"}])
        self.assertFalse(result["effective"])
        self.assertIn("target_has_no_user_quote",result["reasons"])

    def test_forged_target_quote_does_not_establish_a_topic_match(self):
        self.store.append("turn", {"turn_id":"u_forged_old", "user":"我喜欢茉莉茶。"})
        target=self.store.add_judgment("用户旧专注偏好19分钟。",.95,["u_forged_old"],
            evidence_quotes=[{"turn_id":"u_forged_old","quote":"我固定专注19分钟。","evidence_scope":"specified"}])
        self.store.append("turn", {"turn_id":"u_late", "user":self.user})
        result=self.apply(self.proposal(target_id=target["id"],evidence_ids=["u_late"]),
            evidence_ids=["u_late"],evidence_quotes=[{"turn_id":"u_late","quote":self.user,"evidence_scope":"specified"}])
        self.assertFalse(result["effective"])
        self.assertIn("target_user_quote_not_verified",result["reasons"])

    def test_legacy_model_change_without_anchor_cannot_bypass_target_link_check(self):
        self.store.push_buffer({"turn_id":"u_buffer","user":"我明确更正，现在改为专注23分钟。","need":"只想被听"})
        model=lambda prompt: json.dumps([{"judgment":"用户自述改为23分钟。","confidence":.99,
            "supersedes":self.old["id"],"evidence_ids":["u_buffer"]}],ensure_ascii=False)
        result=Consolidator(self.store,llm=model).run_once()
        operation,=result["memory_operations"]
        self.assertFalse(operation["effective"])
        self.assertIn("missing_or_generic_target_anchor",operation["reasons"])

    def test_prompt_separates_current_evidence_ids_from_historical_target_ids(self):
        prompt=Consolidator(self.store)._consolidation_prompt([
            {"turn_id":"u_current","user":"我明确更正，现在改为专注23分钟。"}])
        self.assertIn('本批唯一允许引用的 evidence_ids',prompt)
        self.assertIn('["u_current"]',prompt)
        self.assertIn('历史原话ID不是本批证据',prompt)
        self.assertIn('此处turn_id不能引用为本批证据',prompt)
        self.assertIn('target_anchor',prompt)

    def test_invalid_historical_id_in_model_output_is_not_silently_rewritten(self):
        self.store.push_buffer({"turn_id":"u_current","user":"今天临时专注23分钟，明天恢复原习惯。"})
        model=lambda prompt: json.dumps([{"judgment":"用户仍采用19分钟。","confidence":.9,
            "evidence_ids":["u_old","u_current"]}],ensure_ascii=False)
        with self.assertRaisesRegex(ValueError,"Unknown memory evidence id"):
            Consolidator(self.store,llm=model).run_once()
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))),1)

    def test_complete_string_quotes_normalize_by_exact_source_not_list_position(self):
        events=[{"turn_id":"u_a","user":"我固定专注19分钟，这是我的个人偏好。"},
                {"turn_id":"u_b","user":"我确认以后仍使用这个19分钟习惯。"}]
        for event in events:self.store.push_buffer(event)
        item={"judgment":"用户自述固定专注19分钟。","confidence":.8,"evidence_ids":["u_a","u_b"],
              "evidence_quotes":[events[1]["user"],events[0]["user"]]}
        result=Consolidator(self.store,llm=lambda prompt:json.dumps([item],ensure_ascii=False)).run_once()
        normalization,=result["quote_normalizations"]
        self.assertEqual(normalization["mapped_turn_ids"],["u_b","u_a"])
        self.assertEqual(normalization["kind"],"exact_full_string_quotes_to_source_objects")
        record=self.store.user_judgments(limit=None)[-1]
        self.assertEqual([q["turn_id"] for q in record["evidence_quotes"]][-2:],["u_a","u_b"])
        audit,=[row for row in self.store.read_ledger() if row["kind"]=="memory_quote_normalization"]
        self.assertEqual(audit["normalization"]["mapped_turn_ids"],["u_b","u_a"])
        rebuilt=Store(self.store.root)
        self.assertEqual(rebuilt.user_judgments(limit=None)[-1]["evidence_quotes"],record["evidence_quotes"])

    def test_string_locators_reject_short_generic_fabricated_and_mixed_shapes(self):
        event={"turn_id":"u_buffer","user":"我固定专注19分钟，不是其他时长。"}
        values=[["专注19分钟"],["从未出现的完整原话"],
                [event["user"],{"turn_id":"u_buffer","quote":event["user"]}]]
        for quotes in values:
            with self.subTest(quotes=quotes):
                model=lambda prompt:json.dumps([{"judgment":"用户自述专注19分钟。","confidence":.8,
                    "evidence_ids":["u_buffer"],"evidence_quotes":quotes}],ensure_ascii=False)
                with self.assertRaises(ValueError):Consolidator(self.store,llm=model)._consolidate([event])
        self.assertEqual([row["id"] for row in self.store.user_judgments()],[self.old["id"]])

    def test_string_quotes_require_explicit_ids_instead_of_guessing_from_batch(self):
        event={"turn_id":"u_buffer","user":"我固定专注19分钟。"}
        model=lambda prompt:json.dumps([{"judgment":"用户自述专注19分钟。","confidence":.8,
                                        "evidence_quotes":[event["user"]]}],ensure_ascii=False)
        with self.assertRaisesRegex(ValueError,"explicitly specified source ids"):
            Consolidator(self.store,llm=model)._consolidate([event])

    def test_string_quote_with_two_identical_selected_sources_is_ambiguous(self):
        user="我固定专注19分钟。"
        events=[{"turn_id":"u_a","user":user},{"turn_id":"u_b","user":user}]
        model=lambda prompt:json.dumps([{"judgment":"用户自述专注19分钟。","confidence":.8,
            "evidence_ids":["u_a","u_b"],"evidence_quotes":[user]}],ensure_ascii=False)
        with self.assertRaisesRegex(ValueError,"unique specified user source"):
            Consolidator(self.store,llm=model)._consolidate(events)

    def test_string_quote_never_infers_an_unselected_id_from_a_matching_message(self):
        events=[{"turn_id":"u_a","user":"我固定专注19分钟。"},
                {"turn_id":"u_b","user":"我每周一去例会。"}]
        model=lambda prompt:json.dumps([{"judgment":"用户自述专注19分钟。","confidence":.8,
            "evidence_ids":["u_b"],"evidence_quotes":[events[0]["user"]]}],ensure_ascii=False)
        with self.assertRaisesRegex(ValueError,"unique specified user source"):
            Consolidator(self.store,llm=model)._consolidate(events)

    def test_failed_string_quote_normalization_preserves_real_batch(self):
        event={"turn_id":"u_buffer","user":"我固定专注19分钟，不是23分钟。"}
        self.store.push_buffer(event)
        model=lambda prompt:json.dumps([{"judgment":"用户自述专注23分钟。","confidence":.9,
            "evidence_ids":["u_buffer"],"evidence_quotes":["我固定专注23分钟。"]}],ensure_ascii=False)
        with self.assertRaises(ValueError):Consolidator(self.store,llm=model).run_once()
        self.assertEqual(len(list(self.store.buffer_dir.glob("*.json"))),1)
        self.assertFalse(any(row["kind"]=="memory_quote_normalization" for row in self.store.read_ledger()))

    def test_generic_anchor_modifiers_normalize_to_a_literal_shared_topic_core(self):
        old_user="我个人固定的专注时长是19分钟，这是我明确采用的日常偏好。"
        self.store.append("turn",{"turn_id":"u_decorated_old","user":old_user})
        target=self.store.add_judgment("用户自述专注19分钟。",.95,["u_decorated_old"],
            evidence_quotes=[{"turn_id":"u_decorated_old","quote":old_user,"evidence_scope":"specified"}])
        user="我现在明确把个人固定专注时长改成23分钟，原来19分钟作废。"
        self.store.append("turn",{"turn_id":"u_decorated_new","user":user})
        self.assertNotIn("固定专注时长",old_user)
        proposal=self.proposal(target_id=target["id"],statement=user,evidence_ids=["u_decorated_new"],
                               target_anchor="固定专注时长")
        result=self.apply(proposal, evidence_ids=["u_decorated_new"],evidence_quotes=[
            {"turn_id":"u_decorated_new","quote":user,"evidence_scope":"specified"}])
        self.assertTrue(result["effective"])
        self.assertEqual(result["operation"]["target_anchor"],"固定专注时长")
        self.assertEqual(result["operation"]["target_anchor_core"],"专注")
        self.assertTrue(result["anchor_normalization"]["changed"])

    def test_anchor_core_is_not_fuzzy_matching_or_negation_removal(self):
        for anchor in ("固定专心时长","不专注时长","个人学习专注时长"):
            with self.subTest(anchor=anchor):
                result=self.apply(self.proposal(target_anchor=anchor))
                self.assertFalse(result["effective"])
                self.assertIn("target_anchor_not_in_target_source",result["reasons"])

    def test_normalized_anchor_core_still_rejects_a_cross_topic_target(self):
        user="我偏好的热饮是茉莉茶。"
        self.store.append("turn",{"turn_id":"u_drink","user":user})
        target=self.store.add_judgment("用户自述偏好茉莉茶。",.95,["u_drink"],
            evidence_quotes=[{"turn_id":"u_drink","quote":user,"evidence_scope":"specified"}])
        self.store.append("turn",{"turn_id":"u_late","user":self.user})
        result=self.apply(self.proposal(target_id=target["id"],evidence_ids=["u_late"],target_anchor="个人固定专注时长"),
            evidence_ids=["u_late"],evidence_quotes=[{"turn_id":"u_late","quote":self.user,"evidence_scope":"specified"}])
        self.assertFalse(result["effective"])
        self.assertEqual(result["operation"]["target_anchor_core"],"专注")
        self.assertIn("target_anchor_not_in_target_source",result["reasons"])

    def test_string_locator_restores_the_full_negation_context(self):
        user="更正：我不喜欢咖啡，偏好热的无糖豆浆，不是冰豆浆。"
        locator="喜欢咖啡，偏好热的无糖豆浆"
        event={"turn_id":"u_loc","user":user}
        self.store.push_buffer(event)
        model=lambda prompt:json.dumps([{"judgment":"用户自述偏好热无糖豆浆。","confidence":.8,
            "evidence_ids":["u_loc"],"evidence_quotes":[locator]}],ensure_ascii=False)
        result=Consolidator(self.store,llm=model).run_once()
        quote=self.store.user_judgments(limit=None)[-1]["evidence_quotes"][0]["quote"]
        self.assertEqual(quote,user)
        self.assertIn("我不喜欢咖啡",quote)
        self.assertIn("不是冰豆浆",quote)
        normalization,=result["quote_normalizations"]
        self.assertEqual(normalization["kind"],"locator_to_full_source")
        self.assertEqual(normalization["mappings"],[{"turn_id":"u_loc","locator":locator,
            "full_user_quote":user,"match":"literal_substring"}])

    def test_string_locator_restores_reported_speech_context_not_only_positive_clause(self):
        user="同事说：‘我固定专注29分钟。’这不是我的习惯，我个人仍然专注19分钟。"
        locator="我固定专注29分钟"
        model=lambda prompt:json.dumps([{"judgment":"用户报告同事的说法，不是其本人偏好。","confidence":.8,
            "evidence_ids":["u_loc"],"evidence_quotes":[locator]}],ensure_ascii=False)
        record,=Consolidator(self.store,llm=model)._consolidate([{"turn_id":"u_loc","user":user}])
        self.assertEqual(record["evidence_quotes"][0]["quote"],user)
        self.assertIn("这不是我的习惯",record["evidence_quotes"][0]["quote"])

    def test_shared_locator_across_different_selected_sources_is_rejected(self):
        events=[{"turn_id":"u_a","user":"我固定专注29分钟，每天继续。"},
                {"turn_id":"u_b","user":"我固定专注29分钟，但之后可能会改。"}]
        model=lambda prompt:json.dumps([{"judgment":"用户自述29分钟。","confidence":.9,
            "evidence_ids":["u_a","u_b"],"evidence_quotes":["我固定专注29分钟"]}],ensure_ascii=False)
        with self.assertRaisesRegex(ValueError,"unique specified user source"):
            Consolidator(self.store,llm=model)._consolidate(events)

    def test_locator_matching_is_literal_not_fuzzy_or_whitespace_rewriting(self):
        event={"turn_id":"u_loc","user":"我固定专注29分钟，这次请记住。"}
        for locator in ("我固定专心29分钟", "我固定专注 29 分钟"):
            with self.subTest(locator=locator):
                model=lambda prompt:json.dumps([{"judgment":"用户自述29分钟。","confidence":.9,
                    "evidence_ids":["u_loc"],"evidence_quotes":[locator]}],ensure_ascii=False)
                with self.assertRaisesRegex(ValueError,"unique specified user source"):
                    Consolidator(self.store,llm=model)._consolidate([event])

    def test_locator_cannot_fill_minimum_length_with_whitespace(self):
        event={"turn_id":"u_loc","user":"我个人的长期约定是每周去例会。"}
        model=lambda prompt:json.dumps([{"judgment":"用户每周去例会。","confidence":.9,
            "evidence_ids":["u_loc"],"evidence_quotes":["我       个人"]}],ensure_ascii=False)
        with self.assertRaisesRegex(ValueError,"eight non-whitespace characters"):
            Consolidator(self.store,llm=model)._consolidate([event])


if __name__ == "__main__": unittest.main()
