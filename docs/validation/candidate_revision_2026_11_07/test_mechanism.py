import json
import unittest
from mechanism import coverage_plan, blind_candidates, validate_annotations, choose
from benchmark import trial, pool_messages, label_messages
from unittest.mock import Mock


class MechanismTests(unittest.TestCase):
    def setUp(self):
        self.case = {"id":"N", "user":"别问我问题", "history":[], "facts":[], "boundaries":["别问我问题"],
                     "controls":{"no_questions":True,"no_advice":False}}
        self.hidden = [{"id":"X01","text":"事情已经结束。听起来你很开心。"}]
        self.mapping = {"X01":"C01"}
        self.label = {"id":"X01","attention":1,"focus_evidence":[{"kind":"content","quote":"事情已经结束。"},
                         {"kind":"experience","quote":"听起来你很开心。"}],"initiative":0,"initiative_quote":self.hidden[0]["text"]}

    def parse(self, label): return validate_annotations(json.dumps({"labels":[label]}), self.hidden, self.mapping)[0]

    def test_control_sources_and_no_question_plan(self):
        plan = coverage_plan(self.case)
        self.assertIn("用户已禁问",plan[2]["brief"])
        bad = {**self.case,"boundaries":["不存在的原文"]}
        with self.assertRaises(ValueError): coverage_plan(bad)

    def test_blinding_hides_plan_features_and_preserves_text(self):
        candidates = [{"id":"C01","text":"接话","attention":2,"initiative":1}]
        hidden, mapping = blind_candidates("N-coverage",candidates)
        self.assertEqual(hidden,[{"id":"X01","text":"接话"}]); self.assertEqual(mapping,{"X01":"C01"})

    def test_valid_quotes_are_accepted_not_certified_true(self):
        self.assertEqual((self.parse(self.label)["attention"],self.parse(self.label)["initiative"]),(1,0))

    def test_invented_quote_does_not_fall_back_to_self(self):
        changed = {**self.label,"focus_evidence":[{"kind":"content","quote":"不存在的内容"}]}
        row = self.parse(changed); self.assertIsNone(row["attention"]); self.assertEqual(row["initiative"],0)
        self.assertIsNone(choose([{"id":"C01","text":"回复","attention":1,"initiative":0}],[row],{"target":{"attention":1,"initiative":0}}))

    def test_absence_requires_entire_text(self):
        row = self.parse({**self.label,"initiative_quote":"事情已经结束。"})
        self.assertIsNone(row["initiative"]); self.assertIn("absence_requires_whole_text",row["validation_errors"])

    def test_unknown_is_preserved_and_mixed_requires_two_spans(self):
        row = self.parse({**self.label,"attention":None,"focus_evidence":[]})
        self.assertIsNone(row["attention"])
        row = self.parse({**self.label,"focus_evidence":[self.label["focus_evidence"][0]]})
        self.assertIsNone(row["attention"])

    def test_duplicate_ids_and_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError): validate_annotations('{"labels":[],"labels":[]}', self.hidden,self.mapping)
        with self.assertRaises(ValueError): validate_annotations(json.dumps({"labels":[self.label,self.label]}),self.hidden,self.mapping)

    def test_context_is_equal_and_labels_have_no_original_features(self):
        p0=json.loads(pool_messages(self.case,"baseline")[1]["content"])
        p1=json.loads(pool_messages(self.case,"coverage")[1]["content"])
        msgs,_,_=label_messages(self.case,"N",[{"id":"C01","text":"回复","attention":1,"initiative":0}])
        tagged=json.loads(msgs[1]["content"])
        self.assertEqual(p0["context"],p1["context"]);self.assertEqual(p0["context"],tagged["context"])
        self.assertNotIn("coverage_plan",tagged);self.assertNotIn("attention",tagged["candidates"][0])

    def test_selection_preserves_exact_text(self):
        result=choose([{"id":"C01","text":"原文。"}],[{"id":"C01","attention":2,"initiative":0}],{"target":{"attention":2,"initiative":0}})
        self.assertEqual(result,{"candidate_id":"C01","text":"原文。"})

    def test_label_failure_is_retained_without_retry(self):
        candidates=[{"id":f"C{i:02}","text":f"回复{i}","attention":1,"initiative":0} for i in range(1,7)]
        client=Mock();client.call.side_effect=[{"id":"N-baseline-pool","status":"ok","output":json.dumps({"candidates":candidates})},
                                             {"id":"N-baseline-labels","status":"error"}]
        result=trial(self.case,"baseline",client,[{"id":"P","target":{"attention":1,"initiative":0}}])
        self.assertEqual(client.call.call_count,2);self.assertEqual(result["selections"][1]["status"],"abstained")
        self.assertEqual(result["selections"][0]["status"],"ok")


if __name__=="__main__": unittest.main()
