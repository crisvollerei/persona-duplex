import json
import unittest

from persona.disclosure_evidence import Exchange, attribute
from persona.disclosure_claim import attribute_in_order


class DisclosureEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.pair = Exchange("tang", "yu", "我想留着自己写错的那页。", "好，这一页我不碰。", True)
        self.reading = dict(subject="tang", target="tang", kind="personal", reaction="received",
                            said="留着自己写错的那页", heard="这一页我不碰", why="尊重本人界线")

    def grade(self, pair=None, **updates):
        return attribute(pair or self.pair, json.dumps(self.reading | updates, ensure_ascii=False))

    def test_both_endpoints_must_refer_to_the_first_speaker(self):
        self.assertEqual(self.grade()["vote"], 1)
        for updates in ({"subject":"yu"}, {"target":"yu"}, {"kind":"work"}, {"reaction":"other"}):
            with self.subTest(updates=updates):
                self.assertEqual(self.grade(**updates)["label"], 0)

    def test_archive_mismatch_or_missing_label_is_not_negative_evidence(self):
        for updates in ({"said":"我不碰"}, {"heard":"我同意不碰"}, {"said":""},
                        {"subject":"unknown"}, {"reaction":None}):
            with self.subTest(updates=updates):
                row=self.grade(**updates)
                self.assertIsNone(row["label"])
                self.assertEqual(row["vote"], 0)
        self.assertIsNone(attribute(self.pair, "no json")["label"])

    def test_protocol_can_veto_without_erasing_what_the_two_said(self):
        pair=Exchange("tang","yu",self.pair.first,self.pair.second,False)
        row=self.grade(pair)
        self.assertEqual((row["label"],row["vote"]),(1,0))

    def test_negative_only_on_linked_shaming(self):
        pair=Exchange("tang","yu",self.pair.first,"这种私心也好意思说？以后别开口。",True)
        self.assertEqual(self.grade(pair,reaction="shamed",heard="这种私心也好意思说")["vote"],-1)
        self.assertEqual(self.grade(pair,reaction="shamed",target="yu",heard="这种私心也好意思说")["vote"],0)

    def test_judge_sees_no_privileged_flags_or_private_background(self):
        self.assertEqual(set(json.loads(self.pair.messages()[1]["content"])),
                         {"first_id","second_id","first","second"})
        with self.assertRaises(ValueError):
            Exchange("yu","yu","a","b",True)

    def test_reply_cannot_retroactively_supply_a_claim(self):
        calls=[]
        def no_claim(stage,messages):
            calls.append(stage)
            data=json.loads(messages[1]["content"])
            self.assertEqual(data,{"speaker":"tang","speech":self.pair.first})
            return '{"decision":"none","quote":"","claim":""}'
        self.assertEqual(attribute_in_order(self.pair,no_claim)["label"],0)
        self.assertEqual(calls,["first"])

    def test_two_stage_rejects_quotes_taken_from_the_wrong_turn(self):
        def swapped(stage,msg):
            self.assertEqual(stage,"first")
            return json.dumps(dict(decision="own",quote="这一页我不碰",claim="保护草稿"))
        self.assertIsNone(attribute_in_order(self.pair,swapped)["label"])

    def test_link_selects_same_claim_and_vetoes_invalid_protocol(self):
        def read(stage,msg):
            if stage=="first":return json.dumps(dict(decision="own",quote="自己写错的那页",claim="自己保留算错的草稿"))
            payload=json.loads(msg[1]["content"])
            self.assertEqual(payload["claim"]["speaker"], "tang")
            self.assertEqual(payload["responder"], "yu")
            self.assertEqual(payload["claim"]["quote"], "自己写错的那页")
            return json.dumps(dict(decision="received",quote="这一页我不碰",why="尊重保留草稿的界线"))
        pair=Exchange("tang","yu",self.pair.first,self.pair.second,False)
        row=attribute_in_order(pair,read)
        self.assertEqual((row["label"],row["vote"]),(1,0))
