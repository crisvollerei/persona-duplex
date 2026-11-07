"""Offline checks for fixed research assignments and scoring boundaries."""
import unittest
import run_study as study


class DriverTests(unittest.TestCase):
    def test_assignment(self):
        plan=study.plan()
        self.assertEqual({k:len(v) for k,v in plan.items()},{'r0':48,'r1':156,'r2':156,'b1':144})
        ids=[r['trial_id'] for phase in plan.values() for r in phase]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(plan,study.plan())
        self.assertTrue(set(c['user'] for c in study.make_bank('development')).isdisjoint(c['user'] for c in study.make_bank('holdout')))

    def test_money_unit_and_label_are_separate(self):
        case=next(c for c in study.make_bank('diagnostic') if c['family']=='amount')
        value=case['gold']
        self.assertTrue(study.score_format('结论：'+value+'元',case)['strict_correct'])
        self.assertTrue(study.score_format(value+'元',case)['content_correct'])
        self.assertFalse(study.score_format(value+'元',case)['strict_correct'])
        self.assertFalse(study.score_format('结论：'+value,case)['content_correct'])

    def test_json_truth_and_shape(self):
        case={'kind':'json','family':'json_scalar','gold':{'n':12},'contract':{}}
        self.assertTrue(study.score_format('{"n":12.0}',case)['content_correct'])
        self.assertFalse(study.score_format('{"n":12.0}',case)['format_ok'])
        self.assertFalse(study.score_format('{"n":12,"n":13}',case)['content_covered'])
        self.assertFalse(study.score_format('{"n":true}',case)['content_correct'])
        self.assertFalse(study.score_format('```json\n{"n":12}\n```',case)['format_ok'])
        nested={'kind':'json','family':'json_array','gold':{'selected':[1],'sum':1},'contract':{}}
        self.assertFalse(study.score_format('{"selected":[true],"sum":1}',nested)['content_correct'])

    def test_numeric_table_semantics_and_shape(self):
        case={'kind':'markdown_table','family':'table_numeric','gold':{'甲':'12'},'contract':{'columns':['项目','小计（元）']}}
        text='| 项目 | 小计（元） |\n| --- | --- |\n| 甲 | 12.0 |'
        self.assertTrue(study.score_format(text,case)['content_correct'])
        self.assertFalse(study.score_format(text,case)['format_ok'])

    def test_budget(self):
        budget=study.Budget(799);budget.reserve()
        with self.assertRaises(RuntimeError):budget.reserve()
        self.assertEqual(budget.count,800)


if __name__=='__main__':unittest.main()
