"""Offline evidence routing, balance and scoring checks."""
from collections import Counter
from datetime import datetime
from decimal import Decimal
import unittest
import benchmark as bench


class BenchmarkTests(unittest.TestCase):
    def test_factorial_and_unique_jobs(self):
        cases,corpus=bench.build_cases();jobs=bench.assignments(cases)
        self.assertEqual(len(cases),20);self.assertEqual(len(corpus),120)
        self.assertEqual(len(jobs),224);self.assertEqual(len({r['trial_id'] for r in jobs}),224)
        self.assertEqual(sum(r['kind']=='open' for r in jobs),32)
        counts=Counter((r['method'],r['profile']) for r in jobs if r['kind']!='open')
        self.assertEqual(set(counts.values()),{32});self.assertEqual(len(counts),6)
        self.assertEqual(jobs,bench.assignments(cases))

    def test_retrieval_complete_same_budget(self):
        cases,corpus=bench.build_cases()
        for case in cases:
            result=bench.retrieve(case,corpus)
            self.assertEqual(len(result['documents']),6)
            self.assertTrue(result['required_sources_present'])
            self.assertLessEqual(result['character_count'],2600)
            self.assertEqual(result,bench.retrieve(case,list(reversed(corpus))))
            self.assertTrue(all(d['entity']==case['entity'] for d in result['documents']))
            self.assertNotIn(case['family'],result['context'])
            self.assertNotIn('"type":',result['context'])

    def test_content_not_confused_with_label(self):
        case=bench.build_cases()[0][0]
        self.assertTrue(bench.score(case['gold']+'元',case)['target_correct'])
        self.assertFalse(bench.score(case['gold']+'元',case)['strict_correct'])
        self.assertTrue(bench.score('结论：'+case['gold']+'元',case)['strict_correct'])
        self.assertFalse(bench.score('结论：'+case['gold'],case)['target_correct'])
        self.assertIsNone(bench.score('说明而没有独立目标',case)['target_correct'])

    def test_unknown_and_opposites(self):
        cases,_=bench.build_cases()
        case=next(c for c in cases if c['family']=='insufficient')
        self.assertTrue(bench.score('结论：无法确定',case)['target_correct'])
        self.assertFalse(bench.score('结论：可确定',case)['target_correct'])
        conflict=next(c for c in cases if c['family']=='unresolved')
        self.assertTrue(bench.score('结论：未知',conflict)['strict_correct'])
        minute=next(c for c in cases if c['family']=='updated')
        self.assertTrue(bench.score('结论：“'+minute['gold'][:-2]+'.0 分钟”',minute)['target_correct'])

    def test_gold_targets_score_and_money_independent_arithmetic(self):
        for case in bench.build_cases()[0]:
            if case['kind']=='open':continue
            text='结论：'+case['gold']+('元' if case['kind']=='number' else '')
            self.assertTrue(bench.score(text,case)['strict_correct'],case['id'])
        self.assertEqual(3850-1270-485,2095)
        self.assertEqual(4620-1690-735,2195)

    def test_error_circuit_and_cap(self):
        budget=bench.Budget()
        for _ in range(bench.LIMIT):budget.reserve()
        with self.assertRaises(RuntimeError):budget.reserve()
        budget=bench.Budget()
        for _ in range(3):budget.fail()
        with self.assertRaises(RuntimeError):budget.reserve()
        self.assertEqual(budget.count,0)


if __name__=='__main__':unittest.main()
