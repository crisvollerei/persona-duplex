"""Offline checks of research scoring, assignments and safety bounds."""
import copy
from pathlib import Path
import tempfile
import unittest

import study


class StudyChecks(unittest.TestCase):
    def test_exact_assignment_counts(self):
        jobs = study.jobs()
        self.assertEqual(len(jobs), 150)
        self.assertEqual(len({job['trial_id'] for job in jobs}), 150)
        self.assertEqual(sum(job['experiment'] == 'facts' for job in jobs), 96)
        self.assertEqual(sum(job['experiment'] == 'controlled_memory' for job in jobs), 48)
        self.assertEqual(sum(job['experiment'] == 'learned_update' for job in jobs), 6)
        self.assertEqual(study.jobs(), jobs)

    def test_decimal_point_and_unit_preserved(self):
        target = dict(kind='number', gold='37.5', unit='%')
        for value in ('结论：37.5%', '结论：37.50%。', '结论：<37.5>%', '结论：<37.5%>'):
            self.assertTrue(study.score(value, target)['strict_correct'], value)
        for value in ('结论：375%', '结论：37.5', '结论：37.5元', '结论：<37.5%'):
            self.assertFalse(study.score(value, target)['strict_correct'], value)

    def test_format_and_content_separate(self):
        target = dict(kind='label', gold='未知', unit='')
        score = study.score('结论：未知\n还需要更多依据。', target)
        self.assertTrue(score['labelled_content_correct'])
        self.assertFalse(score['strict_correct'])
        self.assertFalse(study.score('目前未知。', target)['label_present'])
        self.assertTrue(study.score('说明。\n**结论：未知。**', target)['strict_correct'])

    def test_labels_do_not_match_substrings(self):
        target = dict(kind='label', gold='不满足', unit='')
        self.assertFalse(study.score('结论：满足', target)['strict_correct'])
        self.assertFalse(study.score('结论：不满足或满足', target)['strict_correct'])

    def test_time_equivalence(self):
        target = dict(kind='time', gold='01:03', unit='')
        self.assertTrue(study.score('结论：1:03', target)['strict_correct'])
        self.assertFalse(study.score('结论：01:30', target)['strict_correct'])

    def test_learned_slot_pair(self):
        target = dict(kind='label', gold='专注23分钟,休息6分钟', unit='')
        self.assertTrue(study.score('结论：专注23分钟，休息6分钟。', target)['strict_correct'])
        self.assertFalse(study.score('结论：专注23分钟，休息4分钟。', target)['strict_correct'])

    def test_all_gold_answers_and_wrong_controls(self):
        for case in study.facts():
            self.assertTrue(study.score('结论：' + case['gold'] + case['unit'], case)['strict_correct'])
        for case in study.memory_cases():
            for condition in study.CONTROLLED:
                _, _, target = study.memory_input(case, condition)
                self.assertTrue(study.score('结论：' + target['gold'] + target['unit'], target)['strict_correct'])

    def test_confidence_position_orthogonal(self):
        combinations = []
        for case in study.memory_cases():
            _, memories, _ = study.memory_input(case, 'unresolved')
            combinations.append((memories[0]['evidence_ids'][0], max(memories, key=lambda m:m['confidence'])['evidence_ids'][0]))
        self.assertEqual(len(set(combinations)), 4)
        self.assertTrue(all(combinations.count(value) == 2 for value in set(combinations)))

    def test_all_memory_engine_constructions_offline(self):
        budget = study.Budget()
        with tempfile.TemporaryDirectory() as directory:
            for case in study.memory_cases():
                for condition in study.CONTROLLED:
                    _, memories, _ = study.memory_input(case, condition)
                    model = study.engine(study.configs()['neutral'], Path(directory)/case['case_id']/condition, [], budget, memories)
                    self.assertEqual(len(model.memory_snapshot), len(memories))
        self.assertEqual(budget.count, 0)

    def test_hard_budget_and_error_stop(self):
        budget = study.Budget()
        for _ in range(study.HARD_LIMIT): budget.reserve()
        with self.assertRaises(RuntimeError): budget.reserve()
        self.assertEqual(budget.count, 210)
        budget = study.Budget()
        for _ in range(3): budget.fail()
        with self.assertRaises(RuntimeError): budget.reserve()
        self.assertEqual(budget.count, 0)


if __name__ == '__main__': unittest.main()
