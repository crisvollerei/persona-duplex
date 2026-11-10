"""Recall finds effective, evidenced versions without another model call."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from persona.config import PersonaConfig
from persona.llm import make_reply
from persona.session import Session, SessionConfig
from persona.store import Store, rebuild_views

ROOT = Path(__file__).resolve().parent.parent
NOW = '2026-11-10T12:00:00+08:00'


class MemoryRecallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)

    def add_noise(self, n):
        for i in range(n):
            self.store.add_judgment('用户对编号项目' + str(i) + '的记录保持原样。', .7)

    def test_old_relevant_memory_survives_20_100_500_newer_entries(self):
        target = self.store.add_judgment('用户的阅读器是ReaderPro，字体大小为17。', .8)
        current = 0
        for n in (20, 100, 500):
            self.add_noise(n-current); current = n
            with self.subTest(size=n):
                self.assertNotIn(target['id'], [r['id'] for r in self.store.user_judgments()])
                result = self.store.recall_judgments('我的阅读器ReaderPro和字体大小是什么？')
                self.assertIn(target['id'], [r['id'] for r in result])
                self.assertLessEqual(len(result), 5)

    def test_exact_latin_name_ranked_before_similar_name(self):
        target = self.store.add_judgment('ReaderPro阅读器的导出格式为EPUB。', .8)
        self.store.add_judgment('ReaderLite阅读器的导出格式为TXT。', .8)
        self.assertEqual(self.store.recall_judgments('ReaderPro导出格式', 1)[0]['id'], target['id'])

    def test_unsupported_model_memory_is_retained_but_not_recalled(self):
        explicit = self.store.add_judgment('应用明确记录的无引用项。', .8)
        unsupported = self.store.add_judgment('模型猜的阅读器是ReaderLite。', .8,
                                              source='llm', evidence_scope='unsupported')
        old = self.store.add_judgment('旧无来源模型记录。', .8, source='llm')
        self.assertEqual(len(self.store.judgments(limit=None)), 3)
        recalled = self.store.recall_judgments('阅读器', limit=None)
        self.assertEqual([r['id'] for r in recalled], [explicit['id']])
        self.assertNotIn(unsupported['id'], [r['id'] for r in self.store.user_judgments(limit=None)])
        self.assertNotIn(old['id'], [r['id'] for r in self.store.user_judgments(limit=None)])

    def test_global_topic_scope_and_time_window_are_explicit(self):
        global_row = self.store.add_judgment('全局偏好阅读。', .8)
        scoped = self.store.add_judgment('工作阅读设置。', .8, memory_scope='work')
        expired = self.store.add_judgment('昨日临时阅读设置。', .8, expires_at='2026-11-10T00:00:00+08:00')
        future = self.store.add_judgment('明日阅读设置。', .8, valid_from='2026-11-11T00:00:00+08:00')
        self.assertEqual([r['id'] for r in self.store.recall_judgments('阅读', None, at=NOW)], [global_row['id']])
        self.assertEqual({r['id'] for r in self.store.recall_judgments('阅读', None, scope='work', at=NOW)},
                         {global_row['id'], scoped['id']})
        reopened = Store(self.temp.name)
        self.assertEqual(reopened.recall_judgments('阅读', None, scope='work', at=NOW),
                         self.store.recall_judgments('阅读', None, scope='work', at=NOW))

    def test_invalid_scope_and_naive_time_not_silently_permanent(self):
        for kwargs in (dict(memory_scope=''), dict(expires_at='2026-11-10'),
                       dict(valid_from=NOW, expires_at=NOW)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store.add_judgment('临时偏好', .8, **kwargs)

    def test_scope_names_do_not_create_implicit_wildcards(self):
        global_row = self.store.add_judgment('全局阅读设置', .8)
        named = self.store.add_judgment('星号命名空间设置', .8, memory_scope='*')
        self.assertEqual([r['id'] for r in self.store.recall_judgments('设置', None)], [global_row['id']])
        self.assertEqual([r['id'] for r in self.store.recall_judgments('设置', None, scope='work')], [global_row['id']])
        self.assertEqual({r['id'] for r in self.store.recall_judgments('设置', None, scope='*')},
                         {global_row['id'], named['id']})

    def test_temporary_exception_expires_without_invalidating_base(self):
        base = self.store.add_judgment('用户通常专注25分钟。', .8)
        temporary = self.store.add_judgment('用户今晚临时专注35分钟。', .8,
            valid_from='2026-11-10T00:00:00+08:00', expires_at='2026-11-11T00:00:00+08:00')
        self.assertEqual({r['id'] for r in self.store.recall_judgments('专注', None, at=NOW)}, {base['id'], temporary['id']})
        self.assertEqual([r['id'] for r in self.store.recall_judgments('专注', None, at='2026-11-11T00:00:00+08:00')], [base['id']])
        with self.assertRaisesRegex(ValueError, 'scope/window'):
            self.store.add_judgment('错误的未来替换', .8, supersedes=base['id'], valid_from='2026-11-12T00:00:00+08:00')

    def test_inherited_windows_are_revalidated_before_writing(self):
        old = self.store.add_judgment('窗口偏好', .8, valid_from='2026-11-10T00:00:00+08:00',
                                      expires_at='2026-11-11T00:00:00+08:00')
        before = len(self.store.read_ledger())
        with self.assertRaises(ValueError):
            self.store.add_judgment('非法继承窗口', .8, supersedes=old['id'], valid_from='2026-11-12T00:00:00+08:00')
        self.assertEqual(len(self.store.read_ledger()), before)

    def test_character_budget_does_not_cut_a_source_or_negation(self):
        quote = '我不使用ReaderLite，只有周末才使用ReaderPro。' * 100
        self.store.add_judgment('有条件的阅读器偏好', .8, ['u'],
            evidence_quotes=[dict(turn_id='u', quote=quote, evidence_scope='specified')])
        small = self.store.add_judgment('简短应用偏好', .8)
        rows = self.store.recall_judgments('ReaderPro', 5, budget_chars=400)
        self.assertEqual([r['id'] for r in rows], [small['id']])
        self.assertLessEqual(sum(len(json.dumps(r, ensure_ascii=False)) for r in rows), 400)

    def test_custom_retriever_only_selects_effective_ids_and_cannot_mutate_sources(self):
        old = self.store.add_judgment('旧阅读器设置', .8)
        new = self.store.add_judgment('新阅读器设置', .8, supersedes=old['id'])
        def choose(query, candidates):
            candidates[0]['judgment'] = '替换的伪造正文'
            return [new['id']]
        self.assertEqual(self.store.recall_judgments('阅读器', retriever=choose)[0]['judgment'], '新阅读器设置')
        for ids in ([old['id']], ['unknown'], [new['id'], new['id']]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.store.recall_judgments('阅读器', retriever=lambda q, rows: ids)

    def test_revision_and_retraction_do_not_resurrect_after_rebuild(self):
        user = '我明确将代码评审时间固定在下午三点。'
        self.store.append('turn', dict(turn_id='old', user=user))
        old = self.store.add_judgment('用户代码评审时间为下午三点。', .95, ['old'],
            evidence_quotes=[dict(turn_id='old', quote=user, evidence_scope='specified')], memory_scope='work')
        changed = '我明确更正代码评审时间，现在改为下午四点，旧三点作废。'
        self.store.append('turn', dict(turn_id='new', user=changed))
        operation = dict(kind='revise', target_id=old['id'], scope='user_preference',
                         basis='explicit_user_statement', evidence_ids=['new'], statement=changed, target_anchor='代码评审')
        result = self.store.apply_memory_operation(operation, judgment='用户代码评审时间为下午四点。', confidence=.2,
            evidence_ids=['new'], evidence_quotes=[dict(turn_id='new', quote=changed, evidence_scope='specified')])
        self.assertTrue(result['effective'], result['reasons'])
        self.assertEqual(result['replacement']['memory_scope'], 'work')
        removed = '我明确撤销代码评审时间约定，新时间尚未确定。'
        self.store.append('turn', dict(turn_id='gone', user=removed))
        operation.update(kind='retract', target_id=result['replacement_id'], evidence_ids=['gone'], statement=removed)
        final = self.store.apply_memory_operation(operation, evidence_ids=['gone'],
            evidence_quotes=[dict(turn_id='gone', quote=removed, evidence_scope='specified')])
        self.assertTrue(final['effective'], final['reasons'])
        rebuild_views(self.store)
        self.assertEqual(Store(self.temp.name).recall_judgments('代码评审时间', None, scope='work', at=NOW), [])

    def test_frozen_session_freezes_all_memories_then_retrieves_per_question(self):
        target = self.store.add_judgment('用户阅读器名称为ReaderPro。', .8)
        self.add_noise(25)
        cfg = PersonaConfig.load(ROOT/'personas/default.toml')
        seen = []
        def llm(messages):
            seen.append(deepcopy(messages))
            return make_reply('need: 只是想说说话\nconfidence: 0.8\ndirect: ReaderPro\nbaseline: ReaderPro\ndisplace: none',
                              'ReaderPro', 'settle: -\ndelta: -\npending: -')
        with Session(cfg, self.store, llm, options=SessionConfig(memory_mode='frozen')) as session:
            self.assertEqual(len(session.engine.memory_snapshot), 26)
            later = self.store.add_judgment('新出现的秘密阅读器型号AFTER_FREEZE', .8)
            session.say('我之前的阅读器名称是什么？')
        self.assertEqual(len(seen), 1)
        self.assertIn('用户阅读器名称为ReaderPro', str(seen[0]))
        self.assertNotIn('AFTER_FREEZE', str(seen[0]))
        event = [r for r in self.store.read_ledger() if r['kind']=='memory_recall'][-1]
        self.assertIn(target['id'], event['judgment_ids'])
        self.assertNotIn(later['id'], event['judgment_ids'])

    def test_large_recall_budget_also_reaches_the_rendered_source(self):
        quote = '这是完整原话背景。' * 1000 + '我明确选择专注17分钟。'
        self.store.add_judgment('用户有专注时长偏好。', .8, ['long'],
            evidence_quotes=[dict(turn_id='long', quote=quote, evidence_scope='specified')])
        seen = []
        def llm(messages):
            seen.append(messages)
            return make_reply('need: 只是想说说话\nconfidence: 0.8\ndirect: 17分钟\nbaseline: 17分钟\ndisplace: none',
                              '17分钟', 'settle: -\ndelta: -\npending: -')
        cfg = PersonaConfig.load(ROOT/'personas/default.toml')
        with Session(cfg, self.store, llm, options=SessionConfig(memory_mode='frozen', memory_budget_chars=20000)) as session:
            session.say('我的专注时长是多少？')
        self.assertIn('我明确选择专注17分钟。', seen[0][0]['content'])


if __name__ == '__main__':
    unittest.main()
