"""判断记忆迁入账本 + 活跃度驱动的容量。

迁移的目的不是「换个地方存」，而是让判断层**可重放**：
    同一份账本 + 不同版本的固化逻辑 -> 重放出不同的人格轨迹

在此之前，judgments 是账本之外的 jsonl 旁路，
state/ 删了能重建关系，但判断永远重建不出来 —— 那就不算记忆。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona.consolidate import (                                # noqa: E402
    ACTIVITY_PROFILES,
    Consolidator,
    activity_of,
)
from persona.store import (                                      # noqa: E402
    Store,
    append_judgment,
    rebuild_views,
    replay_judgments,
)


class TestJudgmentInLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_judgment_writes_ledger(self):
        self.store.add_judgment("他倾向于先自我否定", 0.7)
        kinds = [r["kind"] for r in self.store.read_ledger()]
        self.assertIn("judgment_add", kinds)

    def test_judgments_survive_rebuild(self):
        """删掉一切可从账本重建的东西，判断必须还在。"""
        self.store.add_judgment("他倾向于先自我否定", 0.7)
        self.store.add_judgment("他每次都绕回工作", 0.6)
        for p in self.store.state_dir.glob("*"):
            p.unlink()
        rebuild_views(self.store)
        self.assertEqual(
            [j["judgment"] for j in self.store.judgments()],
            ["他倾向于先自我否定", "他每次都绕回工作"])

    def test_supersede_hides_old(self):
        a = self.store.add_judgment("他喜欢猫", 0.5)
        self.store.add_judgment("其实他对猫过敏", 0.8, supersedes=a["id"])
        live = [j["judgment"] for j in self.store.judgments()]
        self.assertEqual(live, ["其实他对猫过敏"])

    def test_supersede_chain(self):
        """A <- B <- C：C 推翻 B，B 推翻 A。活跃的只有 C。"""
        a = self.store.add_judgment("A", 0.5)
        b = self.store.add_judgment("B", 0.6, supersedes=a["id"])
        self.store.add_judgment("C", 0.7, supersedes=b["id"])
        self.assertEqual([j["judgment"] for j in self.store.judgments()], ["C"])

    def test_ids_continue_after_restart(self):
        self.store.add_judgment("第一条", 0.5)
        reopened = Store(self.store.root)
        rec = reopened.add_judgment("第二条", 0.5)
        self.assertNotEqual(rec["id"], "j_0001")
        self.assertEqual(len({j["id"] for j in reopened.judgments()}), 2)

    def test_affects_persisted(self):
        """affects 是中期层推动人格参数的依据，必须能存能读。"""
        self.store.add_judgment("他不想被建议", 0.7,
                                affects={"intervention": -1})
        j = self.store.judgments()[0]
        self.assertEqual(j["affects"], {"intervention": -1})

    def test_legacy_jsonl_imported_once(self):
        """老的 judgments.jsonl 自动导入账本，且只导一次。"""
        append_judgment(self.store.judgments_path, "老格式一", 0.6)
        append_judgment(self.store.judgments_path, "老格式二", 0.7)
        first = [j["judgment"] for j in self.store.judgments()]
        self.assertEqual(first, ["老格式一", "老格式二"])
        # 再读一次不应该重复导入
        second = [j["judgment"] for j in self.store.judgments()]
        self.assertEqual(first, second)
        adds = [r for r in self.store.read_ledger() if r["kind"] == "judgment_add"]
        self.assertEqual(len(adds), 2)

    def test_replay_is_pure(self):
        """replay_judgments 不该修改输入。"""
        self.store.add_judgment("x", 0.5)
        ledger = self.store.read_ledger()
        before = json.dumps(ledger, ensure_ascii=False, sort_keys=True)
        replay_judgments(ledger)
        after = json.dumps(ledger, ensure_ascii=False, sort_keys=True)
        self.assertEqual(before, after)


class TestActivityProfile(unittest.TestCase):
    def test_dense_is_stricter(self):
        """密集对话要更多重复才算倾向 —— 把「状态」和「倾向」分开。"""
        self.assertGreater(ACTIVITY_PROFILES["dense"]["threshold"],
                           ACTIVITY_PROFILES["sparse"]["threshold"])

    def test_dense_has_more_capacity(self):
        self.assertGreater(ACTIVITY_PROFILES["dense"]["capacity"],
                           ACTIVITY_PROFILES["sparse"]["capacity"])

    def test_activity_from_gap(self):
        self.assertEqual(activity_of([], gap_hours=0.5), "dense")
        self.assertEqual(activity_of([], gap_hours=6), "medium")
        self.assertEqual(activity_of([], gap_hours=48), "sparse")

    def test_no_timing_defaults_to_sparse(self):
        """没有时间信息时按最宽松处理 —— 离线补跑该把证据都用上。"""
        self.assertEqual(activity_of([]), "sparse")

    def test_consolidator_reports_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            c = Consolidator(store, gap_hours=0.2)
            self.assertEqual(c.profile()["threshold"], 5)

    def test_activity_judged_before_drain(self):
        """活跃度必须在 drain 之前判。

        drain 会清空 buffer，之后再数积压永远是 0，于是永远走 sparse。
        实测踩过：连续聊 6 轮，reported activity 却是 sparse，
        因为 profile() 在 drain_buffer() 之后才调用。逻辑没错，顺序错了。
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            for i in range(6):
                store.push_buffer({"turn_id": f"t_{i}", "user": "x", "say": "y",
                                   "direct": "z", "need": "只想被听",
                                   "displace": "none"})
            out = Consolidator(store).run_once()
            self.assertEqual(out["activity"], "medium",
                             "6 条积压应该判成 medium，不是 sparse")
            self.assertEqual(out["threshold"], 3)

    def test_first_run_with_small_backlog_is_sparse(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            for i in range(2):
                store.push_buffer({"turn_id": f"t_{i}", "user": "x", "say": "y",
                                   "direct": "z", "need": "只想被听",
                                   "displace": "none"})
            out = Consolidator(store).run_once()
            self.assertEqual(out["activity"], "sparse")

    def test_dense_needs_five_repeats(self):
        """密集档：只出现 3 次不该形成判断。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            for i in range(3):
                store.push_buffer({"turn_id": f"t_{i}", "user": "x", "say": "y",
                                   "direct": "z", "need": "只想被听",
                                   "displace": "none"})
            c = Consolidator(store, gap_hours=0.2)
            out = c.run_once()
            self.assertEqual(out["judgments"], 0, "密集档不该 3 次就认")

    def test_sparse_needs_only_two(self):
        """稀疏档：出现 2 次就够了。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            for i in range(2):
                store.push_buffer({"turn_id": f"t_{i}", "user": "x", "say": "y",
                                   "direct": "z", "need": "只想被听",
                                   "displace": "none"})
            c = Consolidator(store, gap_hours=48)
            out = c.run_once()
            self.assertEqual(out["judgments"], 1)
            self.assertEqual(out["activity"], "sparse")


if __name__ == "__main__":
    unittest.main(verbosity=2)
