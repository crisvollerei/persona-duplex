"""Run in this directory. All simulated answers go to a fresh temporary directory."""
import tempfile
import unittest
from pathlib import Path
import listener as l


class ListenerStudy(unittest.TestCase):
    def test_whitelist_each_phase_and_no_future_options(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for step in range(18):
                state=l.view(1,l.load(root,1))
                with self.subTest(step=step):
                    self.assertEqual(set(state["item"]),{"actor","context","partner","say"}|
                                     ({"choices"} if step>=6 else set()))
                    if step>=6:
                        phase="action" if step<12 else "motive"
                        self.assertEqual(state["item"]["choices"],l.form(1)[step%6][phase]["choices"])
                l.submit(root,1,dict(step=step,choice="unclear",familiar=(step==0)))
            self.assertTrue(l.view(1,l.load(root,1))["done"])
            totals=l.report(root)
            for g,n in (("novel",5),("familiar",1)):
                self.assertEqual(sum(v["motive"]["unclear"] for v in totals["groups"][g].values()),n)
            self.assertEqual(totals["completed_forms"],1)
            for i in (0,17,18):
                with self.assertRaises(ValueError):l.submit(root,1,dict(step=i,choice="unclear"))

    def test_abstention_is_not_missing_and_cannot_attribute_text_the_role_did_not_say(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in range(6):l.submit(root,1,dict(step=i,choice="unclear",familiar=False))
            p=l.report(root)
            self.assertEqual(p["completed_forms"],0)
            self.assertEqual(sum(v["action"]["missing"] for v in p["groups"]["partial"].values()),6)
            item=l.form(1)[0]
            with self.assertRaises(ValueError):l.submit(root,1,dict(step=6,choice="A",quote=item["partner"]))
            with self.assertRaises(ValueError):l.submit(root,1,dict(step=7,choice="unclear"))
            l.submit(root,1,dict(step=6,choice=item["action"]["target"],quote=item["say"][:5]))
            group=l.report(root)["groups"]["partial"][item["role"]+"."+item["arm"]]
            self.assertEqual(group["action"]["target"],1)

    def test_full_source_coverage_without_scoring_based_selection(self):
        l.validate_material()
        self.assertEqual(l.register(),l.json.loads((l.HERE/"registration.json").read_bytes()))

if __name__=="__main__":
    unittest.main()
