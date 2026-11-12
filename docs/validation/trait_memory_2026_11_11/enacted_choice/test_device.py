import unittest
import trial as t

class BallotDevice(unittest.TestCase):
    def event(self,say,raw=None,q=None,layout=0):
        q=q or t.SCENES[0];entries,goal=t.menu(q,"tang",layout)
        return t.transition(q,entries,goal,say,raw if raw is not None else "<<<SAY>>>\n"+say)

    def test_no_reading_internal_or_ambiguous_decisions(self):
        for say,raw in [("", "<<<HEAD>>>\n落签：青签"),("落签：白签","落签：白签"),
            ("落签：青签\n落签：白签",None),("我会落签：青签",None),("落签：青签，如果你也这样想。",None),
            ("落签：青签\n你觉得呢？",None),("落签：青签","<<<SAY>>>\n落签：青签\n<<<SAY>>>")]:
            with self.subTest(say=say,raw=raw):
                z=self.event(say,raw);self.assertIsNone(z["target"]);self.assertEqual(z["before"],z["after"])

    def test_menu_is_binding_and_choice_alone_is_sufficient(self):
        for q in t.SCENES:
            for a in t.base.ROLES:
                for n in range(4):
                    opts,goal=t.menu(q,a,n)
                    for x in opts:
                        say="落签："+x["ticket"]
                        z=t.transition(q,opts,goal,say,"<<<SAY>>>\n"+say)
                        self.assertEqual(z["chosen"],x["value"])
                        if q["family"]=="control" and not z["target"]:
                            self.assertEqual(z["status"],"blocked_by_width");self.assertEqual(z["before"],z["after"])
                        else:self.assertEqual(z["after"][q["field"]],x["value"])

if __name__=="__main__":unittest.main()
