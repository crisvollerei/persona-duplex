"""Caller-owned response scope does not claim semantic enforcement."""
from dataclasses import FrozenInstanceError, asdict
import unittest

from persona.scope import ResponseScope, validate_scope


class ResponseScopeTests(unittest.TestCase):
    def test_defaults_are_serializable_and_immutable(self):
        scope = ResponseScope()
        self.assertEqual(asdict(scope), dict(max_steps=None, max_sentences=None, prefix="",
            evidence_mode="open", allow_suggestions=None, allow_questions=None))
        with self.assertRaises(FrozenInstanceError):
            scope.prefix = "changed"

    def test_invalid_fields_fail_before_generation(self):
        invalid = [dict(max_steps=0), dict(max_steps=-1), dict(max_steps=True),
                   dict(max_steps=1.0), dict(max_sentences=False), dict(max_sentences=0),
                   dict(max_sentences="1"), dict(prefix=None), dict(prefix=" \t"),
                   dict(prefix="草稿：\n"), dict(prefix=" 草稿："), dict(prefix="草稿： "),
                   dict(evidence_mode="guess"),
                   dict(allow_suggestions=1), dict(allow_questions="false")]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ResponseScope(**values)

    def test_no_scope_is_not_requested(self):
        result = validate_scope("已有回复", None)
        self.assertEqual(result["status"], "not_requested")
        self.assertIsNone(result["valid"])
        self.assertEqual(result["checked"], [])

    def test_prompt_places_request_and_permission_before_persona(self):
        scope = ResponseScope(max_steps=1, max_sentences=2, prefix="草稿：",
            evidence_mode="records_only", allow_suggestions=False, allow_questions=False)
        prompt = scope.prompt()
        for expected in ("先完成本轮明确任务", "最多提供1个", "最多2句", "逐字前缀",
                         "默认规则", "不提出新增建议", "不向用户提出新增问题", "这些风格本身不改变"):
            self.assertIn(expected, prompt)
        self.assertNotIn("35分钟", prompt)

    def test_allow_does_not_require_behavior_or_change_future_permission(self):
        prompt = ResponseScope(allow_suggestions=True, allow_questions=True).prompt()
        self.assertIn("不要求一定给建议", prompt)
        self.assertIn("不要求一定问", prompt)
        self.assertIn("不授予长期权限", prompt)
        result = validate_scope("收到。", ResponseScope(allow_suggestions=True, allow_questions=True))
        self.assertTrue(result["valid"])

    def test_prefix_is_exact_and_no_reply_rewriting_occurs(self):
        body = "先说明一下。\n草稿：今天发布。"
        result = validate_scope(body, ResponseScope(prefix="草稿："))
        self.assertFalse(result["valid"])
        self.assertIn("missing_required_prefix", result["reasons"])
        self.assertEqual(body, "先说明一下。\n草稿：今天发布。")
        self.assertTrue(validate_scope(" \n草稿：今天发布。 ", ResponseScope(prefix="草稿："))["valid"])

    def test_explicit_numbered_step_limit(self):
        variants = ["1. 先分出纽扣。\n2. 再分类线轴。",
                    "1、先分出纽扣。\n2、再分类线轴。",
                    "（1）先分出纽扣。\n（2）再分类线轴。",
                    "第一步：先分出纽扣。\n第二步：再分类线轴。",
                    "一、先分出纽扣。\n二、再分类线轴。"]
        for body in variants:
            with self.subTest(body=body):
                result = validate_scope(body, ResponseScope(max_steps=1))
                self.assertFalse(result["valid"])
                self.assertEqual(result["reasons"], ["max_steps_exceeded"])
                self.assertEqual(result["observed"]["max_steps"], 2)

    def test_unstructured_composite_action_is_not_semantically_counted(self):
        result = validate_scope("把东西倒出，分两堆，再放进容器。", ResponseScope(max_steps=1))
        self.assertEqual(result["status"], "partially_assessed")
        self.assertIsNone(result["valid"])
        self.assertEqual(result["not_assessed"], [dict(field="max_steps", reason="no_explicit_numbered_steps")])

    def test_numbered_subitems_do_not_become_main_steps(self):
        body = "1. 分离纽扣\n   1. 查看盒底\n   2. 查看角落"
        result = validate_scope(body, ResponseScope(max_steps=1))
        self.assertTrue(result["valid"])
        self.assertEqual(result["checked"], ["nonempty_reply", "max_steps"])

    def test_inline_numbering_code_tables_and_quotes_are_unassessed(self):
        bodies = ["1. 先查看 2. 再收起", "```text\n1. 示例\n2. 示例\n```",
                  "> 1. 用户提到的步骤", "| 1. | 仅为表格值 |"]
        for body in bodies:
            with self.subTest(body=body):
                result = validate_scope(body, ResponseScope(max_steps=1))
                self.assertIsNone(result["valid"])
                self.assertIn("max_steps", [row["field"] for row in result["not_assessed"]])

    def test_simple_sentence_limit_and_decimals(self):
        scope = ResponseScope(max_sentences=1)
        for body in ("字号17，行距1.4。", "还没确定", "收到了！？", "结果为35。"):
            with self.subTest(body=body):
                self.assertTrue(validate_scope(body, scope)["valid"])
        for body in ("今天，它终于离岸了。愿它一路被看见。", "先回答。再提一个问题？", "第一句。尾部再追加"):
            with self.subTest(body=body):
                result = validate_scope(body, scope)
                self.assertFalse(result["valid"])
                self.assertIn("max_sentences_exceeded", result["reasons"])

    def test_sentence_counter_does_not_judge_code_quotes_or_abbreviations(self):
        bodies = ['她写道：“今天发布。值得庆祝。”', '```json\n{"n":1.4}\n```',
                  'Mr. Lin arrives.', '第一行\n第二行', '先等等……再决定',
                  '{"value":"这一句。下一句。"}', "| 一句。 | 另一句。 |",
                  "（说明：尚未确认。）", "体积（升）是2.3。"]
        for body in bodies:
            with self.subTest(body=body):
                result = validate_scope(body, ResponseScope(max_sentences=1))
                self.assertIsNone(result["valid"])
                self.assertEqual(result["not_assessed"][0]["field"], "max_sentences")

    def test_semantic_fields_never_certified_from_keywords(self):
        scope = ResponseScope(evidence_mode="records_only", allow_suggestions=False, allow_questions=False)
        for body in ("收到。", "用户写的是“你愿意吗？”。", "不妨先去检查接口。", "从041依次递增。"):
            with self.subTest(body=body):
                result = validate_scope(body, scope)
                self.assertIsNone(result["valid"])
                self.assertEqual(result["status"], "partially_assessed")
                self.assertEqual({row["field"] for row in result["not_assessed"]},
                                 {"evidence_mode", "allow_suggestions", "allow_questions"})

    def test_known_failure_is_preserved_even_with_unassessed_semantics(self):
        result = validate_scope("说明：先问个问题？", ResponseScope(prefix="草稿：", allow_questions=False))
        self.assertFalse(result["valid"])
        self.assertTrue(result["not_assessed"])
        self.assertEqual(result["status"], "violated")

    def test_empty_reply_is_a_failure_not_successful_restraint(self):
        result = validate_scope(" \n", ResponseScope(allow_suggestions=False))
        self.assertFalse(result["valid"])
        self.assertIn("empty_reply", result["reasons"])

    def test_wrong_argument_types(self):
        with self.assertRaises(TypeError):
            validate_scope(None, ResponseScope())
        with self.assertRaises(TypeError):
            validate_scope("text", {})


if __name__ == "__main__":
    unittest.main()
