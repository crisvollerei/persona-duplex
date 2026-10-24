"""Section recovery must not certify an invalid measurement or state proposal."""
from __future__ import annotations

import unittest

from persona.config import DISPLACE_NONE
from persona.parse import Head, ParsedTurn, Tail, parse, parse_head, section_errors
from persona.prompt import SECTION_HEAD, SECTION_SAY, SECTION_TAIL, build_history_entry
from persona.stream import StreamParser


HEAD = ("need: check the sum\nconfidence: 0.8\ndirect: 4\n"
        "baseline: It is 4.\ndisplace: none\n")
SAY = "It is 4."
TAIL = "settle: -\npending: -\n"
VALID = SECTION_HEAD + HEAD + SECTION_SAY + SAY + SECTION_TAIL + TAIL


def stream(raw, *, expression_only=False, width=1):
    parser = StreamParser(expression_only=expression_only)
    visible = []
    for i in range(0, len(raw), width):
        visible.extend(parser.feed(raw[i:i + width]))
    return parser.finish(), "".join(visible)


class TestSectionContract(unittest.TestCase):
    def assert_invalid_in_both(self, raw, expected):
        batch = parse(raw)
        self.assertIn(expected, batch.structural_errors)
        self.assertFalse(batch.validity()["valid"])
        self.assertFalse(batch.tail_usable)
        for width in (1, 7, len(raw)):
            with self.subTest(width=width):
                incremental, visible = stream(raw, width=width)
                self.assertEqual(incremental.structural_errors, batch.structural_errors)
                self.assertFalse(incremental.validity()["valid"])
                self.assertFalse(incremental.tail_usable)
                self.assertNotIn("direct:", visible)
        return batch

    def test_valid_default_is_usable(self):
        for parsed in (parse(VALID), stream(VALID)[0]):
            self.assertEqual(parsed.structural_errors, [])
            self.assertTrue(parsed.validity()["valid"])
            self.assertTrue(parsed.tail_usable)

    def test_say_before_head_is_not_a_valid_inner_reference(self):
        raw = SECTION_SAY + SAY + SECTION_HEAD + HEAD + SECTION_TAIL + TAIL
        parsed = self.assert_invalid_in_both(raw, "invalid_section_order")
        self.assertEqual(parsed.say, SAY)

    def test_tail_before_say_is_invalid(self):
        raw = SECTION_HEAD + HEAD + SECTION_TAIL + TAIL + SECTION_SAY + SAY
        self.assert_invalid_in_both(raw, "invalid_section_order")

    def test_duplicate_markers_are_not_silently_collapsed(self):
        for marker, name in ((SECTION_HEAD, "head"), (SECTION_SAY, "say"),
                             (SECTION_TAIL, "tail")):
            with self.subTest(marker=marker):
                raw = VALID.replace(marker, marker + marker)
                self.assert_invalid_in_both(raw, "duplicate_" + name + "_marker")

    def test_missing_markers_are_invalid_even_when_content_is_recovered(self):
        parsed = self.assert_invalid_in_both(
            VALID.replace(SECTION_SAY, "\n"), "missing_say_marker")
        self.assertEqual(parsed.say, SAY)
        self.assert_invalid_in_both(VALID.replace(SECTION_HEAD, ""), "missing_head_marker")
        self.assert_invalid_in_both(VALID.replace(SECTION_TAIL, ""), "missing_tail_marker")

    def test_plain_reply_remains_visible_but_is_not_a_protocol_sample(self):
        parsed = self.assert_invalid_in_both(SAY, "missing_head_marker")
        self.assertEqual(parsed.say, SAY)
        self.assertEqual(stream(SAY)[1], SAY)

    def test_text_before_head_breaks_the_internal_first_contract(self):
        self.assert_invalid_in_both("I already answered.\n" + VALID, "text_before_head")

    def test_leading_whitespace_is_not_an_early_answer(self):
        self.assertEqual(section_errors(" \n\t" + VALID), [])

    def test_loose_markers_are_invalid_not_tolerated_protocol(self):
        self.assert_invalid_in_both(VALID.replace(SECTION_HEAD, "<<<head>>>"),
                                    "invalid_section_marker")

    def test_explicit_persona_expression_may_omit_tail(self):
        raw = SECTION_HEAD + "baseline: It is 4.\ndisplace: none\n" + SECTION_SAY + SAY
        self.assertIn("missing_tail_marker", parse(raw).structural_errors)
        for parsed in (parse(raw, expression_only=True),
                       stream(raw, expression_only=True)[0]):
            self.assertEqual(parsed.structural_errors, [])
            self.assertEqual(parsed.say, SAY)
            self.assertFalse(parsed.tail_usable)

    def test_expression_stage_still_requires_head_before_say(self):
        raw = SECTION_SAY + SAY + SECTION_HEAD + "baseline: It is 4.\ndisplace: none"
        for parsed in (parse(raw, expression_only=True),
                       stream(raw, expression_only=True)[0]):
            self.assertIn("invalid_section_order", parsed.structural_errors)

    def test_structured_tool_path_does_not_require_text_markers(self):
        parsed = ParsedTurn(Head(need="check", confidence=0.8, direct="4",
                                 baseline=SAY), SAY, Tail())
        self.assertEqual(parsed.structural_errors, [])
        self.assertTrue(parsed.tail_usable)
        self.assertTrue(parsed.validity(protocol="tool")["valid"])

    def test_final_audit_does_not_delay_streaming_until_tail(self):
        parser = StreamParser()
        emitted = parser.feed(SECTION_HEAD + HEAD + SECTION_SAY + SAY)
        self.assertEqual("".join(emitted), SAY)
        self.assertEqual(parser.feed(SECTION_TAIL + TAIL), [])
        self.assertTrue(parser.finish().validity()["valid"])

    def test_audit_includes_chunks_after_output_was_blocked(self):
        parser = StreamParser()
        parser.feed(SECTION_HEAD + HEAD + SECTION_SAY + SAY + "<|internal|>")
        self.assertTrue(parser.output_blocked)
        parser.feed(SECTION_TAIL + TAIL + SECTION_HEAD + HEAD)
        parsed = parser.finish()
        self.assertIn("duplicate_head_marker", parsed.structural_errors)
        self.assertIn("invalid_section_order", parsed.structural_errors)


class TestOneDisplacement(unittest.TestCase):
    def test_distinct_kinds_are_rejected_without_selecting_a_winner(self):
        for value in ("降级 / 转移", "截断、悬置", "降级（转移主语）", "none / 降级"):
            with self.subTest(value=value):
                head = parse_head(HEAD.replace("displace: none", "displace: " + value))
                self.assertIn("multiple_displace", head.errors)
                self.assertEqual(head.displace, DISPLACE_NONE)

    def test_legal_single_kind_annotation_is_preserved(self):
        for value in ("降级（判断→问题）", "降级 (判断→问题)"):
            with self.subTest(value=value):
                head = parse_head(HEAD.replace("displace: none", "displace: " + value))
                self.assertEqual(head.displace, "降级")
                self.assertEqual(head.errors, [])

    def test_multiple_displacements_invalidate_batch_and_stream(self):
        raw = VALID.replace("displace: none", "displace: 降级 / 转移")
        for parsed in (parse(raw), stream(raw)[0]):
            self.assertIn("multiple_displace", parsed.validity()["reasons"])
            self.assertFalse(parsed.validity()["valid"])

    def test_arbitrary_text_containing_a_kind_is_not_a_valid_enum(self):
        head = parse_head(HEAD.replace("displace: none", "displace: 这次要降级然后随意写"))
        self.assertIn("invalid_displace", head.errors)
        self.assertEqual(head.displace, DISPLACE_NONE)


class TestMultilineReferences(unittest.TestCase):
    def test_direct_and_baseline_keep_conditions_in_batch_and_stream(self):
        direct = "The answer is 4 under these assumptions.\n  First, both inputs are 2.\nOtherwise ask for inputs."
        baseline = "It is 4 with those inputs.\nIf you meant other inputs, tell me."
        raw = (SECTION_HEAD + "need: check the sum\nconfidence: 0.8\ndirect: " + direct
               + "\nbaseline: " + baseline + "\ndisplace: none\n"
               + SECTION_SAY + SAY + SECTION_TAIL + TAIL)
        for parsed in (parse(raw), stream(raw)[0]):
            self.assertEqual(parsed.head.direct, direct)
            self.assertEqual(parsed.head.baseline, baseline)
            self.assertTrue(parsed.validity()["valid"])

    def test_reference_value_may_begin_on_the_next_line(self):
        raw = VALID.replace("direct: 4", "direct:\n  4, assuming integer arithmetic.")
        for parsed in (parse(raw), stream(raw)[0]):
            self.assertEqual(parsed.head.direct, "4, assuming integer arithmetic.")
            self.assertNotIn("missing_direct", parsed.head.errors)

    def test_empty_reference_does_not_swallow_following_field(self):
        raw = VALID.replace("direct: 4", "direct: \n\t")
        for parsed in (parse(raw), stream(raw)[0]):
            self.assertEqual(parsed.head.direct, "")
            self.assertEqual(parsed.head.baseline, SAY)
            self.assertIn("missing_direct", parsed.head.errors)

    def test_reordered_head_fields_remain_separate(self):
        head = parse_head("direct: First condition.\nSecond condition.\nCONFIDENCE：0.8\n"
                          "baseline: A question.\nA qualification.\nneed: clarify\ndisplace: none")
        self.assertEqual(head.direct, "First condition.\nSecond condition.")
        self.assertEqual(head.baseline, "A question.\nA qualification.")
        self.assertEqual(head.need, "clarify")
        self.assertEqual(head.confidence, 0.8)

    def test_multiline_reference_stops_at_tail_fields_or_markers(self):
        for boundary in ("pending: -", "settle: -", "delta:\n  trust: +1", SECTION_SAY):
            with self.subTest(boundary=boundary):
                head = parse_head("direct: First.\nSecond.\n" + boundary + "\nOther content.")
                self.assertEqual(head.direct, "First.\nSecond.")

    def test_multiline_fields_round_trip_through_formatted_history(self):
        head = Head(need="check", confidence=0.8, direct="4.\nAssuming the given inputs.",
                    baseline="It is 4.\nTell me if an input differs.")
        history = build_history_entry(SAY, head, Tail())
        for parsed in (parse(history), stream(history)[0]):
            self.assertEqual(parsed.head.direct, head.direct)
            self.assertEqual(parsed.head.baseline, head.baseline)
            self.assertTrue(parsed.validity()["valid"])

    def test_missing_say_does_not_expose_a_late_direct_continuation(self):
        raw = (SECTION_HEAD + "need: check\nconfidence: 0.8\nbaseline: It is 4.\n"
               "displace: none\ndirect: A private reference.\nA private condition.\n"
               + SECTION_TAIL + TAIL)
        for parsed in (parse(raw), stream(raw)[0]):
            self.assertEqual(parsed.say, "")
            self.assertEqual(parsed.head.direct, "A private reference.\nA private condition.")
            self.assertIn("missing_say_marker", parsed.structural_errors)


if __name__ == "__main__":
    unittest.main()
