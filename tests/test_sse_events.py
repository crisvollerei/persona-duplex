"""Offline transport tests: actual adapter calls, synthetic HTTP response bytes."""
import json
import unittest
from unittest.mock import patch

from persona.llm import openai_stream


def event(value, *, ending="\n\n"):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ("data: " + text + ending).encode("utf-8")


def content(text="你好", reason=None, **metadata):
    return dict(choices=[dict(delta=dict(content=text), finish_reason=reason)], **metadata)


class Response:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        return iter(self.chunks)


class TestSSEEvents(unittest.TestCase):
    def adapter(self):
        return openai_stream("requested", api_key="test", base_url="https://example.invalid/v1")

    def call(self, chunks, fn=None):
        fn = fn or self.adapter()
        response = Response(chunks)
        with patch("persona.llm.urllib.request.urlopen", return_value=response) as http:
            output = list(fn([dict(role="user", content="问题")]))
        self.assertEqual(http.call_count, 1)
        body = json.loads(http.call_args.args[0].data)
        self.assertTrue(body["stream"])
        self.assertTrue(response.closed)
        return output, fn.observations.response, fn

    def test_usage_only_final_event_preserves_previous_finish(self):
        output, observed, _ = self.call([
            event(content(reason="stop", id="response-1", model="actual", system_fingerprint="fingerprint")),
            event(dict(id=None, model=None, system_fingerprint=None, choices=[],
                       usage=dict(prompt_tokens=4, completion_tokens=2, total_tokens=6))),
            event("[DONE]"),
        ])
        self.assertEqual(output, ["你好"])
        self.assertEqual(observed, dict(id="response-1", model="actual", system_fingerprint="fingerprint",
                                        finish_reason="stop", usage=dict(prompt_tokens=4, completion_tokens=2, total_tokens=6)))

    def test_usage_is_retained_when_choices_are_missing(self):
        _, observed, _ = self.call([
            event(content(reason="stop")), event(dict(model="actual", usage=dict(total_tokens=6))), event("[DONE]")])
        self.assertEqual(observed["usage"], dict(total_tokens=6))
        self.assertEqual(observed["model"], "actual")
        self.assertEqual(observed["finish_reason"], "stop")

    def test_partial_usage_merges_without_erasing_values_with_null(self):
        _, observed, _ = self.call([
            event(content(reason="stop", usage=dict(prompt_tokens=4, prompt_tokens_details=dict(cached_tokens=2)))),
            event(dict(choices=[], usage=dict(prompt_tokens=None, completion_tokens=2,
                total_tokens=6, prompt_tokens_details=dict(cached_tokens=None)))),
            event(dict(choices=[], usage=None)), event("[DONE]")])
        self.assertEqual(observed["usage"], dict(prompt_tokens=4, completion_tokens=2, total_tokens=6,
                                                prompt_tokens_details=dict(cached_tokens=2)))

    def test_explicit_zero_usage_is_observed_not_dropped(self):
        _, observed, _ = self.call([
            event(content(reason="stop", usage=dict(total_tokens=9))),
            event(dict(choices=[], usage=dict(total_tokens=0, completion_tokens=0))), event("[DONE]")])
        self.assertEqual(observed["usage"]["total_tokens"], 0)
        self.assertEqual(observed["usage"]["completion_tokens"], 0)

    def test_missing_usage_is_not_fabricated(self):
        _, observed, _ = self.call([event(content(reason="stop")), event("[DONE]")])
        self.assertIsNone(observed.get("usage"))
        self.assertNotIn("usage", observed)

    def test_observations_reset_between_calls(self):
        fn = self.adapter()
        self.call([event(content(reason="stop", usage=dict(total_tokens=9))), event("[DONE]")], fn)
        _, observed, _ = self.call([event(content(reason="stop")), event("[DONE]")], fn)
        self.assertNotIn("usage", observed)

    def test_usage_only_done_stream_still_exposes_metadata(self):
        output, observed, _ = self.call([event(dict(id="only-metadata", choices=[], usage=dict(total_tokens=0))),
                                          event("[DONE]")])
        self.assertEqual(output, [])
        self.assertEqual(observed["id"], "only-metadata")
        self.assertEqual(observed["usage"]["total_tokens"], 0)
        self.assertNotIn("finish_reason", observed)

    def test_legal_multiple_data_lines_form_one_json_event(self):
        raw = ('data: {"id":"multi","model":"actual","choices":[{"delta":{"content":"你好"},\n'
               'data: "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n').encode()
        output, observed, _ = self.call(raw.splitlines(keepends=True))
        self.assertEqual(output, ["你好"])
        self.assertEqual(observed["id"], "multi")
        self.assertEqual(observed["finish_reason"], "stop")

    def test_legacy_single_json_lines_without_empty_separators_remain_compatible(self):
        raw = (event(content(reason="stop"), ending="\n") +
               event(dict(choices=[], usage=dict(total_tokens=6)), ending="\n") +
               event("[DONE]", ending=""))
        output, observed, _ = self.call(raw.splitlines(keepends=True))
        self.assertEqual(output, ["你好"])
        self.assertEqual(observed["usage"]["total_tokens"], 6)
        self.assertEqual(observed["finish_reason"], "stop")

    def test_utf8_bom_and_arbitrary_byte_boundaries_are_preserved(self):
        raw = b"\xef\xbb\xbf" + event(content("你好，世界。", reason="stop")) + event("[DONE]")
        output, _, _ = self.call([raw[i:i+1] for i in range(len(raw))])
        self.assertEqual(output, ["你好，世界。"])

    def test_crlf_split_between_bytes_and_cr_only_lines(self):
        for ending in ("\r\n\r\n", "\r\r"):
            with self.subTest(ending=repr(ending)):
                raw = event(content(reason="stop"), ending=ending) + event("[DONE]", ending=ending)
                output, _, _ = self.call([raw[i:i+1] for i in range(len(raw))])
                self.assertEqual(output, ["你好"])

    def test_comments_and_non_data_fields_do_not_become_response_metadata(self):
        raw = (b": keepalive\nretry: 2000\nid: transport-id\nevent: update\n" +
               event(content(reason="stop", id="completion-id")) + event("[DONE]"))
        output, observed, _ = self.call([raw])
        self.assertEqual(output, ["你好"])
        self.assertEqual(observed["id"], "completion-id")

    def test_eof_without_newline_accepts_explicit_stop(self):
        output, observed, _ = self.call([event(content(reason="stop"), ending="")])
        self.assertEqual(output, ["你好"])
        self.assertEqual(observed["finish_reason"], "stop")

    def test_done_without_newline_and_without_stop_is_compatible(self):
        output, observed, _ = self.call([event(content()), event("[DONE]", ending="")])
        self.assertEqual(output, ["你好"])
        self.assertIsNone(observed.get("finish_reason"))

    def test_eof_without_completion_keeps_partial_and_raises_without_retry(self):
        fn = self.adapter()
        response = Response([event(content("已交付"))])
        with patch("persona.llm.urllib.request.urlopen", return_value=response) as http:
            iterator = fn([])
            self.assertEqual(next(iterator), "已交付")
            with self.assertRaisesRegex(RuntimeError, "中断"):
                next(iterator)
        self.assertEqual(http.call_count, 1)
        self.assertTrue(response.closed)

    def test_length_finish_fails_after_partial_delivery(self):
        fn = self.adapter()
        response = Response([event(content("部分", reason="length", usage=dict(total_tokens=4))), event("[DONE]")])
        with patch("persona.llm.urllib.request.urlopen", return_value=response) as http:
            iterator = fn([])
            self.assertEqual(next(iterator), "部分")
            with self.assertRaisesRegex(RuntimeError, "length"):
                next(iterator)
        self.assertEqual(http.call_count, 1)
        self.assertEqual(fn.observations.response["finish_reason"], "length")
        self.assertEqual(fn.observations.response["usage"]["total_tokens"], 4)

    def test_error_event_after_stop_is_not_hidden_and_metadata_survives(self):
        fn = self.adapter()
        response = Response([event(content("已交付", reason="stop", model="actual")),
                             event(dict(error=dict(message="upstream failure"), choices=[], usage=dict(total_tokens=8)))])
        with patch("persona.llm.urllib.request.urlopen", return_value=response) as http:
            iterator = fn([])
            self.assertEqual(next(iterator), "已交付")
            with self.assertRaisesRegex(RuntimeError, "服务错误"):
                next(iterator)
        self.assertEqual(http.call_count, 1)
        self.assertEqual(fn.observations.response["finish_reason"], "stop")
        self.assertEqual(fn.observations.response["usage"]["total_tokens"], 8)

    def test_malformed_event_is_not_mixed_into_next_event(self):
        output, _, _ = self.call([b"data: {bad json}\n\n", event(content(reason="stop")), event("[DONE]")])
        self.assertEqual(output, ["你好"])

    def test_other_choices_do_not_expand_supported_output(self):
        payload = dict(choices=[dict(delta=dict(content="第一条"), finish_reason="stop"),
                                dict(delta=dict(content="第二条"), finish_reason="stop")])
        output, _, _ = self.call([event(payload), event("[DONE]")])
        self.assertEqual(output, ["第一条"])

    def test_usage_after_stop_is_processed_before_plain_eof(self):
        output, observed, _ = self.call([
            event(content(reason="stop")), event(dict(choices=[], usage=dict(total_tokens=6)), ending="")])
        self.assertEqual(output, ["你好"])
        self.assertEqual(observed["usage"]["total_tokens"], 6)
        self.assertEqual(observed["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
