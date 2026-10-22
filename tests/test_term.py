"""终端与模型选择测试。

两个都是「用户一跑就撞上」的问题：
  1. Windows cmd 不认 ANSI 码，用户看到 `[90m` 这种垃圾
  2. 默认模型写死 gpt-4o-mini，换个端点直接 400
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persona import term                                     # noqa: E402
from persona.cli import _default_model                       # noqa: E402


class TestColorDetection(unittest.TestCase):
    def test_forced_off(self):
        with mock.patch.dict(os.environ, {"PERSONA_COLOR": "0"}):
            p = term._Palette()
            self.assertFalse(p.enabled)
            self.assertEqual(p("x", "red"), "x")

    def test_forced_on(self):
        with mock.patch.dict(os.environ, {"PERSONA_COLOR": "1"}):
            p = term._Palette()
            self.assertTrue(p.enabled)
            self.assertIn("\033[", p("x", "red"))

    def test_no_tty_disables(self):
        """输出被重定向时必须关掉，否则污染日志。"""
        with mock.patch.dict(os.environ, {"PERSONA_COLOR": ""}), \
             mock.patch.object(sys.stdout, "isatty", return_value=False):
            self.assertFalse(term._Palette().enabled)

    def test_unknown_style_passthrough(self):
        with mock.patch.dict(os.environ, {"PERSONA_COLOR": "1"}):
            p = term._Palette()
            self.assertIn("99", p("x", "99"))

    def test_helpers_never_crash(self):
        for fn in (term.dim, term.cyan, term.green, term.yellow,
                   term.red, term.blue, term.bold):
            self.assertIsInstance(fn("test"), str)


class TestNoAnsiLeak(unittest.TestCase):
    """真的跑一遍 CLI，确认管道输出里没有转义码。"""

    def test_piped_output_is_plain(self):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PERSONA_COLOR"] = ""
        out = subprocess.run(
            [sys.executable, "-m", "persona", "demo", "-d",
             str(ROOT / "_tmp_ansi_test")],
            capture_output=True, text=True, encoding="utf-8",
            cwd=ROOT, env=env, timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        self.assertNotIn("\033[", out.stdout, "管道输出里出现了 ANSI 转义码")
        self.assertNotIn("[90m", out.stdout, "转义码没被解析成文字了")
        import shutil
        shutil.rmtree(ROOT / "_tmp_ansi_test", ignore_errors=True)


class TestModelSelection(unittest.TestCase):
    def test_explicit_env_wins(self):
        with mock.patch.dict(os.environ, {"PERSONA_MODEL": "my-model",
                                          "PERSONA_BASE_URL": "https://api.deepseek.com/v1"}):
            self.assertEqual(_default_model(), "my-model")

    def test_deepseek_endpoint(self):
        with mock.patch.dict(os.environ, {"PERSONA_MODEL": "",
                                          "PERSONA_BASE_URL": "https://api.deepseek.com/v1"}):
            self.assertEqual(_default_model(), "deepseek-chat")

    def test_openai_fallback(self):
        with mock.patch.dict(os.environ, {"PERSONA_MODEL": "",
                                          "PERSONA_BASE_URL": "https://api.openai.com/v1"}):
            self.assertEqual(_default_model(), "gpt-4o-mini")

    def test_no_env_at_all(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("PERSONA_MODEL", "PERSONA_BASE_URL")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(_default_model(), "gpt-4o-mini")


class TestErrorMessage(unittest.TestCase):
    def test_http_error_includes_detail(self):
        """400 时必须把服务端的说明带出来，否则用户不知道错在哪。"""
        from persona.llm import openai_compatible
        import urllib.error

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                __import__("io").BytesIO(
                    b'{"error":{"message":"model not found"}}')
            )

        fn = openai_compatible("bad-model", api_key="k")
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaises(RuntimeError) as ctx:
                fn([{"role": "user", "content": "hi"}])
        self.assertIn("400", str(ctx.exception))
        self.assertIn("model not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
