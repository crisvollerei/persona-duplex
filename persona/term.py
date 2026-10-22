"""终端颜色。

为什么需要它：Windows 的 cmd.exe 默认不启用 VT 处理，
直接输出 ANSI 转义码会被当成普通文字打印出来 —— 用户看到的
`[90m`、`[36m` 就是这么来的。

判定顺序：
  1. 显式设置 PERSONA_COLOR=0/1 时听它的
  2. 输出被重定向到文件/管道时，关掉
  3. Windows 上尝试开启 VT 模式；开不了就关掉
  4. 其余情况按是否 tty 判断

全部失败也不会报错 —— 只是没有颜色而已，不该影响功能。
"""
from __future__ import annotations

import os
import sys

_CODES = {
    "dim": "90",
    "cyan": "36",
    "green": "32",
    "yellow": "33",
    "red": "31",
    "blue": "34",
    "bold": "1",
    "reset": "0",
}


def _enable_windows_vt() -> bool:
    """尝试给 Windows 控制台开启 ANSI 支持。"""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # 7 = STD_OUTPUT_HANDLE
        handle = kernel32.GetStdHandle(-11)
        if handle in (0, -1):
            return False
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _decide() -> bool:
    forced = os.environ.get("PERSONA_COLOR", "").strip().lower()
    if forced in ("0", "false", "no", "off"):
        return False
    if forced in ("1", "true", "yes", "on"):
        return True

    # 重定向到文件或管道时不要颜色，否则会污染日志
    if not sys.stdout.isatty():
        return False

    return _enable_windows_vt()


class _Palette:
    """可开关的调色板。

    `enabled` 在 import 时判定一次。如果后续发现环境变了（少见），
    调 refresh() 重判。
    """

    def __init__(self) -> None:
        self.enabled = _decide()

    def refresh(self) -> None:
        self.enabled = _decide()

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        prefix = "".join(f"\033[{_CODES.get(s, s)}m" for s in styles)
        return f"{prefix}{text}\033[0m"

    def raw(self, code: str) -> str:
        """需要自己拼前缀时用（比如逐字流式）。"""
        return f"\033[{_CODES.get(code, code)}m" if self.enabled else ""

    @property
    def reset(self) -> str:
        return "\033[0m" if self.enabled else ""


P = _Palette()


# 便捷函数：c.dim("...") / c.red("...", "bold")
def dim(s: str) -> str:
    return P(s, "dim")


def cyan(s: str) -> str:
    return P(s, "cyan")


def green(s: str) -> str:
    return P(s, "green")


def yellow(s: str) -> str:
    return P(s, "yellow")


def red(s: str) -> str:
    return P(s, "red")


def blue(s: str) -> str:
    return P(s, "blue")


def bold(s: str) -> str:
    return P(s, "bold")
