"""Minimal arrow-key TUI (select / checkbox / text input).

Zero non-stdlib imports — loads in microseconds, unlike questionary/prompt_toolkit.
Works on any Unix terminal (including WSL).  Falls back to numbered-list input when
stdin is not a TTY (e.g. piped or in CI).
"""

from __future__ import annotations

import sys
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_RESET    = "\x1b[0m"
_BOLD     = "\x1b[1m"
_DIM      = "\x1b[2m"
_CYAN     = "\x1b[36m"
_GREEN    = "\x1b[32m"
_RED      = "\x1b[31m"

_CLEAR_LINE  = "\x1b[2K\r"   # erase whole line, return to column 0
_CURSOR_UP   = "\x1b[1A"     # move cursor up one line

_KEY_UP    = "\x1b[A"
_KEY_DOWN  = "\x1b[B"
_KEY_ENTER = ("\r", "\n")
_KEY_ABORT = ("\x03", "\x04", "\x1b")  # Ctrl-C, Ctrl-D, ESC


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_key() -> str:
    """Read one keypress (or escape sequence) from stdin in raw mode."""
    import tty
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Try to consume the rest of an escape sequence (e.g. "[A")
            nxt = sys.stdin.read(1)
            if nxt == "[":
                ch += "[" + sys.stdin.read(1)
            else:
                ch += nxt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _erase_lines(n: int) -> None:
    """Erase n rendered lines and leave the cursor at the start of the first one.

    After print() the cursor is on the blank line *below* the last rendered
    line.  We need one cursor-up per rendered line to get back to the top.
    """
    for _ in range(n):
        sys.stdout.write(_CURSOR_UP + _CLEAR_LINE)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select(prompt: str, choices: list[str]) -> Optional[str]:
    """Single-select arrow-key menu.  Returns the chosen string or None."""
    if not choices:
        return None

    n = len(choices)

    if not _is_interactive():
        # Non-TTY fallback: numbered list + input()
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        while True:
            try:
                raw = input(f"{prompt} (1-{n}): ").strip()
                idx = int(raw) - 1
                if 0 <= idx < n:
                    return choices[idx]
            except (ValueError, EOFError, KeyboardInterrupt):
                return None

    idx = 0
    n_lines = n + 2  # prompt + choices + hint

    def _render(cur: int) -> None:
        print(f"{_BOLD}?{_RESET} {prompt}")
        for i, c in enumerate(choices):
            arrow = f"{_CYAN}\u00bb{_RESET}" if i == cur else " "
            print(f"  {arrow} {c}")
        print(f"{_DIM}  [\u2191/\u2193 move  Enter select  Ctrl-C cancel]{_RESET}", flush=True)

    _render(idx)
    try:
        while True:
            key = _read_key()
            if key in _KEY_ENTER:
                _erase_lines(n_lines)
                print(f"{_BOLD}?{_RESET} {prompt} {_CYAN}{choices[idx]}{_RESET}")
                return choices[idx]
            elif key == _KEY_UP:
                idx = (idx - 1) % n
            elif key == _KEY_DOWN:
                idx = (idx + 1) % n
            elif key in _KEY_ABORT:
                raise KeyboardInterrupt
            else:
                continue
            _erase_lines(n_lines)
            _render(idx)
    except KeyboardInterrupt:
        _erase_lines(n_lines)
        print(f"{_RED}Cancelled.{_RESET}")
        return None


def checkbox(prompt: str, choices: list[str]) -> list[str]:
    """Multi-select checkbox menu.  Returns list of selected strings."""
    if not choices:
        return []

    n = len(choices)

    if not _is_interactive():
        print(f"{prompt}")
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        raw = input("Enter numbers separated by commas (or blank for none): ").strip()
        if not raw:
            return []
        result = []
        for part in raw.split(","):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < n:
                    result.append(choices[idx])
            except ValueError:
                pass
        return result

    selected = [False] * n
    idx = 0
    n_lines = n + 2

    def _render(cur: int) -> None:
        print(f"{_BOLD}?{_RESET} {prompt}")
        for i, c in enumerate(choices):
            mark   = f"{_GREEN}\u2714{_RESET}" if selected[i] else " "
            arrow  = f"{_CYAN}\u00bb{_RESET}" if i == cur else " "
            print(f"  {arrow} [{mark}] {c}")
        print(f"{_DIM}  [\u2191/\u2193 move  Space toggle  Enter confirm  Ctrl-C cancel]{_RESET}", flush=True)

    _render(idx)
    try:
        while True:
            key = _read_key()
            if key in _KEY_ENTER:
                result = [choices[i] for i in range(n) if selected[i]]
                _erase_lines(n_lines)
                summary = (f"{_CYAN}{len(result)} selected{_RESET}" if result
                           else f"{_DIM}none selected{_RESET}")
                print(f"{_BOLD}?{_RESET} {prompt} {summary}")
                return result
            elif key == " ":
                selected[idx] = not selected[idx]
            elif key == _KEY_UP:
                idx = (idx - 1) % n
            elif key == _KEY_DOWN:
                idx = (idx + 1) % n
            elif key in _KEY_ABORT:
                raise KeyboardInterrupt
            else:
                continue
            _erase_lines(n_lines)
            _render(idx)
    except KeyboardInterrupt:
        _erase_lines(n_lines)
        print(f"{_RED}Cancelled.{_RESET}")
        return []


def text(prompt: str, default: str = "") -> str:
    """Single-line text input with an optional default."""
    hint = f" [{_DIM}{default}{_RESET}]" if default else ""
    try:
        val = input(f"{_BOLD}?{_RESET} {prompt}{hint}: ").strip()
        return val or default
    except (EOFError, KeyboardInterrupt):
        return default
