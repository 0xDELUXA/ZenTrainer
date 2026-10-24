"""How zt talks: rich for what you read, questionary for what you answer.

Off a terminal (piped, in a script) rich drops the styling by itself, and a question stops zt
rather than take a default: a default can be "install".
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from typing import NoReturn

from rich.console import Console
from rich.live import Live
from rich.text import Text

from . import repo

console = Console(highlight=False)
errors = Console(stderr=True, highlight=False)

ACCENT = "#e8ff3a"  # the app's yellow
SOFT = "#c4aae6"  # lavender


def say(msg: str) -> None:
    console.print(Text.assemble(("› ", "magenta"), msg))


def ok(msg: str) -> None:
    console.print(Text.assemble(("✓ ", "green"), msg))


def warn(msg: str) -> None:
    console.print(Text.assemble(("! ", "yellow"), msg))


def die(msg: str) -> NoReturn:
    errors.print(Text.assemble(("✗ ", "red"), msg))
    sys.exit(1)


def run(cmd: list, **kw) -> None:
    """Run a command in the open (its own output shows), saying what it is first."""
    console.print(Text("$ " + " ".join(str(c) for c in cmd), style="dim"))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def open_browser(url: str) -> None:
    """Open a page without the browser writing into this terminal (Chromium prints zygote and
    GPU noise to whatever started it, over the menu or the monitor)."""
    import shutil
    import webbrowser

    opener = shutil.which("xdg-open") if sys.platform.startswith("linux") else None
    if opener:
        subprocess.Popen([opener, url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return
    # elsewhere webbrowser does it, with this process's output pointed away while it starts one
    saved = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        webbrowser.open(url)
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


# ---- questions ----------------------------------------------------------------------------------

def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _style():
    import questionary

    return questionary.Style([("qmark", "fg:magenta bold"), ("question", "bold"), ("answer", f"fg:{ACCENT}"),
                              ("pointer", f"fg:{ACCENT} bold"), ("highlighted", f"fg:{ACCENT} bold noreverse"),
                              ("selected", f"fg:{ACCENT} noreverse"), ("instruction", "fg:#888888")])


def _answer(q):
    if not interactive():
        die("no terminal to answer in; stopping. Run it in a terminal, or use the commands (zt --help)")
    ans = q.ask()
    if ans is None:  # Ctrl-C
        raise KeyboardInterrupt
    return ans


def ask(question: str, default: str = "") -> str:
    import questionary

    return _answer(questionary.text(question, default=default, style=_style())).strip()


def confirm(question: str, default: bool = True) -> bool:
    import questionary

    return _answer(questionary.confirm(question, default=default, style=_style()))


def choose(question: str, choices: list[tuple[str, str]], default: str | None = None) -> str:
    """One of (title, value), picked with the arrow keys; returns the value."""
    import questionary

    from questionary.prompts.common import InquirerControl

    items = [questionary.Choice(title, value=value) for title, value in choices]
    q = questionary.select(question, choices=items, style=_style())
    # start on the default without passing it as `default`: questionary would draw that item as
    # "selected" for good, so it'd look picked wherever the pointer went
    start = next((i for i, c in enumerate(items) if c.value == default), 0)
    for control in q.application.layout.find_all_controls():
        if isinstance(control, InquirerControl):
            control.pointed_at = start
    return _answer(q)


def checkboxes(question: str, options: list[str]) -> list[str]:
    import questionary

    return _answer(questionary.checkbox(question, choices=options, style=_style()))


def folder(question: str) -> str:
    """A folder path, completed as you type; "" to skip."""
    import questionary

    return _answer(questionary.path(question, only_directories=True, style=_style())).strip()


# ---- the logo -----------------------------------------------------------------------------------

LOTUS = [
    "                  ▄▀▄",
    "             ▄▀▄ █   █ ▄▀▄",
    "            █   ▀▄ █ ▄▀   █",
    "             ▀▄▄  ▀█▀  ▄▄▀",
    "           ~~~~ ▀▀▀▀▀▀▀ ~~~~",
]
WORDMARK = [
    "▀█ █▀▀ █▄ █ ▀█▀ █▀█ ▄▀█ █ █▄ █ █▀▀ █▀█",
    "█▄ ██▄ █ ▀█  █  █▀▄ █▀█ █ █ ▀█ ██▄ █▀▄",
]
INDENT = " " * 8


def _logo() -> Text:
    t = Text()
    for line in LOTUS:
        t.append(INDENT + line + "\n", style=SOFT)
    for line in WORDMARK:
        t.append(INDENT + line + "\n", style=ACCENT)
    return t


def intro() -> None:
    """The logo, decoded out of falling glyphs: about a second, and only on a terminal."""
    if not console.is_terminal or repo.local().get("intro") is False or os.environ.get("ZT_NO_INTRO"):
        console.print(_logo())
        return
    rows = [(INDENT + l, SOFT) for l in LOTUS] + [(INDENT + w, ACCENT) for w in WORDMARK]
    cells = [(r, c) for r, (line, _) in enumerate(rows) for c, ch in enumerate(line) if ch != " "]
    random.shuffle(cells)
    glyphs = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄ0123456789"
    frames = 24

    def frame(n: int) -> Text:
        shown = set(cells[: len(cells) * n // frames])
        t = Text()
        for r, (line, style) in enumerate(rows):
            for c, ch in enumerate(line):
                if ch == " ":
                    t.append(" ")
                elif (r, c) in shown:
                    t.append(ch, style=style)
                elif random.random() < 0.5:
                    t.append(random.choice(glyphs), style="#50c878")
                else:
                    t.append("·", style="dim")
            t.append("\n")
        return t

    with Live(frame(0), console=console, refresh_per_second=30, transient=False) as live:
        for n in range(1, frames + 1):
            time.sleep(0.04)
            live.update(frame(n))
