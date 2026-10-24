"""zt monitor: a live dashboard of ZenTrainer in the terminal (Textual).

The queue and recent runs on the left; the selected run (the training one, until you pick
another) on the right, with its progress, a loss sparkline and the GPU's memory. It only talks
to the HTTP API, polling every couple of seconds, so it watches a ZenTrainer on this machine or,
through a tunnel, a headless one elsewhere. Keys: pause/resume, stop (asks first), save a
checkpoint, open the web app.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from rich.console import Group
from rich.progress_bar import ProgressBar as Bar
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Label, Sparkline, Static

from .api import STATUS_STYLE, TERMINAL, new_rows
from .ui import ACCENT, open_browser

POLL = 2.0  # seconds
LOSS_POINTS = 240  # the sparkline's width in steps


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f}"


def _clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {seconds % 60:02d}s"


class Confirm(ModalScreen[bool]):
    """Yes or no, over the dashboard."""

    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel")]

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.question)
            with Horizontal(id="buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class Monitor(App):
    TITLE = "ZenTrainer"
    CSS = f"""
    Screen {{ layout: vertical; }}
    #body {{ height: 1fr; }}
    #left {{ width: 3fr; min-width: 40; }}
    #right {{ width: 4fr; padding: 0 1; }}
    #queue {{ height: auto; padding: 0 1; border: round $panel; }}
    #runs {{ height: 1fr; border: round $panel; }}
    #run {{ height: auto; border: round $panel; padding: 0 1; }}
    #loss {{ height: 6; border: round $panel; padding: 0 1; }}
    #loss > .sparkline--max-color {{ color: {ACCENT}; }}
    #system {{ height: auto; border: round $panel; padding: 0 1; }}
    #note {{ height: 1; padding: 0 1; color: $text-muted; }}
    Confirm {{ align: center middle; }}
    #dialog {{ width: 60; height: auto; padding: 1 2; border: thick $warning; background: $surface; }}
    #buttons {{ height: auto; margin-top: 1; align: center middle; }}
    #buttons Button {{ margin: 0 1; }}
    """
    BINDINGS = [
        Binding("p", "pause", "Pause/resume"),
        Binding("s", "stop", "Stop"),
        Binding("c", "save", "Save checkpoint"),
        Binding("o", "open", "Open web app"),
        Binding("t", "follow_training", "Back to the training run"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, port: int) -> None:
        super().__init__()
        self.port = port
        self.picked: str | None = None  # a run chosen in the table; None: whichever is training
        self.selected: str | None = None
        self.detail: dict | None = None
        self.metrics: dict[str, list[dict]] = {}  # run id -> rows so far (the tail is enough)
        self.counts: dict[str, int] = {}

    # ---- the API, from a worker thread: errors become None, never an exit -----------------------
    def _req(self, method: str, path: str, body=None):
        url = f"http://127.0.0.1:{self.port}/zentrainer/api{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            try:
                return {"error": json.load(e).get("error") or e.reason}
            except Exception:
                return {"error": str(e.reason)}
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static(id="queue")
                yield DataTable(id="runs", cursor_type="row", zebra_stripes=True)
            with Vertical(id="right"):
                yield Static(id="run")
                yield Sparkline([], id="loss", summary_function=max)
                yield Static(id="system")
        yield Static(id="note")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"127.0.0.1:{self.port}"
        t = self.query_one("#runs", DataTable)
        t.add_columns("status", "steps", "name", "run")
        self.query_one("#queue", Static).border_title = "Queue"
        t.border_title = "Runs (Enter to watch one)"
        self.query_one("#run", Static).border_title = "Run"
        self.query_one("#loss", Sparkline).border_title = "Loss (recent steps)"
        self.query_one("#system", Static).border_title = "Memory"
        self.poll()
        self.set_interval(POLL, self.poll)

    # ---- polling ---------------------------------------------------------------------------------
    @work(thread=True, exclusive=True)
    def poll(self) -> None:
        queue = self._req("GET", "/queue")
        if queue is None:
            self.call_from_thread(self._offline)
            return
        runs = (self._req("GET", "/runs") or {}).get("runs", [])
        system = self._req("GET", "/system") or {}
        current = (queue.get("current") or {}).get("run_id")
        selected = self.picked or current or (runs[0]["run_id"] if runs else None)
        detail = None
        if selected:
            since = self.counts.get(selected)
            detail = self._req("GET", f"/runs/{selected}?since={since or 0}")
            if detail and "run" in detail:
                rows, count = new_rows(detail, since)
                kept = (self.metrics.get(selected, []) if since is not None else []) + rows
                self.metrics[selected] = kept[-LOSS_POINTS * 2:]
                self.counts[selected] = count
        self.call_from_thread(self._show, queue, runs, system, selected, detail)

    def _offline(self) -> None:
        self.query_one("#note", Static).update(Text(f"no ZenTrainer answering on :{self.port}: retrying…", style="red"))

    def _show(self, queue: dict, runs: list, system: dict, selected: str | None, detail: dict | None) -> None:
        self.selected = selected
        if detail and "run" in detail:
            self.detail = detail
        if "no ZenTrainer answering" in str(self.query_one("#note", Static).render()):
            self.query_one("#note", Static).update("")
        self._show_queue(queue)
        self._show_runs(runs)
        self._show_run()
        self._show_system(system)

    def _show_queue(self, q: dict) -> None:
        t = Text()
        t.append("paused" if q.get("paused") else "running", style="yellow" if q.get("paused") else "green")
        cur = q.get("current")
        t.append("  ·  " + (f"training {cur.get('name') or cur['run_id']}" if cur else "nothing training"))
        for r in q.get("items", []):
            t.append(f"\n  next: {r.get('name') or r['run_id']}", style="cyan")
        self.query_one("#queue", Static).update(t)

    def _show_runs(self, runs: list) -> None:
        table = self.query_one("#runs", DataTable)
        # the cursor stays on the same run across refreshes (rows move as runs start and end);
        # at first it's on the run being watched
        if table.row_count:
            keep = str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
        else:
            keep = self.selected
        table.clear()
        shown = runs[:50]
        for r in shown:
            status = r.get("status", "?")
            table.add_row(Text(status, style=STATUS_STYLE.get(status, "")), str(r.get("final_step") or ""),
                          r.get("name", ""), Text(r["run_id"], style="dim"), key=r["run_id"])
        ids = [r["run_id"] for r in shown]
        if keep in ids:
            table.move_cursor(row=ids.index(keep), animate=False)

    def _show_run(self) -> None:
        d, panel = self.detail, self.query_one("#run", Static)
        if not d or d["run"]["run_id"] != self.selected:
            panel.update(Text("no runs yet" if not self.selected else "loading…", style="dim"))
            return
        run, spec = d["run"], d.get("spec") or {}
        rows = self.metrics.get(self.selected, [])
        status = run.get("status", "?")
        total = spec.get("optim", {}).get("steps") or 0
        last = rows[-1] if rows else {}
        step = last.get("step") or run.get("final_step") or 0
        rate = last.get("rate") or last.get("it_s") or 0
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim")
        grid.add_column()
        head = Text(run.get("name") or run["run_id"], style="bold")
        head.append(f"  {status}", style=STATUS_STYLE.get(status, ""))
        if run.get("message"):
            head.append(f"  {run['message']}", style="dim")
        grid.add_row("step", f"{step:,} / {total:,}" if total else f"{step:,}")
        if last:
            grid.add_row("loss", f"{last.get('loss') or 0:.4f}   lr {last.get('lr') or 0:.2e}")
        if rate:
            eta = (total - step) / rate if total and status not in TERMINAL else 0
            grid.add_row("speed", f"{1 / rate:.2f} s/step" + (f"   about {_clock(eta)} left" if eta else ""))
        val = (d.get("validation") or [None])[-1]
        if val:
            grid.add_row("validation", f"{val.get('val_loss', 0):.4f} at step {val.get('step')}")
        grid.add_row("samples", str(len(d.get("samples") or [])))
        ck = (d.get("checkpoints") or [None])[-1]
        if ck:
            grid.add_row("checkpoint", f"step {ck.get('step')}  {ck.get('lora_name') or ck.get('path', '')}")
        bar = Bar(total=total or None, completed=step, width=None, complete_style=ACCENT, finished_style="green")
        panel.update(Group(head, bar, grid))
        losses = [m.get("loss") or 0 for m in rows][-LOSS_POINTS:]
        smooth, out = None, []
        for x in losses:  # a little smoothing: the raw loss jumps step to step
            smooth = x if smooth is None else smooth * 0.9 + x * 0.1
            out.append(smooth)
        self.query_one("#loss", Sparkline).data = out

    def _show_system(self, system: dict) -> None:
        s = (system.get("samples") or [None])[-1]
        if not s:
            self.query_one("#system", Static).update(Text("no readings", style="dim"))
            return
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim", width=5)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")
        for name, used, total in (("VRAM", s.get("vram_used"), s.get("vram_total")), ("RAM", s.get("ram_used"), s.get("ram_total"))):
            if used is not None and total:
                grid.add_row(name, Bar(total=total, completed=used, complete_style=ACCENT), f"{_gb(used)} / {_gb(total)} GB")
        self.query_one("#system", Static).update(grid)

    # ---- picking a run, and acting on it ----------------------------------------------------------
    @on(DataTable.RowSelected)
    def pick(self, event: DataTable.RowSelected) -> None:
        self.picked = str(event.row_key.value)
        self.detail = None
        self.poll()

    def action_follow_training(self) -> None:
        self.picked = None
        self.poll()

    def _note(self, text: str, style: str = "") -> None:
        self.query_one("#note", Static).update(Text(text, style=style))

    @work(thread=True)
    def _command(self, run_id: str, command: str) -> None:
        r = self._req("POST", f"/runs/{run_id}/control", {"command": command})
        msg = "no answer from ZenTrainer" if r is None else r.get("error") or f"{command}: sent to {run_id}"
        self.call_from_thread(self._note, msg, "red" if r is None or "error" in (r or {}) else "green")
        self.call_from_thread(self.poll)

    def _live_run(self) -> tuple[str, str] | None:
        if not self.detail or self.detail["run"]["run_id"] != self.selected:
            return None
        status = self.detail["run"].get("status")
        if status in TERMINAL or status == "queued":
            self._note(f"{self.selected} isn't training ({status})", "yellow")
            return None
        return self.selected, status

    def action_pause(self) -> None:
        live = self._live_run()
        if live:
            self._command(live[0], "resume" if live[1] == "paused" else "pause")

    def action_save(self) -> None:
        live = self._live_run()
        if live:
            self._command(live[0], "save")

    def action_stop(self) -> None:
        live = self._live_run()
        if not live:
            return

        def done(yes: bool | None) -> None:
            if yes:
                self._command(live[0], "stop")

        self.push_screen(Confirm(f"Stop {self.detail['run'].get('name') or live[0]}? It saves its state first, "
                                 "so it can resume later."), done)

    def action_open(self) -> None:
        open_browser(f"http://127.0.0.1:{self.port}/zentrainer/" + (f"runs/{self.selected}" if self.selected else ""))


def run(port: int) -> None:
    Monitor(port).run()
