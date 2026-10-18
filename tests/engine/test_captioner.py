"""Caption jobs: which images they take, and that undo puts back exactly what they changed."""

from pathlib import Path

import pytest
from PIL import Image

from zentrainer import captioner, datasets
from zentrainer.harness.data import scan


@pytest.fixture
def ds(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    root.mkdir()
    for n in ("a", "b", "c"):
        Image.new("RGB", (64, 32)).save(root / f"{n}.png")
    (root / "a.txt").write_text("hand written a\n")
    (root / "b.nl.txt").write_text("old nl b\n")
    monkeypatch.setattr(captioner, "_file", lambda: tmp_path / "jobs.json")
    monkeypatch.setattr(captioner, "_dataset_root", lambda job: root)
    monkeypatch.setattr(captioner, "_jobs", {})
    monkeypatch.setattr(captioner, "_loaded", True)
    monkeypatch.setattr(captioner.events, "emit", lambda *a, **k: None)
    return root


def job(**kw):
    base = dict(dataset="ds", set_name="default", clip_name="x", clip_type="y", prompt="p")
    return captioner.create(**{**base, **kw})


def names(items):
    return sorted(it.path.name for it in items)


def test_modes_pick_images(ds):
    items = scan(ds, warn=False)
    assert names(captioner.todo(job(mode="missing"), items)) == ["b.png", "c.png"]
    assert names(captioner.todo(job(mode="missing", set_name="nl"), items)) == ["a.png", "c.png"]
    assert names(captioner.todo(job(mode="all"), items)) == ["a.png", "b.png", "c.png"]
    assert names(captioner.todo(job(mode="selected", paths=["c.png"]), items)) == ["c.png"]
    assert len(captioner.todo(job(mode="all", preview=2), items)) == 2


def write(j, ds, rel, text):
    """What run() does for one image."""
    from zentrainer.harness.data import caption_path

    target = caption_path(ds / rel, j.set_name)
    if target.is_file():
        j.previous[rel] = target.read_text()
    target.write_text(text + "\n")
    j.written[rel] = text


def test_undo_restores_and_removes(ds):
    j = job(mode="all")
    write(j, ds, "a.png", "auto a")  # replaced a hand-written caption
    write(j, ds, "c.png", "auto c")  # a new one
    j.status = "done"
    assert captioner.undo(j) == 2
    assert (ds / "a.txt").read_text() == "hand written a\n"
    assert not (ds / "c.txt").exists()
    assert j.status == "undone"


def test_undo_keeps_later_edits(ds):
    j = job(mode="all")
    write(j, ds, "a.png", "auto a")
    write(j, ds, "c.png", "auto c")
    (ds / "c.txt").write_text("fixed by hand\n")  # edited after the job
    j.status = "done"
    assert captioner.undo(j) == 1
    assert (ds / "a.txt").read_text() == "hand written a\n"
    assert (ds / "c.txt").read_text() == "fixed by hand\n"
    assert "kept 1" in j.message


def test_jobs_survive_a_restart(ds, monkeypatch):
    j = job(mode="missing")
    captioner.update(j, status="running", written={"c.png": "x"})
    monkeypatch.setattr(captioner, "_jobs", {})
    monkeypatch.setattr(captioner, "_loaded", False)
    back = captioner.get(j.job_id)
    assert back.written == {"c.png": "x"}
    assert back.status == "stopped"  # it was running when ComfyUI went down


def test_mask_stats(tmp_path):
    m = Image.new("L", (100, 50), 0)
    m.paste(255, (0, 0, 50, 50))  # left half trained
    path = tmp_path / "m.png"
    m.save(path)
    cover, w, h = datasets.mask_stats(Path(path))
    assert abs(cover - 0.5) < 0.02 and (w, h) == (100, 50)


def test_clean_all_states_keeps_resumable_runs(tmp_path, monkeypatch):
    from zentrainer import storage
    from zentrainer.runs import RunStore

    states = tmp_path / "states"
    monkeypatch.setattr(storage, "_roots", lambda: {"states": states})
    store = RunStore(tmp_path / "runs")
    for run_id, status in (("done1", "done"), ("stopped1", "stopped")):
        store.create(run_id, {"name": run_id})
        store.update(run_id, status=status)
        (states / run_id).mkdir(parents=True)
        (states / run_id / "s.pt").write_bytes(b"x" * 100)
    freed, runs = storage.clean_all_states(store)
    assert (freed, runs) == (100, 1)
    assert not (states / "done1").exists()
    assert (states / "stopped1" / "s.pt").exists()  # it can still resume
