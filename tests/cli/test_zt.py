"""zt keeps a user's checkout safe: an update only fast-forwards a clean one, the web app is the only
thing it replaces (and only by a complete build), and it finds a free port rather than crash.
Everything runs on throwaway git repositories under tmp_path."""

import json
import socket
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("rich")

from zt import api, ports, repo, update, web  # noqa: E402

LOCK = {"comfyui": {"repo": "x", "ref": "x", "describe": "x"}, "python": "3.12",
        "torch": {"version": "2.0", "variants": {"cpu": {}}}, "port": 8189}


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def clones(tmp_path):
    """An upstream, a maintainer's clone that pushes to it, and the user's clone zt works in."""
    up, dev, user = tmp_path / "up.git", tmp_path / "dev", tmp_path / "user"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(up))
    git(tmp_path, "clone", "-q", str(up), str(dev))
    (dev / "comfy.lock").write_text(json.dumps(LOCK))
    (dev / "web").mkdir()
    (dev / "web" / "index.html").write_text("v1")
    (dev / "README.md").write_text("hi\n")
    git(dev, "add", "-A")
    git(dev, "commit", "-qm", "first")
    git(dev, "push", "-q", "origin", "HEAD:main")
    git(tmp_path, "clone", "-q", str(up), str(user))
    repo.set_root(user)
    # the user's own things, which nothing may touch
    (user / "comfyui" / "models").mkdir(parents=True)
    (user / "comfyui" / "models" / "mine.safetensors").write_text("precious")
    (user / ".zentrainer").mkdir()
    (user / ".zentrainer" / "local.json").write_text('{"port": 8189}')
    yield up, dev, user
    repo.set_root(Path(__file__).resolve().parents[2])


def upstream_commit(dev: Path, msg: str = "more") -> None:
    with open(dev / "README.md", "a") as f:
        f.write(msg + "\n")
    git(dev, "commit", "-qam", msg)
    git(dev, "push", "-q", "origin", "HEAD:main")


def untouched(user: Path) -> bool:
    return ((user / "comfyui" / "models" / "mine.safetensors").read_text() == "precious"
            and (user / ".zentrainer" / "local.json").read_text() == '{"port": 8189}')


def test_an_update_fast_forwards_a_clean_checkout(clones):
    _, dev, user = clones
    upstream_commit(dev, "new feature")
    found = update.check()
    assert found and found[1][0].endswith("new feature")
    assert update.apply(found, sync=False)
    assert git(user, "log", "-1", "--format=%s") == "new feature"
    assert untouched(user)


def test_an_update_leaves_local_changes_alone(clones):
    _, dev, user = clones
    upstream_commit(dev)
    (user / "README.md").write_text("my edit\n")
    head = git(user, "rev-parse", "HEAD")
    assert not update.apply(update.check(), sync=False)
    assert git(user, "rev-parse", "HEAD") == head and (user / "README.md").read_text() == "my edit\n"
    assert untouched(user)


def test_an_update_leaves_local_commits_alone(clones):
    _, dev, user = clones
    upstream_commit(dev)
    (user / "notes.txt").write_text("mine")
    git(user, "add", "notes.txt")
    git(user, "commit", "-qm", "my own commit")
    head = git(user, "rev-parse", "HEAD")
    assert not update.apply(update.check(), sync=False)
    assert git(user, "rev-parse", "HEAD") == head


def test_nothing_new_is_nothing_to_do(clones):
    assert update.check() is None


def test_the_prebuilt_web_app_is_used_only_when_it_matches(clones, monkeypatch, tmp_path):
    up, dev, user = clones
    monkeypatch.setattr(web.shutil, "which", lambda name, *a, **k: None if name == "pnpm" else "x")
    want = web.source()

    def publish(source: str) -> None:
        build = tmp_path / f"build-{source[:6]}"
        build.mkdir()
        (build / "index.html").write_text(f"built from {source}")
        (build / "SOURCE").write_text(source + "\n")
        git(build, "init", "-q", "-b", "web-dist")
        git(build, "add", "-A")
        git(build, "commit", "-qm", "build")
        git(build, "push", "-qf", str(up), "HEAD:refs/heads/web-dist")

    publish("0" * 40)  # a build of some other version: not used
    web.ensure()
    assert not (repo.DIST / "index.html").exists()
    publish(want)
    web.ensure()
    assert (repo.DIST / "index.html").read_text() == f"built from {want}"
    assert web.built_from() == want and not (repo.DIST / "SOURCE").exists()
    assert not (repo.WEB / ".dist-new").exists() and not (repo.WEB / ".dist-old").exists()
    assert untouched(user)


def test_a_taken_port_moves_to_the_next_free_one(clones):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        taken = s.getsockname()[1]
        (repo.LOCAL).write_text(json.dumps({"port": taken}))
        assert ports.taken(taken)
        got = ports.free()
        assert got != taken and not ports.taken(got)
        with pytest.raises(SystemExit):  # a --port given on the command line is kept to
            ports.free(taken)


def test_follow_reads_only_new_rows_from_either_server():
    rows = [{"step": i} for i in range(1, 6)]
    assert api.new_rows({"metrics": rows[3:], "metrics_count": 5}, 3) == (rows[3:], 5)  # with ?since
    assert api.new_rows({"metrics": rows}, 3) == (rows[3:], 5)  # an older server sends them all
    assert api.new_rows({"metrics": rows}, None) == (rows, 5)


def test_settings_on_the_command_line():
    spec = api.apply_sets({"optim": {"lr": 1}}, ["optim.lr=0.0002", "samples.prompts=[\"a cat\"]", "name=plain"])
    assert spec == {"optim": {"lr": 0.0002}, "samples": {"prompts": ["a cat"]}, "name": "plain"}
