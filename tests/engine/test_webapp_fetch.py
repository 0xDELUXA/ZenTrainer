"""When the node pack fetches the prebuilt web app (server/routes/webapp.py), offline."""

import subprocess

import pytest

pytest.importorskip("aiohttp")
from zentrainer.server.routes import webapp


@pytest.fixture
def clone(tmp_path, monkeypatch):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "app.ts").write_text("x")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "web")
    monkeypatch.setattr(webapp, "ROOT", tmp_path)
    monkeypatch.setattr(webapp, "WEB_DIST", tmp_path / "web" / "dist")
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return tmp_path / "web" / "dist", head


def test_a_clone_without_the_app_wants_its_version(clone):
    dist, head = clone
    assert webapp._wanted() == head


def test_a_fetched_app_is_replaced_only_when_web_moved_on(clone):
    dist, head = clone
    dist.mkdir()
    (dist / "index.html").write_text("<html>")
    (dist / ".source").write_text(head + "\n")
    assert webapp._wanted() is None
    (dist / ".source").write_text("0" * 40 + "\n")  # a pull moved web/ on
    assert webapp._wanted() == head


def test_a_build_of_your_own_is_left_alone(clone):
    dist, _ = clone
    dist.mkdir()
    (dist / "index.html").write_text("<html>")
    assert webapp._wanted() is None


def test_not_a_git_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "ROOT", tmp_path)
    monkeypatch.setattr(webapp, "WEB_DIST", tmp_path / "web" / "dist")
    assert webapp._wanted() is None  # a Registry install: nothing to fetch with
