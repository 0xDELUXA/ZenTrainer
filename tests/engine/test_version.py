"""The engine's version (what /zentrainer/api/health reports) is the one the Registry publishes."""

import re
from pathlib import Path

import zentrainer


def test_the_engine_and_the_pack_agree_on_the_version():
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert zentrainer.__version__ == re.search(r'^version = "([^"]+)"', pyproject, re.M)[1]
