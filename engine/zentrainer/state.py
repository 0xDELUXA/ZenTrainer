"""Process-wide singletons, created lazily so importing the package stays cheap."""

from __future__ import annotations

import os

from .runs.store import RunStore

_store: RunStore | None = None


def store() -> RunStore:
    global _store
    if _store is None:
        import folder_paths

        _store = RunStore(os.path.join(folder_paths.get_user_directory(), "zentrainer", "runs"))
    return _store
