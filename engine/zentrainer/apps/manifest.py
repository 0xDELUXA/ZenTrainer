"""App manifests: load, validate, list.

Built-ins live in engine/apps/<id>/app.json, the user's in <user>/zentrainer/apps/<id>/.
A manifest binds inputs to workflow nodes by title ("Positive.text"); every binding is
checked against every workflow at load time, so a broken app fails here, not mid-run.

  workflows   files beside app.json, API or editor format (a normal ComfyUI save,
              converted on load). Each may have `when`: {"layout": "checkpoint"|"split"}
              for a file-picker model, or {"family": "<id>"} for one workflow per model
              family, picked by a `family` input.
  inputs      `component` (or the older `type`) says how it's drawn; `bind` is
              "Title.input", or alternatives "Sampler.seed | Noise.noise_seed" (the first the
              workflow has). `optional` inputs bind only where their node exists; the UI
              hides them for other workflows. `tip`, `section` ("main" | "advanced") and
              `default` (none = keep the workflow's own value) are presentation.
  layout      the shared view that draws it ("generate"; "classic" for the old one).
  cover       an image file beside app.json for the Apps page.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUILTIN = Path(__file__).resolve().parent.parent.parent / "apps"

INPUT_TYPES = {"prompt", "text", "number", "slider", "seed", "toggle", "choice", "model", "family", "model-file",
               "loras", "lora", "resolution", "size"}
SWEEPABLE = {"prompt", "text", "number", "seed", "choice", "lora"}
# Standard titles for model loaders: a "model" input fills whichever of these exist.
MODEL_LOADERS = {
    "Load Checkpoint": {"ckpt_name": "ckpt_name"},
    "Load Diffusion Model": {"unet_name": "unet_name", "unet_weight_dtype": "weight_dtype"},
    "Load Text Encoder": {"clip_name": "clip_name", "clip_type": "type"},
    "Load VAE": {"vae_name": "vae_name"},
}
LORA_CHAIN = "LoRAs"  # title of the LoraLoaderModelOnly placeholder a lora/loras input expands


class AppError(ValueError):
    pass


@dataclass
class App:
    id: str
    root: Path
    manifest: dict[str, Any]
    workflows: list[tuple[dict, dict]] = field(default_factory=list)  # (when, graph)
    builtin: bool = True

    @property
    def inputs(self) -> list[dict]:
        return self.manifest["inputs"]

    def families(self) -> list[str]:
        return [w["family"] for w, _ in self.workflows if w.get("family")]

    def bound_in(self, graph: dict) -> list[str]:
        """Inputs this workflow can take: optional ones whose node it lacks are left out."""
        return [i["id"] for i in self.inputs if not i.get("optional") or all(resolve_bind(graph, b) for b in _binds(i))]

    def input(self, input_id: str) -> dict:
        for i in self.inputs:
            if i["id"] == input_id:
                return i
        raise AppError(f"{self.id}: no input {input_id!r}")

    def public(self) -> dict:
        m = self.manifest
        return {"id": self.id, "name": m["name"], "description": m.get("description", ""), "icon": m.get("icon"),
                "category": m.get("category", "generate"), "inputs": m["inputs"], "outputs": m.get("outputs", []),
                "builtin": self.builtin, "extends": m.get("extends"), "layout": m.get("layout", "classic"),
                "cover": bool(m.get("cover")), "families": self.families(),
                "layouts": sorted({w.get("layout") for w, _ in self.workflows if w.get("layout")})}


def titles(graph: dict) -> dict[str, str]:
    """title -> node id"""
    return {n.get("_meta", {}).get("title", nid): nid for nid, n in graph.items()}


def _binds(inp: dict) -> list[str]:
    b = inp.get("bind")
    return [b] if isinstance(b, str) else list(b or [])


def resolve_bind(graph: dict, bind: str) -> str | None:
    """The first of "A.x | B.y" this workflow has, or None."""
    t = titles(graph)
    for alt in (a.strip() for a in bind.split("|")):
        title, _, field_ = alt.partition(".")
        if title in t and (not field_ or field_ in graph[t[title]]["inputs"] or title == LORA_CHAIN):
            return alt
    return None


def _check_bind(app_id: str, graph: dict, bind: str, where: str) -> None:
    if "|" in bind:
        if resolve_bind(graph, bind) is None:
            raise AppError(f"{app_id}: {where} binds to {bind!r}, none of which the workflow has")
        return
    title, _, field_ = bind.partition(".")
    t = titles(graph)
    if title not in t:
        raise AppError(f"{app_id}: {where} binds to node {title!r}, which isn't in the workflow")
    if title == LORA_CHAIN and field_ in ("", "strength"):
        return
    if field_ and field_ not in graph[t[title]]["inputs"]:
        raise AppError(f"{app_id}: {where} binds to {bind!r}, but that node has no input {field_!r}")


def _read(root: Path) -> dict:
    try:
        return json.loads((root / "app.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise AppError(f"{root.name}: can't read app.json ({e})") from e


def _resolve_extends(root: Path, m: dict, seen: tuple[str, ...] = ()) -> dict:
    """An app can build on another: `"extends": "generate"` inherits the parent's inputs,
    outputs and workflows. The child adds inputs (same id = replace, in place),
    removes some (`"remove": [ids]`) and may swap `workflows`/`outputs` outright.
    Parents are looked up next to the child, then among the built-ins."""
    parent_id = m.get("extends")
    if not parent_id:
        return m
    if parent_id in seen:
        raise AppError(f"{m.get('id', root.name)}: extends loop through {parent_id!r}")
    for base in (root.parent, BUILTIN):
        if (base / parent_id / "app.json").is_file():
            proot = base / parent_id
            break
    else:
        raise AppError(f"{m.get('id', root.name)}: extends unknown app {parent_id!r}")
    parent = _resolve_extends(proot, _read(proot), (*seen, parent_id))
    # parent workflow paths are relative to the parent's folder; rebase onto the child's
    pworkflows = [{**w, "file": str((proot / w["file"]).resolve())} for w in parent["workflows"]]
    removed = set(m.get("remove", []))
    own = {i["id"]: i for i in m.get("inputs", [])}
    inputs = [own.pop(i["id"], i) for i in parent["inputs"] if i["id"] not in removed] + list(own.values())
    merged = {**parent, **{k: v for k, v in m.items() if k not in ("inputs", "remove", "extends")}}
    merged["inputs"] = inputs
    merged.setdefault("workflows", pworkflows)
    if "workflows" not in m:
        merged["workflows"] = pworkflows
    merged["extends"] = parent_id
    return merged


def load(root: Path, builtin: bool = True) -> App:
    m = _resolve_extends(root, _read(root))
    app_id = m.get("id") or root.name
    for key in ("name", "workflows", "inputs"):
        if key not in m:
            raise AppError(f"{app_id}: app.json needs {key!r}")
    allowed = (root.parent.resolve(), BUILTIN.resolve())  # own apps folder, or inherited built-ins
    workflows = []
    for w in m["workflows"]:
        path = (root / w["file"]).resolve()
        if not any(a in path.parents for a in allowed):
            raise AppError(f"{app_id}: workflow {w['file']!r} is outside the apps folders")
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise AppError(f"{app_id}: can't read workflow {w['file']!r} ({e})") from e
        if "nodes" in graph:  # an editor-format save: to the API format the queue runs
            from . import catalog
            from .workflow import WorkflowError, to_api

            try:
                graph = to_api(graph, catalog.object_info())
            except WorkflowError as e:
                raise AppError(f"{app_id}: can't convert {w['file']!r} ({e})") from e
        workflows.append((w.get("when", {}), graph))
    if not workflows:
        raise AppError(f"{app_id}: no workflows")
    ids = set()
    for inp in m["inputs"]:
        if "component" in inp:  # the newer name for type
            inp["type"] = inp["component"]
        iid, typ = inp.get("id"), inp.get("type")
        if not iid or iid in ids:
            raise AppError(f"{app_id}: inputs need unique ids ({iid!r})")
        ids.add(iid)
        if typ not in INPUT_TYPES:
            raise AppError(f"{app_id}: input {iid!r} has unknown type {typ!r}")
        if inp.get("sweep") and typ not in SWEEPABLE:
            raise AppError(f"{app_id}: input {iid!r} ({typ}) can't be swept")
        for _, graph in workflows:
            for b in _binds(inp):
                if inp.get("optional") and resolve_bind(graph, b) is None:
                    continue  # this workflow just doesn't have it
                _check_bind(app_id, graph, b, f"input {iid!r}")
            if typ in ("lora", "loras") and LORA_CHAIN not in titles(graph):
                raise AppError(f"{app_id}: input {iid!r} needs a {LORA_CHAIN!r} node in every workflow")
    for out in m.get("outputs", []):
        for _, graph in workflows:
            _check_bind(app_id, graph, out["from"], f"output {out.get('id')!r}")
    return App(app_id, root, m, workflows, builtin)


def list_apps(user_root: Path | None) -> tuple[list[App], list[str]]:
    """All loadable apps (user ones override built-ins by id) and the load errors."""
    found: dict[str, App] = {}
    errors: list[str] = []
    for base, builtin in ((BUILTIN, True), (user_root, False)):
        if base is None or not base.is_dir():
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
            try:
                app = load(d, builtin)
                found[app.id] = app
            except AppError as e:
                errors.append(str(e))
    return sorted(found.values(), key=lambda a: (a.manifest.get("category", ""), a.manifest["name"])), errors


def pick_workflow(app: App, layout: str | None, family: str | None = None) -> dict:
    if family is not None:
        for when, graph in app.workflows:
            if when.get("family") == family:
                return copy.deepcopy(graph)
        raise AppError(f"{app.id} has no workflow for {family!r}")
    for when, graph in app.workflows:
        if not when or when.get("layout") in (None, layout):
            return copy.deepcopy(graph)
    raise AppError(f"{app.id} has no workflow for a {layout!r} model")
