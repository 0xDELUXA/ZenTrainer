"""UI-format workflow -> API-format prompt, without a browser.

ComfyUI only executes API prompts ({id: {class_type, inputs}}); templates and saved workflows
are the frontend's UI format, which the browser converts in graphToPrompt (executionUtil.ts +
ExecutableNodeDTO.ts). This is a port of that conversion so the harness can run any template:

  * widgets_values are positional; names come from object_info input order (required, then
    optional), skipping the frontend's extra slots (control_after_generate, upload buttons).
  * links win over widget values; a link that resolves to nothing leaves the widget value.
  * mode 2 (muted) nodes are dropped, mode 4 (bypassed) nodes are routed through (the
    first input matching the consumer's type), Reroutes are followed, PrimitiveNodes push their
    value into the widget they feed, Notes vanish.
  * subgraphs are flattened recursively. Inner nodes get the frontend's execution ids,
    "<instance>:<inner>" ("<outer>:<inner instance>:<node>" when nested); a widget promoted to
    the instance (a subgraph input wired to an inner widget) overrides the inner value.

exposed_inputs() maps extra.linearData (the frontend's App mode) onto the converted prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MUTED, BYPASS = 2, 4
SUBGRAPH_IN = -10  # origin id of links from a subgraph's input node
VIRTUAL = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}
# Input types the frontend draws as widgets (ComfyWidgets + core extensions); anything else is a socket.
WIDGET_TYPES = {"INT", "FLOAT", "BOOLEAN", "STRING", "COMBO", "MARKDOWN", "COLOR", "IMAGECOMPARE", "BOUNDING_BOX",
                "CHART", "GALLERIA", "PAINTER", "COMPOSITOR", "TEXTAREA", "CURVE", "RANGE", "VIDEO_EDIT",
                "RESOLUTION_PREVIEW", "BOUNDING_BOXES", "COLORS", "CAMERA_ANGLE_VIEW", "CAMERA_INFO_STATE", "LOAD_3D",
                "LOAD_3D_ADVANCED", "PREVIEW_3D", "AUDIO_UI", "AUDIOUPLOAD", "AUDIO_RECORD", "WEBCAM",
                "COMFY_DYNAMICCOMBO_V3"}
# Widgets without a widgets_values slot (widget.serialize = false); None = not sent in the prompt either.
UNSAVED = {"IMAGECOMPARE": ["", ""], "RESOLUTION_PREVIEW": None}
CONTROL = {"fixed", "increment", "decrement", "randomize", "increment-wrap"}
BUTTONS = {"image", "video", "audio", "upload", "upload3dmodel", "uploadExtraResources", "clear"}
HELPERS = {"control_after_generate", "control_filter_list", "upload"}  # frontend-only widgets
UPLOAD_FLAGS = ("image_upload", "animated_image_upload", "video_upload", "audio_upload", "file_upload", "mesh_upload")


class WorkflowError(ValueError):
    pass


# --- object_info specs ---------------------------------------------------------------------------

def _opts(spec: list) -> dict:
    return spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}


def _type(spec: list) -> str:
    return "COMBO" if isinstance(spec[0], list) else str(spec[0])


def _specs(info: dict) -> list[tuple[str, list]]:
    inp, order = info.get("input", {}), info.get("input_order") or {}
    return [(k, inp[sec][k]) for sec in ("required", "optional") for k in (order.get(sec) or inp.get(sec) or {})
            if k in (inp.get(sec) or {})]


def _widget_type(spec: list) -> str | None:
    o = _opts(spec)
    if o.get("forceInput"):
        return None
    t = o.get("widgetType") or _type(spec)
    return t if t in WIDGET_TYPES else None


def _default(spec: list) -> Any:
    o = _opts(spec)
    if "default" in o:
        return o["default"]
    choices = spec[0] if isinstance(spec[0], list) else o.get("options")
    if choices:
        c = choices[0]
        return c["key"] if isinstance(c, dict) else c
    return None


def _option(spec: list, key: Any) -> dict:
    """The sub-inputs a dynamic combo brings for the selected key."""
    for opt in _opts(spec).get("options") or []:
        if isinstance(opt, dict) and opt.get("key") == key:
            return opt.get("inputs") or {}
    return {}


def input_spec(info: dict, name: str, values: dict | None = None) -> list | None:
    """The object_info spec for `name`, descending into dynamic combos for "combo.sub" names."""
    specs = dict(_specs(info))
    if name in specs:
        return specs[name]
    head, _, rest = name.partition(".")
    if rest and head in specs and _type(specs[head]) == "COMFY_DYNAMICCOMBO_V3":
        key = (values or {}).get(head, _default(specs[head]))
        sub = _option(specs[head], key)
        sub_info = {"input": sub, "input_order": {s: list(v) for s, v in sub.items()}}
        return input_spec(sub_info, rest, {k.partition(".")[2]: v for k, v in (values or {}).items() if k.startswith(head + ".")})
    return None


def describe_spec(spec: list | None) -> dict:
    if not spec:
        return {}
    o = {k: v for k, v in _opts(spec).items() if k not in ("options", "tooltip")}
    out = {"type": _type(spec), **o, "image_upload": any(_opts(spec).get(f) for f in UPLOAD_FLAGS)}
    choices = spec[0] if isinstance(spec[0], list) else _opts(spec).get("options")
    if choices is not None:
        out["choices"] = [c["key"] if isinstance(c, dict) else c for c in choices]
    return out


# --- widget values -------------------------------------------------------------------------------

def widget_values(node: dict, info: dict) -> dict[str, Any]:
    """Name the node's widgets_values: a list in widget order (what the frontend restores from by
    default), or a dict by name. Newer saves also carry widgets_values_named; it's only trusted
    when it names widgets the node no longer has, i.e. the list was saved against an older schema."""
    raw, named = node.get("widgets_values"), node.get("widgets_values_named")
    if not isinstance(raw, list):
        return _name_widgets(raw if isinstance(raw, dict) else named, info)[0]
    out = _name_widgets(raw, info)[0]
    if isinstance(named, dict) and any(k not in out and k.rsplit(" ", 1)[-1] not in HELPERS for k in named):
        return _name_widgets(named, info)[0]
    return out


def _name_widgets(raw: Any, info: dict) -> tuple[dict[str, Any], int]:
    """(named values, how many list slots were consumed)."""
    named = raw if isinstance(raw, dict) else None
    vals = list(raw) if isinstance(raw, list) else []
    out: dict[str, Any] = {}
    pos = 0

    def take() -> Any:
        nonlocal pos
        pos += 1
        return vals[pos - 1] if pos <= len(vals) else None

    def peek() -> Any:
        return vals[pos] if pos < len(vals) else None

    def walk(specs: list[tuple[str, list]], prefix: str) -> None:
        nonlocal pos
        for name, spec in specs:
            wt = _widget_type(spec)
            key = prefix + name
            if wt in UNSAVED and UNSAVED[wt] is not None:
                out[key] = UNSAVED[wt]
            if wt is None or wt in UNSAVED:
                continue
            if isinstance(named, dict):
                v = named.get(key, _default(spec))
            else:
                v = take() if pos < len(vals) else _default(spec)
            out[key] = v
            o = _opts(spec)
            if wt == "INT" and o.get("control_after_generate", name in ("seed", "noise_seed")) and peek() in CONTROL:
                take()
            elif wt == "COMBO" and o.get("control_after_generate") and peek() in CONTROL:
                take()
                if isinstance(peek(), str) and peek() not in _choices(specs, name):  # control_filter_list
                    take()
            elif wt == "COMBO" and not o.get("widgetType") and any(o.get(f) for f in UPLOAD_FLAGS):
                while pos < len(vals) and (peek() is None or isinstance(peek(), dict) or peek() in BUTTONS):
                    take()
            elif wt == "BOUNDING_BOX":
                pos += 4  # x/y/width/height sub-widgets mirror the box
            if wt == "COMFY_DYNAMICCOMBO_V3":
                sub = _option(spec, v)
                walk([(k, s) for sec in ("required", "optional") for k, s in (sub.get(sec) or {}).items()], key + ".")

    walk(_specs(info), "")
    return out, pos


def _choices(specs: list[tuple[str, list]], name: str) -> list:
    spec = dict(specs)[name]
    c = spec[0] if isinstance(spec[0], list) else _opts(spec).get("options") or []
    return [x["key"] if isinstance(x, dict) else x for x in c]


def _wrap(value: Any, wtype: str | None) -> Any:
    # A list in an API prompt means a link, so the frontend boxes list-valued widgets.
    if wtype == "CURVE" and value is not None:
        return {"__type__": "CURVE", "__value__": value}
    return {"__value__": value} if isinstance(value, list) else value


# --- graph ---------------------------------------------------------------------------------------

@dataclass
class _Link:
    origin_id: int
    origin_slot: int
    target_id: int
    target_slot: int
    type: str


@dataclass
class _Graph:
    nodes: dict[int, dict]
    links: dict[int, _Link]
    sub: dict | None = None  # the subgraph definition, None for the root graph


def _graph(nodes: list[dict], links: list, sub: dict | None = None) -> _Graph:
    ls = {}
    for l in links or []:
        if isinstance(l, dict):
            ls[l["id"]] = _Link(l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"], l.get("type", "*"))
        elif l:
            ls[l[0]] = _Link(l[1], l[2], l[3], l[4], l[5] if len(l) > 5 else "*")
    return _Graph({n["id"]: n for n in nodes}, ls, sub)


def _valid(a: Any, b: Any) -> bool:
    """litegraph's isValidConnection: wildcards match anything, comma lists match on overlap."""
    a, b = ("" if a in (None, "*", 0) else str(a)).lower(), ("" if b in (None, "*", 0) else str(b)).lower()
    if not a or not b or a == b:
        return True
    return any(x == y for x in a.split(",") for y in b.split(","))


@dataclass
class _Dto:
    """One node at one place in the flattened graph (ExecutableNodeDTO)."""
    id: str
    node: dict
    graph: _Graph
    path: tuple[str, ...]
    host: "_Dto | None"
    inputs: list[dict] = field(default_factory=list)
    sub: _Graph | None = None  # set when this node is a subgraph instance

    @property
    def mode(self) -> int:
        return self.node.get("mode") or 0

    @property
    def type(self) -> str:
        return self.node.get("type", "")


class _Converter:
    def __init__(self, workflow: dict, object_info: dict):
        self.oi = object_info
        self.subgraphs = {s["id"]: _graph(s.get("nodes", []), s.get("links", []), s)
                          for s in (workflow.get("definitions") or {}).get("subgraphs") or []}
        self.root = _graph(workflow.get("nodes", []), workflow.get("links", []))
        self.dtos: dict[str, _Dto] = {}
        self._values: dict[str, dict] = {}
        self._promoted: dict[str, dict[int, Any]] = {}
        for node in self.root.nodes.values():
            self._add(node, self.root, (), None)

    # flattening
    def _add(self, node: dict, graph: _Graph, path: tuple[str, ...], host: _Dto | None, depth: int = 0) -> None:
        dto = _Dto(":".join(path + (str(node["id"]),)), node, graph, path, host)
        dto.sub = self.subgraphs.get(node.get("type"))
        dto.inputs = self._instance_inputs(dto) if dto.sub else list(node.get("inputs") or [])
        self.dtos[dto.id] = dto
        if dto.sub and dto.mode not in (MUTED, BYPASS):
            if depth > 32:
                raise WorkflowError(f"subgraph nesting too deep at node {dto.id}")
            for inner in dto.sub.nodes.values():
                self._add(inner, dto.sub, path + (str(node["id"]),), dto, depth + 1)

    def _instance_inputs(self, dto: _Dto) -> list[dict]:
        """The frontend rebuilds an instance's inputs from the subgraph's; the saved ones only carry links."""
        saved = {i.get("name"): i for i in dto.node.get("inputs") or []}
        out = []
        for si in dto.sub.sub.get("inputs") or []:
            s = saved.get(si["name"], {})
            out.append({"name": si["name"], "type": si.get("type", "*"), "link": s.get("link"),
                        "label": s.get("label") or si.get("label")})
        return out

    # widget values, including what subgraph instances promote
    def values(self, dto: _Dto) -> dict[str, Any]:
        if dto.id not in self._values:
            info = self.oi.get(dto.type)
            self._values[dto.id] = widget_values(dto.node, info) if info else {}
        return self._values[dto.id]

    def _sub_targets(self, inst: _Dto, slot: int) -> list[tuple[_Dto, int]]:
        """Inner (dto, input index) pairs a subgraph input feeds."""
        si = (inst.sub.sub.get("inputs") or [])[slot]
        return [t for lid in si.get("linkIds") or [] if (t := self.link_target(inst.sub, inst.id, lid))]

    def link_target(self, graph: _Graph, prefix: str, lid: int) -> tuple[_Dto, int] | None:
        """(consumer dto, input index) of a link. Found by link id: an instance's saved inputs are
        partial, so target_slot doesn't index its (rebuilt) inputs."""
        l = graph.links.get(lid)
        dto = l and self.dtos.get(f"{prefix}:{l.target_id}" if prefix else str(l.target_id))
        if not dto:
            return None
        idx = next((i for i, inp in enumerate(dto.inputs) if inp.get("link") == lid), l.target_slot)
        return (dto, idx) if idx < len(dto.inputs) else None

    def _widget_of(self, dto: _Dto, idx: int) -> tuple[bool, Any]:
        """(is a widget, its value) for input idx: an inner widget, or a nested instance's promoted input."""
        inp = dto.inputs[idx]
        if dto.sub:
            prom = self.promoted(dto)
            return (idx in prom, prom.get(idx))
        name = (inp.get("widget") or {}).get("name")
        vals = self.values(dto)
        return (name is not None and name in vals, vals.get(name))

    def promoted(self, inst: _Dto) -> dict[int, Any]:
        """{subgraph input index: value} for inputs promoted to widgets on the instance."""
        if inst.id in self._promoted:
            return self._promoted[inst.id]
        self._promoted[inst.id] = {}
        widgeted: dict[int, Any] = {}
        for i in range(len(inst.inputs)):
            for inner, idx in self._sub_targets(inst, i):
                ok, v = self._widget_of(inner, idx)
                if ok:
                    widgeted[i] = v
                    break
        raw = inst.node.get("widgets_values")
        proxies = (inst.node.get("properties") or {}).get("proxyWidgets")
        if isinstance(raw, list) and proxies:  # legacy: values follow proxyWidgets, not inputs
            for (src, name, *_), v in zip(proxies, raw):
                for i in widgeted:
                    if v is not None and any(str(d.node["id"]) == str(src) and (d.inputs[k].get("widget") or {}).get("name") == name
                                             for d, k in self._sub_targets(inst, i)):
                        widgeted[i] = v
        elif isinstance(raw, list):
            for i, v in zip(sorted(widgeted), raw):
                if v is not None:
                    widgeted[i] = v
        self._promoted[inst.id] = widgeted
        return widgeted

    # link resolution (ExecutableNodeDTO.resolveInput / resolveOutput)
    def resolve_input(self, dto: _Dto, idx: int, typ: Any = None, seen: frozenset = frozenset()) -> tuple | None:
        key = (dto.id, "I", idx)
        if key in seen:
            raise WorkflowError(f"circular link at node {dto.id}")
        seen = seen | {key}
        inp = dto.inputs[idx] if idx < len(dto.inputs) else None
        link = inp and inp.get("link") is not None and dto.graph.links.get(inp["link"])
        if not link:
            return None
        if link.origin_id == SUBGRAPH_IN and dto.host:
            host = dto.host
            if link.origin_slot >= len(host.inputs):
                return None
            if host.inputs[link.origin_slot].get("link") is not None:
                return self.resolve_input(host, link.origin_slot, None, seen)
            prom = self.promoted(host)
            return ("value", prom[link.origin_slot]) if link.origin_slot in prom else None
        origin = self.dtos.get(":".join(dto.path + (str(link.origin_id),)))
        if not origin:
            return None
        return self.resolve_output(origin, link.origin_slot, typ if typ is not None else inp.get("type"), seen)

    def resolve_output(self, dto: _Dto, slot: int, typ: Any, seen: frozenset) -> tuple | None:
        key = (dto.id, "O", slot)
        if key in seen:
            raise WorkflowError(f"circular link at node {dto.id}")
        seen = seen | {key}
        if dto.mode == MUTED:
            return None
        if dto.mode == BYPASS:
            idx = self._bypass_slot(dto, slot, typ)
            return None if idx < 0 else self.resolve_input(dto, idx, None, seen)
        if dto.sub:
            outs = dto.sub.sub.get("outputs") or []
            lids = outs[slot].get("linkIds") or [] if slot < len(outs) else []
            l = lids and dto.sub.links.get(lids[0])
            if not l:
                return None
            if l.origin_id == SUBGRAPH_IN:  # an input wired straight to an output
                return self.resolve_input(dto, l.origin_slot, None, seen)
            inner = self.dtos.get(f"{dto.id}:{l.origin_id}")
            return inner and self.resolve_output(inner, l.origin_slot, typ, seen)
        if dto.type == "Reroute":
            return self.resolve_input(dto, 0, typ, seen)
        if dto.type == "PrimitiveNode":
            wv = dto.node.get("widgets_values")
            return ("primitive", wv[0], dto.id) if isinstance(wv, list) and wv else None
        if dto.type in VIRTUAL:
            return None
        return ("link", dto.id, slot)

    def _bypass_slot(self, dto: _Dto, slot: int, typ: Any) -> int:
        ins = dto.inputs
        outs = dto.node.get("outputs") or []
        otype = outs[slot].get("type") if slot < len(outs) else "*"
        if typ in ("*", "", None):
            return slot if len(ins) > slot else 0
        if slot < len(ins) and _valid(ins[slot].get("type"), otype) and _valid(ins[slot].get("type"), typ):
            return slot
        for i, inp in enumerate(ins):
            if inp.get("type") == typ:
                return i
        return next((i for i, inp in enumerate(ins) if _valid(inp.get("type"), otype) and _valid(inp.get("type"), typ)), -1)

    # emission
    def emitted(self, dto: _Dto) -> bool:
        if dto.sub or dto.type in VIRTUAL or dto.mode in (MUTED, BYPASS):
            return False
        return all(self.dtos[":".join(dto.path[:i + 1])].mode not in (MUTED, BYPASS) for i in range(len(dto.path)))

    def prompt(self) -> dict:
        out: dict[str, dict] = {}
        for dto in self.dtos.values():
            if not self.emitted(dto):
                continue
            info = self.oi.get(dto.type) or {}
            specs = dict(_specs(info)) if info else {}
            vals = dict(self.values(dto))
            inputs: dict[str, Any] = {}
            for k, v in vals.items():
                spec = specs.get(k) or input_spec(info, k, vals) if info else None
                inputs[k] = _wrap(v, spec and _widget_type(spec))
            for i, inp in enumerate(dto.inputs):
                r = self.resolve_input(dto, i)
                if r is None or (r[0] == "primitive" and not inp.get("widget")):
                    continue
                inputs[inp.get("name")] = [r[1], r[2]] if r[0] == "link" else _wrap(r[1], None)
            out[dto.id] = {"inputs": inputs, "class_type": dto.type,
                           "_meta": {"title": _title(dto, self.oi)}}
        for node in out.values():  # drop links to nodes that weren't emitted
            for k in [k for k, v in node["inputs"].items() if isinstance(v, list) and len(v) == 2 and v[0] not in out]:
                del node["inputs"][k]
        return out


def to_api(workflow: dict, object_info: dict) -> dict:
    """Convert a UI-format workflow to an API prompt {id: {class_type, inputs, _meta: {title}}}."""
    if "nodes" not in workflow:
        if all(isinstance(v, dict) and "class_type" in v for v in workflow.values()):
            return workflow  # already an API prompt
        raise WorkflowError("not a ComfyUI workflow (no nodes)")
    return _Converter(workflow, object_info).prompt()


# --- App mode ------------------------------------------------------------------------------------

_WIDGET_ID = re.compile(r"^([^:]+):([^:]+):([^:]+)$")
_Exposed = tuple  # (concrete targets [(dto, input name)], label, linked)


def _widget_name(dto: _Dto, idx: int) -> str:
    return (dto.inputs[idx].get("widget") or {}).get("name") or dto.inputs[idx].get("name")


def _concrete(conv: _Converter, dto: _Dto, idx: int) -> list[tuple[_Dto, str]]:
    """The real widgets an input lands on, following promotion down through nested subgraphs."""
    if not dto.sub:
        return [(dto, _widget_name(dto, idx))]
    return [t for inner, k in conv._sub_targets(dto, idx) for t in _concrete(conv, inner, k)]


def _on_node(conv: _Converter, dto: _Dto, widget: str) -> _Exposed | None:
    """Widget `widget` of node `dto`: a promoted instance input, a PrimitiveNode's value, or a plain widget."""
    if dto.sub:
        for i, inp in enumerate(dto.inputs):
            if inp["name"] == widget and i in conv.promoted(dto):
                return _concrete(conv, dto, i), inp.get("label") or widget, inp.get("link") is not None
        return None
    if dto.type == "PrimitiveNode":
        targets = [(d, _widget_name(d, i)) for d in conv.dtos.values() if conv.emitted(d)
                   for i, inp in enumerate(d.inputs) if inp.get("widget")
                   and (r := conv.resolve_input(d, i)) and r[0] == "primitive" and r[2] == dto.id]
        return (targets, dto.node.get("title") or widget, False) if targets else None
    if widget not in conv.values(dto):
        return None
    inp = next((i for i in dto.inputs if (i.get("widget") or {}).get("name") == widget), {})
    return [(dto, widget)], inp.get("label") or widget, inp.get("link") is not None


def _find_input(conv: _Converter, ref: Any, widget: str) -> tuple[_Dto, _Exposed] | None:
    """Resolve a linearData input ref the way appModeStore does."""
    sid = str(ref)
    if m := _WIDGET_ID.match(sid):  # canonical widget id "graphId:nodeId:name"; nodeId is a root node
        sid, widget = m.group(2), m.group(3)
    elif ":" in sid:  # "subgraphUuid:localId" locator: the host instance on the root graph
        sid = sid.split(":")[-1]
    roots = [conv.dtos[str(n)] for n in conv.root.nodes]
    if (dto := conv.dtos.get(sid)) and not dto.path and (e := _on_node(conv, dto, widget)):
        return dto, e
    # legacy: an inner node id + widget, promoted on exactly one root instance
    hits = []
    for inst in (d for d in roots if d.sub):
        found = False
        for i, inp in enumerate(inst.inputs):
            tg = conv._sub_targets(inst, i)
            if i in conv.promoted(inst) and tg and str(tg[0][0].node["id"]) == sid and _widget_name(*tg[0]) == widget:
                hits.append((inst, (_concrete(conv, inst, i), inp.get("label") or inp["name"], inp.get("link") is not None)))
                found = True
        proxies = (inst.node.get("properties") or {}).get("proxyWidgets") or []
        inner = conv.dtos.get(f"{inst.id}:{sid}")
        if not found and inner and any(str(p[0]) == sid and p[1] == widget for p in proxies):
            if e := _on_node(conv, inner, widget):  # proxied but not wired to a subgraph input
                hits.append((inst, e))
    return hits[0] if len(hits) == 1 else None


def _title(dto: _Dto, object_info: dict) -> str:
    return dto.node.get("title") or (dto.sub.sub.get("name") if dto.sub else None) or \
        (object_info.get(dto.type) or {}).get("display_name") or dto.type


def exposed_inputs(workflow: dict, object_info: dict) -> dict:
    """Resolve extra.linearData to where each input lands in to_api()'s prompt.

    {inputs: [{node_id, input, class_type, spec, value, title, label, targets, linked, ref}],
     outputs: [{ref, node_ids}], unresolved: [ref, ...]} -- refs the frontend would drop too.
    `targets` lists every [node_id, input] a promoted input fans out to; node_id/input is the first.
    """
    data = (workflow.get("extra") or {}).get("linearData") or {}
    conv = _Converter(workflow, object_info)
    api = conv.prompt()
    res: dict[str, list] = {"inputs": [], "outputs": [], "unresolved": []}
    for entry in data.get("inputs") or []:
        ref, widget = entry[0], entry[1]
        found = _find_input(conv, ref, widget)
        if not found:
            res["unresolved"].append(list(entry))
            continue
        owner, (targets, label, linked) = found
        live = [(d, n) for d, n in targets if d.id in api]
        d, name = (live or targets or [(owner, widget)])[0]
        info = object_info.get(d.type) or {}
        vals = conv.values(d)
        v = api.get(d.id, {}).get("inputs", {}).get(name, vals.get(name))
        if isinstance(v, dict) and "__value__" in v:
            v = v["__value__"]
        res["inputs"].append({
            "ref": list(entry), "node_id": d.id if d.id in api else None, "input": name, "class_type": d.type,
            "spec": describe_spec(input_spec(info, name, vals)), "value": None if isinstance(v, list) else v,
            "linked": linked or isinstance(v, list), "title": _title(owner, object_info),
            "label": label, "targets": [[x.id, n] for x, n in live]})
    for ref in data.get("outputs") or []:
        sid = str(ref)
        if sid in conv.dtos and not conv.dtos[sid].path:
            d = conv.dtos[sid]
            ids = [d.id] if d.id in api else [i for i in api if i.startswith(d.id + ":") and
                                              (object_info.get(api[i]["class_type"]) or {}).get("output_node")]
        else:  # an inner node id: every instance of it
            ids = [i for i, d in conv.dtos.items() if d.path and str(d.node["id"]) == sid and i in api]
        (res["outputs"] if ids else res["unresolved"]).append({"ref": ref, "node_ids": ids} if ids else ref)
    return res
