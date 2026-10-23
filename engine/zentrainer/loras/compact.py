"""Compacting a LoRA or LoKr to a size budget for the whole file, not a fixed share per layer.

Every layer's change is split into directions (singular vectors), each with its energy (its
singular value squared, the share of the change it carries) and its cost (the parameters it
takes to store). All the file's directions are ranked together by energy per parameter, and
the copy keeps the top of that ranking until it holds the share you ask for, or fills the size
you give. Layers that change a lot get rank; quiet ones get a little or are left out.

LoRA layers split by QR of their two thin factors and an SVD of the small core (as merge.resize
does). A LoKr layer is w1 ⊗ w2, whose singular values are the products of w1's and w2's, so:
  - kept as a LoKr, w1 stays (it's tiny) and w2 is cut to its top directions, stored as
    w2_a @ w2_b: the best LoKr of that size, exactly;
  - turned into a LoRA, its directions are the pairs (i, j), u1_i ⊗ u2_j and v1_i ⊗ v2_j with
    value s1_i·s2_j: the best LoRA of that rank, built without the full matrix.

With the base model it was trained on, each layer's energy is measured against that layer's
own weight (|change|² / |W|²): a change that's large but small beside a large weight matters
less. Only the weights' sizes are read, one at a time; int8 rows times their scales count, and
rotated (convrot) weights keep their size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import merge

LOKR = ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2")
PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "unet.")
POINTS = 200


@dataclass
class Layer:
    name: str
    kind: str  # "lora", "lokr" (kept as one), "lokr_lora" (made a LoRA)
    s: torch.Tensor  # singular values, largest first
    unit: int  # parameters per kept direction
    fixed: int  # parameters paid once the layer keeps any direction (LoKr's w1 and alpha)
    weight: float = 1.0  # 1 / |base weight|² when weighed by the base model


@dataclass
class Analysis:
    layers: list[Layer]
    other: int  # parameters copied as they are (norms, diffs, LoHa layers)
    unsupported: list[str]
    base_matched: int | None
    original_params: int
    # every direction, best first: layer index, position in the layer, energy share, parameters
    li: torch.Tensor = field(repr=False, default=None)
    pos: torch.Tensor = field(repr=False, default=None)
    gain: torch.Tensor = field(repr=False, default=None)
    cost: torch.Tensor = field(repr=False, default=None)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _groups(sd: dict[str, torch.Tensor]) -> tuple[dict[str, dict], dict[str, torch.Tensor]]:
    """LoKr layers (name -> their tensors by part), and the rest of the file."""
    lokr: dict[str, dict] = {}
    for k in sd:
        for part in LOKR:
            if k.endswith("." + part):
                lokr.setdefault(k[: -len(part) - 1], {})
    rest = {}
    for k, v in sd.items():
        head, _, part = k.rpartition(".")
        if head in lokr and (part in LOKR or part in ("alpha", "dora_scale")):
            lokr[head][part] = v
        else:
            rest[k] = v
    return lokr, rest


def _lokr_factors(p: dict, dev: str) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """(w1, w2 flattened to 2-D, w2's kernel shape) with the layer's scale baked into w1, as
    ComfyUI applies it: alpha / rank of the decomposed factor (w2's when both are)."""
    f = lambda k: p[k].float().to(dev)
    w1 = f("lokr_w1") if "lokr_w1" in p else f("lokr_w1_a") @ f("lokr_w1_b")
    w2 = f("lokr_w2") if "lokr_w2" in p else f("lokr_w2_a") @ f("lokr_w2_b")
    dim = p["lokr_w2_b"].shape[0] if "lokr_w2" not in p else p["lokr_w1_b"].shape[0] if "lokr_w1" not in p else None
    scale = float(p["alpha"]) / dim if dim and "alpha" in p else 1.0
    tail = tuple(w2.shape[2:]) if w2.dim() > 2 else ()
    return w1 * scale, w2.reshape(w2.shape[0], -1), tail


def _lora_svd(layer: dict, dev: str):
    up, down = layer["up"].float().to(dev), layer["down"].float().to(dev)
    u2 = up.reshape(up.shape[0], -1) * merge._scale(layer)
    d2 = down.reshape(down.shape[0], -1)
    qu, ru = torch.linalg.qr(u2)
    qd, rd = torch.linalg.qr(d2.T)
    uc, s, vc = torch.linalg.svd(ru @ rd.T)
    return qu @ uc, s, vc @ qd.T  # (out, r), (r,), (r, in*k)


def _pairs(s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A Kronecker product's singular values (largest first) and which (i, j) each is."""
    v = (s1[:, None] * s2[None, :]).reshape(-1)
    order = torch.argsort(v, descending=True)
    return v[order], order


def _base_key(name: str, base_keys: dict[str, str]) -> str | None:
    names = {name}
    for p in PREFIXES:
        if name.startswith(p):
            names.add(name[len(p):])
    for n in names:
        for cand in (n, *(p + n for p in PREFIXES)):
            if cand + ".weight" in base_keys:
                return base_keys[cand + ".weight"]
    return None


def _base_norms(base: str | os.PathLike, layers: list[Layer], shapes: dict[str, int]) -> int:
    """Set each layer's weight to 1/|W|² from the base model (int8 and fp8 rows times their
    scales); layers not found take the median. Returns how many were found."""
    from safetensors import safe_open

    found: dict[int, float] = {}
    with safe_open(str(base), framework="pt") as f:
        keys = set(f.keys())
        base_keys = {k: k for k in keys}
        for i, layer in enumerate(layers):
            key = _base_key(layer.name, base_keys)
            if key is None:
                continue
            w = f.get_tensor(key)
            if w.numel() != shapes[layer.name]:
                continue
            w = w.float()
            stem = key[: -len("weight")]
            for sk in (key + "_scale", stem + "weight_scale", stem + "scale_weight"):
                if sk in keys:
                    sc = f.get_tensor(sk).float()
                    w = w * (sc.reshape(-1, *[1] * (w.dim() - 1)) if sc.numel() == w.shape[0] else sc)
                    break
            n2 = float(w.square().sum())
            if n2 > 0:
                found[i] = n2
    if found:
        med = sorted(found.values())[len(found) // 2]
        for i, layer in enumerate(layers):
            layer.weight = 1.0 / found.get(i, med)
    return len(found)


def analyze(path: str | os.PathLike, fmt: str = "same", base: str | os.PathLike | None = None) -> Analysis:
    """Every layer's directions. fmt "same" keeps the file's kind; "lora" makes LoKr layers LoRA."""
    from safetensors.torch import load_file

    if fmt not in ("same", "lora"):
        raise ValueError(f"unknown format {fmt!r}")
    sd = load_file(str(path))
    original = sum(v.numel() for v in sd.values())
    dev = _device()
    lokr, rest = _groups(sd)
    loras, other = merge._split(rest)
    layers: list[Layer] = []
    shapes: dict[str, int] = {}
    unsupported = []
    for name, layer in sorted(loras.items()):
        _, s, _ = _lora_svd(layer, dev)
        out_dim, in_dim = layer["up"].shape[0], layer["down"][0].numel()
        layers.append(Layer(name, "lora", s.cpu(), out_dim + in_dim, 1))
        shapes[name] = out_dim * in_dim
    for name, p in sorted(lokr.items()):
        if "lokr_t2" in p:  # a Tucker-decomposed conv LoKr: copied as it is
            unsupported.append(name)
            other.update({f"{name}.{k}": v for k, v in p.items()})
            continue
        w1, w2, _ = _lokr_factors(p, dev)
        s1 = torch.linalg.svdvals(w1)
        s2 = torch.linalg.svdvals(w2)
        shapes[name] = w1.numel() * w2.numel()
        if fmt == "same":
            layers.append(Layer(name, "lokr", (s2 * s1.square().sum().sqrt()).cpu(), w2.shape[0] + w2.shape[1], w1.numel() + 1))
        else:
            v, _ = _pairs(s1, s2)
            layers.append(Layer(name, "lokr_lora", v.cpu(), w1.shape[0] * w2.shape[0] + w1.shape[1] * w2.shape[1], 1))
    for k in [k for k in other if k.endswith((".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b"))]:
        unsupported.append(k.rpartition(".")[0])
    an = Analysis(layers, sum(v.numel() for v in other.values()), sorted(set(unsupported)), None, original)
    if base is not None:
        an.base_matched = _base_norms(base, layers, shapes)
    _rank(an)
    return an


def _rank(an: Analysis) -> None:
    li, pos, gain, cost = [], [], [], []
    for i, layer in enumerate(an.layers):
        n = layer.s.numel()
        g = layer.s.double().square() * layer.weight
        li.append(torch.full((n,), i, dtype=torch.int32))
        pos.append(torch.arange(n, dtype=torch.int32))
        gain.append(g)
        c = torch.full((n,), float(layer.unit), dtype=torch.float64)
        c[0] += layer.fixed
        cost.append(c)
    if not li:
        an.li = an.pos = torch.zeros(0, dtype=torch.int32)
        an.gain = an.cost = torch.zeros(0, dtype=torch.float64)
        return
    li, pos, gain, cost = torch.cat(li), torch.cat(pos), torch.cat(gain), torch.cat(cost)
    # energy per parameter, best first; within a layer that's already largest first, so any
    # prefix of the ranking keeps each layer's top directions
    order = torch.argsort(gain / cost, descending=True, stable=True)
    total = gain.sum().clamp(min=1e-30)
    an.li, an.pos, an.gain, an.cost = li[order], pos[order], gain[order] / total, cost[order]


def _ranked(an: Analysis, max_rank: int = 0):
    if max_rank > 0:
        m = an.pos < max_rank
        return an.li[m], an.pos[m], an.gain[m], an.cost[m]
    return an.li, an.pos, an.gain, an.cost


def ranks(an: Analysis, keep: float | None = None, params: int | None = None, max_rank: int = 0) -> list[int]:
    """Each layer's kept rank for a share of the energy (keep, 0-1) or a parameter budget."""
    li, _, gain, cost = _ranked(an, max_rank)
    if keep is not None:
        n = int(torch.searchsorted(gain.cumsum(0), torch.tensor(min(keep, 1.0) - 1e-12, dtype=torch.float64)).item()) + 1
    elif params is not None:
        n = int(torch.searchsorted(cost.cumsum(0) + an.other, torch.tensor(float(params), dtype=torch.float64), right=True).item())
    else:
        raise ValueError("give keep or params")
    n = max(0, min(n, li.numel()))
    out = [0] * len(an.layers)
    for i, c in zip(*torch.unique(li[:n], return_counts=True)):
        out[int(i)] = int(c)
    return out


def curve(an: Analysis, max_rank: int = 0) -> list[tuple[int, float]]:
    """(parameters, share kept) along the ranking, for a size-against-quality chart."""
    _, _, gain, cost = _ranked(an, max_rank)
    if not gain.numel():
        return [(an.other, 1.0)]
    g, c = gain.cumsum(0), cost.cumsum(0) + an.other
    idx = torch.unique(torch.cat([torch.linspace(0, g.numel() - 1, POINTS).round().long(),
                                  torch.searchsorted(g, torch.tensor([0.9, 0.95, 0.99, 0.999], dtype=torch.float64)).clamp(max=g.numel() - 1)]))
    return [(an.other, 0.0)] + [(int(c[i]), float(g[i])) for i in idx]


def summary(an: Analysis, ks: list[int]) -> dict:
    kept = sum(int(k > 0) for k in ks)
    params = an.other + sum(layer.fixed + k * layer.unit for layer, k in zip(an.layers, ks) if k)
    allt = sum(float(layer.s.double().square().sum() * layer.weight) for layer in an.layers) or 1.0
    share = sum(float(layer.s[:k].double().square().sum() * layer.weight) for layer, k in zip(an.layers, ks) if k)
    nz = [k for k in ks if k]
    return {"params": params, "kept": share / allt, "layers": len(ks), "layers_kept": kept,
            "mean_rank": sum(nz) / len(nz) if nz else 0, "max_rank": max(nz, default=0), "ranks": ks}


def build(path: str | os.PathLike, out: str | os.PathLike, an: Analysis, ks: list[int], dtype: str = "bf16",
          overwrite: bool = False) -> dict:
    """Write the compacted file: each layer at its rank in `ks` (0: left out)."""
    from safetensors.torch import load_file, save_file

    from ..models.tensors import DTYPES

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype if dtype in DTYPES else "bf16"]
    path, out = Path(path), Path(out)
    if path.resolve() == out.resolve():
        raise ValueError("pick a new name: the compacted file can't replace the one it's made from")
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out.name} already exists")
    sd = load_file(str(path))
    dev = _device()
    lokr, rest = _groups(sd)
    loras, other = merge._split(rest)
    result: dict[str, torch.Tensor] = {}
    for layer, k in zip(an.layers, ks):
        if not k:
            continue
        name = layer.name
        if layer.kind == "lora":
            src = loras[name]
            u, s, v = _lora_svd(src, dev)
            root = s[:k].sqrt()
            up_shape, down = src["up"].shape, src["down"]
            result[f"{name}.lora_up.weight"] = (u[:, :k] * root).reshape(up_shape[0], k, *up_shape[2:]).cpu()
            result[f"{name}.lora_down.weight"] = (root[:, None] * v[:k]).reshape(k, *down.shape[1:]).cpu()
            result[f"{name}.alpha"] = torch.tensor(float(k))
            continue
        p = lokr[name]
        w1, w2, tail = _lokr_factors(p, dev)
        if layer.kind == "lokr":
            u, s, v = torch.linalg.svd(w2, full_matrices=False)
            root = s[:k].sqrt()
            result[f"{name}.lokr_w1"] = w1.cpu()
            result[f"{name}.lokr_w2_a"] = (u[:, :k] * root).cpu()
            result[f"{name}.lokr_w2_b"] = (root[:, None] * v[:k]).cpu()
            result[f"{name}.alpha"] = torch.tensor(float(k))  # alpha / rank = 1: the scale is in w1
        else:
            u1, s1, v1 = torch.linalg.svd(w1, full_matrices=False)
            u2, s2, v2 = torch.linalg.svd(w2, full_matrices=False)
            vals, order = _pairs(s1, s2)
            i, j = order[:k] // s2.numel(), order[:k] % s2.numel()
            root = vals[:k].sqrt()
            up = (u1[:, i][:, None, :] * u2[:, j][None, :, :]).reshape(-1, k) * root  # kron(u1_i, u2_j)
            down = (v1[i][:, :, None] * v2[j][:, None, :]).reshape(k, -1) * root[:, None]
            out_dim = w1.shape[0] * w2.shape[0]
            result[f"{name}.lora_up.weight"] = up.reshape(out_dim, k, *([1] * len(tail))).cpu()
            result[f"{name}.lora_down.weight"] = down.reshape(k, -1, *tail).cpu() if tail else down.cpu()
            result[f"{name}.alpha"] = torch.tensor(float(k))
        if "dora_scale" in p:
            result[f"{name}.dora_scale"] = p["dora_scale"]
    for name in an.unsupported:  # Tucker LoKr and LoHa layers, as they were
        result.update({k: v for k, v in sd.items() if k.startswith(name + ".")})
    result.update({k: v for k, v in other.items() if k not in result})
    meta = {}
    try:
        from ..models.tensors import read_header

        meta = read_header(path)[0]
    except (OSError, ValueError):
        pass
    st = summary(an, ks)
    meta["zentrainer_compact"] = (f"from {path.name}: {st['layers_kept']}/{st['layers']} layers, "
                                  f"mean rank {st['mean_rank']:.1f}, {st['kept'] * 100:.2f}% kept")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.part")  # hidden, and not a .safetensors ComfyUI would list
    try:
        save_file({k: (v.to(torch_dtype) if v.is_floating_point() else v).contiguous() for k, v in result.items()},
                  str(tmp), metadata=meta)
        os.chmod(tmp, 0o644)
        if out.exists() and not overwrite:
            raise FileExistsError(f"{out.name} already exists")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return {**{k: v for k, v in st.items() if k != "ranks"}, "bytes": out.stat().st_size}
