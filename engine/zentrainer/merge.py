"""Merging LoRAs, and shrinking one: the "soup" malcolmrey trains (one subject, a few short runs
with different optimizers, merged) and anyone's stack baked into one file.

Merge is exact: each layer's members are stacked side by side (up matrices concatenated, down
matrices concatenated, each scaled by its strength and alpha/rank), so the merged LoRA of
rank r1 + r2 + ... gives the same output as loading the members together at those strengths.

Resize keeps, per layer, the fewest singular directions that hold `energy` of the layer's
change (capped at `max_rank`), via QR of the two thin factors and an SVD of the small core, so
it never builds the full out × in matrix. Directions the members agree on carry most of the
energy; scattered disagreement is what's dropped.

LoRA files are read in ComfyUI's layouts: `<layer>.lora_up.weight` / `.lora_down.weight` /
`.alpha` (also diffusers' `lora_B` / `lora_A`). Anything else a member carries (diffs, norms)
is summed by strength.
"""

from __future__ import annotations

from pathlib import Path

import torch

UP = (".lora_up.weight", ".lora_B.weight")
DOWN = (".lora_down.weight", ".lora_A.weight")


def _split(sd: dict[str, torch.Tensor]) -> tuple[dict[str, dict], dict[str, torch.Tensor]]:
    """layer -> {up, down, alpha}, and the tensors that aren't LoRA factors."""
    layers: dict[str, dict] = {}
    other = {}
    for k, v in sd.items():
        for suf in UP:
            if k.endswith(suf):
                layers.setdefault(k[: -len(suf)], {})["up"] = v
                break
        else:
            for suf in DOWN:
                if k.endswith(suf):
                    layers.setdefault(k[: -len(suf)], {})["down"] = v
                    break
            else:
                if k.endswith(".alpha"):
                    layers.setdefault(k[: -len(".alpha")], {})["alpha"] = v
                else:
                    other[k] = v
    return {k: v for k, v in layers.items() if "up" in v and "down" in v}, other


def _scale(layer: dict) -> float:
    rank = layer["down"].shape[0]
    alpha = float(layer["alpha"]) if "alpha" in layer else float(rank)
    return alpha / rank


def merge(members: list[tuple[dict[str, torch.Tensor], float]]) -> dict[str, torch.Tensor]:
    """Exact merge of (state_dict, strength) members; the result loads at strength 1."""
    split = [(_split(sd), s) for sd, s in members]
    names = sorted({k for (layers, _), _ in split for k in layers})
    out: dict[str, torch.Tensor] = {}
    for name in names:
        ups, downs = [], []
        for (layers, _), strength in split:
            layer = layers.get(name)
            if layer is None:
                continue
            up = layer["up"].float()
            down = layer["down"].float()
            ups.append(up * (strength * _scale(layer)))
            downs.append(down)
        shape_tail = ups[0].shape[2:]  # conv kernels: (out, r, 1, 1) / (r, in, kh, kw)
        up = torch.cat([u.reshape(u.shape[0], u.shape[1], *shape_tail) for u in ups], dim=1)
        down = torch.cat(downs, dim=0)
        out[f"{name}.lora_up.weight"] = up
        out[f"{name}.lora_down.weight"] = down
        out[f"{name}.alpha"] = torch.tensor(float(down.shape[0]))  # scale 1: strengths are baked in
    for (_, other), strength in split:
        for k, v in other.items():
            out[k] = out[k] + v.float() * strength if k in out else v.float() * strength
    return out


def resize(sd: dict[str, torch.Tensor], max_rank: int, energy: float = 0.999, device: str | None = None) -> tuple[dict, dict]:
    """Shrink every layer to at most `max_rank`, keeping the directions holding `energy` of its
    change. Returns (state dict, stats {layers, mean_rank, kept_energy})."""
    layers, other = _split(sd)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, torch.Tensor] = {}
    ranks, kept = [], []
    for name, layer in layers.items():
        up, down = layer["up"].float(), layer["down"].float()
        tail = down.shape[2:]
        s = _scale(layer)
        u2 = up.reshape(up.shape[0], -1).to(dev) * s  # (out, r)
        d2 = down.reshape(down.shape[0], -1).to(dev)  # (r, in*k)
        qu, ru = torch.linalg.qr(u2)  # (out, r), (r, r)
        qd, rd = torch.linalg.qr(d2.T)  # (in*k, r), (r, r)
        uc, sv, vc = torch.linalg.svd(ru @ rd.T)
        e = (sv**2).cumsum(0) / (sv**2).sum().clamp(min=1e-20)
        k = int((e < energy).sum().item()) + 1
        k = max(1, min(k, max_rank, sv.shape[0]))
        root = sv[:k].sqrt()
        new_up = (qu @ uc[:, :k]) * root  # (out, k)
        new_down = (root[:, None] * vc[:k]) @ qd.T  # (k, in*k)
        out[f"{name}.lora_up.weight"] = new_up.reshape(up.shape[0], k, *up.shape[2:]).cpu()
        out[f"{name}.lora_down.weight"] = new_down.reshape(k, *down.shape[1:2], *tail).cpu() if tail else new_down.cpu()
        out[f"{name}.alpha"] = torch.tensor(float(k))
        ranks.append(k)
        kept.append(float(e[k - 1]))
    out.update(other)
    stats = {"layers": len(ranks), "mean_rank": sum(ranks) / max(1, len(ranks)),
             "kept_energy": sum(kept) / max(1, len(kept))}
    return out, stats


def merge_files(members: list[tuple[Path, float]], out: Path, max_rank: int = 0, energy: float = 0.999,
                dtype: torch.dtype = torch.bfloat16) -> dict:
    """Merge LoRA files (path, strength) into `out`; with max_rank > 0, shrink it too."""
    from safetensors.torch import load_file, save_file

    sds = [(load_file(str(p)), s) for p, s in members]
    merged = merge(sds)
    stats: dict = {"rank": None}
    if max_rank > 0:
        merged, st = resize(merged, max_rank, energy)
        stats.update(st)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.safetensors")
    meta = {"zentrainer_merge": ", ".join(f"{p.name}@{s:g}" for p, s in members),
            "zentrainer_resize": f"max_rank={max_rank} energy={energy}" if max_rank else "none"}
    save_file({k: v.to(dtype).contiguous() for k, v in merged.items()}, str(tmp), metadata=meta)
    tmp.replace(out)
    stats["bytes"] = out.stat().st_size
    return stats
