"""A .safetensors file's tensors and metadata, read from its header alone (no weights loaded,
so a 20 GB model opens at once), and saving a copy of one: another name, other metadata, float
weights in another precision, some tensors left out.

The copy streams: tensors are written one at a time straight from the source file, so a model
never has to fit in memory. Quantized weights (int8, fp8) and their scales are copied as they
are; a precision change applies only to plain float weights.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
from pathlib import Path

FOLDERS = ("loras", "diffusion_models", "checkpoints", "vae", "text_encoders")

BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8,
         "U32": 4, "U16": 2, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
FLOATS = ("F64", "F32", "F16", "BF16")
DTYPES = {"fp32": "F32", "fp16": "F16", "bf16": "BF16"}
# sidecars of quantized layers: they stay as they were saved, the loaders expect their precision
QUANT_SIDECARS = re.compile(r"(scale|scale_weight|scale_input|weight_scale|input_scale|comfy_quant)$")
MAX_HEADER = 100 * 1024 * 1024


def read_header(path: str | os.PathLike) -> tuple[dict[str, str], dict[str, dict], int]:
    """(metadata, {name: {dtype, shape, data_offsets}}, where the data starts)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > MAX_HEADER:
            raise ValueError("not a safetensors file (its header is too large)")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise ValueError("not a safetensors file")
    meta = header.pop("__metadata__", None) or {}
    return {str(k): str(v) for k, v in meta.items()}, header, 8 + n


def _natural(name: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _params(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _kind(names: list[str], tensors: dict[str, dict]) -> dict:
    """LoRA (and its algorithm and ranks) or a model, and whether it's quantized."""
    out: dict = {"kind": "model"}
    downs = [n for n in names if n.endswith((".lora_down.weight", ".lora_A.weight"))]
    if downs:
        ranks = sorted({tensors[n]["shape"][0] for n in downs})
        out.update(kind="lora", algorithm="LoRA", layers=len(downs), ranks=ranks)
    else:
        for mark, algorithm in ((".lokr_", "LoKr"), (".hada_", "LoHa")):
            layers = {n.split(mark)[0] for n in names if mark in n}
            if layers:
                out.update(kind="lora", algorithm=algorithm, layers=len(layers))
                break
    dtypes = {tensors[n]["dtype"] for n in names}
    out["quantized"] = bool(dtypes & {"I8", "U8", "F8_E4M3", "F8_E5M2"}) and out["kind"] == "model"
    return out


def inspect(path: str | os.PathLike) -> dict:
    """What the file holds: every tensor (name, dtype, shape, params), its metadata, and totals."""
    meta, tensors, _ = read_header(path)
    names = sorted(tensors, key=_natural)
    rows, by_dtype = [], {}
    total = 0
    for n in names:
        t = tensors[n]
        p = _params(t["shape"])
        total += p
        d = by_dtype.setdefault(t["dtype"], {"tensors": 0, "params": 0, "bytes": 0})
        d["tensors"] += 1
        d["params"] += p
        d["bytes"] += t["data_offsets"][1] - t["data_offsets"][0]
        rows.append({"name": n, "dtype": t["dtype"], "shape": t["shape"], "params": p})
    return {"size": os.path.getsize(path), "params": total, "by_dtype": by_dtype, "metadata": meta,
            "tensors": rows, **_kind(names, tensors)}


def _dropped(name: str, drop: list[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in drop)


def _convertible(name: str, dtype: str) -> bool:
    return dtype in FLOATS and not QUANT_SIDECARS.search(name)


def plan(src: str | os.PathLike, dtype: str | None = None, drop: list[str] | None = None) -> dict:
    """What a copy would hold: {tensors, bytes, converted, dropped}."""
    _, tensors, _ = read_header(src)
    drop = drop or []
    to = DTYPES.get(dtype or "")
    kept = conv = 0
    size = 0
    for n, t in tensors.items():
        if _dropped(n, drop):
            continue
        kept += 1
        nbytes = t["data_offsets"][1] - t["data_offsets"][0]
        if to and t["dtype"] != to and _convertible(n, t["dtype"]):
            nbytes = _params(t["shape"]) * BYTES[to]
            conv += 1
        size += nbytes
    return {"tensors": kept, "bytes": size, "converted": conv, "dropped": len(tensors) - kept}


def save_copy(src: str | os.PathLike, dst: str | os.PathLike, metadata: dict[str, str] | None = None,
              dtype: str | None = None, drop: list[str] | None = None, overwrite: bool = False) -> dict:
    """Write a copy of `src` at `dst`. metadata None keeps the file's; a dict replaces it. dtype
    'bf16' / 'fp16' / 'fp32' recasts plain float weights; `drop` leaves out the tensors named, or
    everything under a prefix ("diffusion_model.layers.22")."""
    import torch

    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("pick a new name: a copy can't replace the file it's read from")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst.name} already exists")
    if dtype and dtype not in DTYPES:
        raise ValueError(f"unknown precision {dtype!r}")
    old_meta, tensors, start = read_header(src)
    drop = drop or []
    to = DTYPES.get(dtype or "")
    torch_dtype = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}

    names = [n for n in sorted(tensors, key=lambda n: tensors[n]["data_offsets"][0]) if not _dropped(n, drop)]
    if not names:
        raise ValueError("that leaves no tensors to save")
    header: dict = {}
    meta = old_meta if metadata is None else {str(k): str(v) for k, v in metadata.items()}
    if meta:
        header["__metadata__"] = meta
    jobs, offset, converted = [], 0, 0
    for n in names:
        t = tensors[n]
        a, b = t["data_offsets"]
        out_dtype, nbytes = t["dtype"], b - a
        if to and t["dtype"] != to and _convertible(n, t["dtype"]):
            out_dtype, nbytes = to, _params(t["shape"]) * BYTES[to]
            converted += 1
        header[n] = {"dtype": out_dtype, "shape": t["shape"], "data_offsets": [offset, offset + nbytes]}
        jobs.append((n, start + a, b - a, t["dtype"], out_dtype, t["shape"]))
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # the data starts 8-byte aligned

    dst.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst.parent).free
    if free < offset + len(raw) + 256 * 1024 * 1024:
        raise OSError(f"not enough disk space: the copy needs {(offset + len(raw)) / 2**30:.1f} GB, "
                      f"{free / 2**30:.1f} GB is free")
    tmp = dst.with_name(f".{dst.stem}.part")  # hidden, and not a .safetensors ComfyUI would list
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            fout.write(struct.pack("<Q", len(raw)))
            fout.write(raw)
            for _name, at, nbytes, have, want, shape in jobs:
                fin.seek(at)
                if have == want:
                    left = nbytes
                    while left:
                        chunk = fin.read(min(left, 64 * 1024 * 1024))
                        if not chunk:
                            raise OSError(f"{src.name} ended early")
                        fout.write(chunk)
                        left -= len(chunk)
                elif nbytes:
                    t = torch.frombuffer(bytearray(fin.read(nbytes)), dtype=torch_dtype[have]).reshape(shape)
                    fout.write(t.to(torch_dtype[want]).reshape(-1).view(torch.uint8).numpy().tobytes())
            fout.flush()
            os.fsync(fout.fileno())
        os.chmod(tmp, 0o644)
        if dst.exists() and not overwrite:  # another save of the same name finished first
            raise FileExistsError(f"{dst.name} already exists")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return {"bytes": dst.stat().st_size, "tensors": len(names), "converted": converted,
            "dropped": len(tensors) - len(names)}
