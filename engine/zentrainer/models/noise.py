"""A model's own noise schedule, read from its file's header: what New run's Focus control shows
the training steps against (which noise levels, so which of composition / form / detail).

ComfyUI picks a model's config from its weights' names and shapes; a safetensors header lists
exactly those, so the config (and from it the sampling schedule) comes without loading a weight.
"""

from __future__ import annotations

import json
import struct
from functools import lru_cache
from pathlib import Path

STEPS = 201  # σ sampled at this many evenly spaced points of the schedule


class _Shape:
    """A tensor's stand-in for model detection: shape and dtype, no data."""

    def __init__(self, shape, dtype):
        import torch

        self.shape = torch.Size(shape)
        self.dtype = dtype

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def dim(self):
        return len(self.shape)

    def __getattr__(self, name):
        raise AttributeError(f"model detection read {name!r} of a weight")


def _header(path: Path) -> tuple[dict, dict | None]:
    import torch

    dtypes = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
              "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
              "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    meta = head.pop("__metadata__", None)
    return {k: _Shape(v["shape"], dtypes.get(v["dtype"], torch.float32)) for k, v in head.items()}, meta


def _config(path: str):
    """The model's config and its weights' stand-ins, as ComfyUI would detect them (or None)."""
    import comfy.model_detection as md
    import comfy.utils

    sd, meta = _header(Path(path))
    inner = comfy.utils.state_dict_prefix_replace(sd, {md.unet_prefix_from_state_dict(sd): ""}, filter_keys=True)
    sd = inner or sd  # as ComfyUI's loader: a checkpoint's diffusion model, or the file as is
    cfg = md.model_config_from_unet(sd, "", metadata=meta)
    return (cfg, sd) if cfg is not None else None


@lru_cache(maxsize=64)
def _profile(path: str, mtime: float) -> dict:
    from ..harness.objective import is_flow

    found = _config(path)
    if found is None:
        return {"known": False}
    cfg, sd = found
    import torch

    # flow models set their type in their model class, not the config: the model, built on the
    # meta device (no memory, no weights), carries the sampling schedule ComfyUI would use
    with torch.device("meta"):
        ms = cfg.get_model(sd, "", device="meta").model_sampling
    sigmas = [float(ms.percent_to_sigma(i / (STEPS - 1))) for i in range(STEPS)]
    return {"known": True, "model": type(cfg).__name__, "flow": bool(is_flow(ms)),
            "shift": cfg.sampling_settings.get("shift"), "sigmas": sigmas, "schedules": _schedules(path, STEPS - 1)}


@lru_cache(maxsize=64)
def _sampling(path: str):
    """The model's sampling schedule with data, for ComfyUI's schedulers (the meta model's has
    none: the same sampling class built again on the CPU), or None."""
    import torch

    found = _config(path)
    if found is None:
        return None
    cfg, sd = found
    try:
        with torch.device("meta"):
            ms = cfg.get_model(sd, "", device="meta").model_sampling
        with torch.device("cpu"):
            return type(ms)(cfg)
    except Exception:
        return None


def _schedules(path: str, steps: int, names: list[str] | None = None) -> dict[str, list[float]]:
    """ComfyUI's schedulers at `steps` steps on this model ({name: σ, full noise first}): the levels
    Timesteps "a scheduler" trains on, as objective.scheduler_sigmas gives the trainer."""
    import comfy.samplers

    from ..harness.objective import scheduler_sigmas

    ms = _sampling(path)
    if ms is None:
        return {}
    out = {}
    for name in names or comfy.samplers.SCHEDULER_NAMES:
        try:
            out[name] = [round(float(x), 5) for x in scheduler_sigmas(ms, name, steps)]
        except Exception:  # a scheduler this model can't run
            continue
    return out


def schedule(folder: str, name: str, scheduler: str, steps: int) -> list[float]:
    """One scheduler's exact σ at `steps` steps on this model, as training will use them ([] if unknown)."""
    import folder_paths

    path = folder_paths.get_full_path(folder, name)
    if not path or not path.endswith(".safetensors") or not 1 <= steps <= 1000:
        return []
    try:
        return _schedules(path, steps, [scheduler]).get(scheduler, [])
    except Exception:
        return []


def profile(folder: str, name: str) -> dict:
    """{known, model, flow, shift, sigmas, schedules}: the schedule's σ at evenly spaced points from full
    noise (first) to clean (last), as training draws them with no shift of its own."""
    import folder_paths

    path = folder_paths.get_full_path(folder, name)
    if not path or not path.endswith(".safetensors"):
        return {"known": False}
    try:
        return {**_profile(path, Path(path).stat().st_mtime), "folder": folder, "name": name}
    except Exception:
        return {"known": False}
