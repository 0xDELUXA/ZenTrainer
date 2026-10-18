"""Which parts of each loaded model are in VRAM right now.

ComfyUI's dynamic VRAM (comfy_aimdo) gives a model a virtual address range (its vbar) and
streams weights into it page by page (32 MB pages); `get_residency()` flags each page as in
VRAM or not. Each layer it streams remembers its slice of that range (`module._v`), so a
block's share in VRAM is its layers' pages that are resident. Layers loaded outright (small
ones, or a model loaded without dynamic VRAM) count by where their weight is.
"""

from __future__ import annotations

import re

PAGE = 32 << 20  # comfy_aimdo's VBAR_PAGE_SIZE
_BLOCK = re.compile(r"^(.*?\.\d+)(?:\.|$)")  # "diffusion_model.blocks.12.attn.qkv" -> "...blocks.12"


def _group(name: str) -> str:
    m = _BLOCK.match(name)
    return m.group(1) if m else (name.split(".")[0] or "model")


def report() -> list[dict]:
    import comfy.model_management as mm

    out = []
    for lm in list(mm.current_loaded_models):
        patcher = lm.model
        if patcher is None:
            continue
        try:
            out.append(_model(patcher))
        except Exception as e:  # a model mid-load or mid-unload: report what we can
            out.append({"name": type(patcher.model).__name__, "error": str(e)})
    return out


def _model(patcher) -> dict:
    model = patcher.model
    vbar = patcher._vbar_get() if patcher.is_dynamic() else None
    resident = vbar.get_residency() if vbar is not None else None
    groups: dict[str, list[float]] = {}  # name -> [bytes, bytes in VRAM]
    for name, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is None or not hasattr(w, "nbytes") or any(True for _ in m.children()):
            continue
        g = groups.setdefault(_group(name), [0.0, 0.0])
        v = getattr(m, "_v", None)
        if v is not None and resident is not None:
            _, addr, size = v
            first, last = (addr - vbar.base_addr) // PAGE, (addr - vbar.base_addr + max(size, 1) - 1) // PAGE
            pages = range(int(first), min(int(last) + 1, len(resident)))
            share = sum(resident[p] & 1 for p in pages) / max(len(pages), 1)
            g[0] += size
            g[1] += size * share
        else:
            size = w.nbytes + (m.bias.nbytes if getattr(m, "bias", None) is not None else 0)
            g[0] += size
            g[1] += size if w.device.type == "cuda" else 0
    total = sum(b for b, _ in groups.values())
    in_vram = sum(v for _, v in groups.values())
    return {
        "name": type(model).__name__,
        "dynamic": vbar is not None,
        "bytes": int(total),
        "vram": int(in_vram),
        "groups": [{"name": k, "bytes": int(b), "vram": int(v)} for k, (b, v) in groups.items() if b > 0],
    }
