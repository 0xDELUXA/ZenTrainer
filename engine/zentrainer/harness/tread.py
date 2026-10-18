"""TREAD token routing (Krause et al., arXiv 2501.04765) for MiniMax H3 training: on a video
step, a random share of the clip's tokens leaves the sequence before block `start` and rejoins,
unchanged, before block `end`, so the blocks in between run on fewer tokens. The trained LoRA is
an ordinary one: only training steps route; samples and validation always run whole. Fizgig
ships it on for H3 clips (half the video tokens, blocks 2 to 47).

It plugs into ComfyUI's own per-block replace hook (transformer_options patches_replace "dit",
("double_block", i)), whose args carry everything a block sees: the packed sequence (tokens,
channels), the per-token modulation segments [(start, end, row)], the rope table
(1, tokens, 1, ., 2, 2) and the sequence layout (which span is the video). Text, audio and
reference tokens always stay.
"""

from __future__ import annotations

import torch


def _remap_segments(segments: list, keep: torch.Tensor, va: int, vb: int, vkeep: torch.Tensor) -> list:
    """Modulation segments for the shortened sequence. Every non-video token stays, so those
    spans just move; the video span shrinks to the kept tokens (a per-row tensor is indexed)."""
    out = []
    dropped = (vb - va) - vkeep.numel()
    for a, b, row in segments:
        if b <= va:
            out.append((a, b, row))
        elif a >= vb:
            out.append((a - dropped, b - dropped, row))
        else:  # the video span (or a segment inside it)
            lo, hi = max(a, va), min(b, vb)
            inside = vkeep[(vkeep >= lo) & (vkeep < hi)]
            if inside.numel() == 0:
                continue
            start = va + int((vkeep < lo).sum())
            r = row[(inside - a)] if isinstance(row, torch.Tensor) and row.ndim > 0 else row
            out.append((start, start + inside.numel(), r))
    return out


def patches(n_blocks: int, share: float, generator: torch.Generator | None = None,
            start: int = 2, end: int | None = None) -> dict:
    """patches_replace["dit"] entries that route `share` of the video tokens around blocks
    [start, end). Build a fresh set per training step (it holds that step's routing)."""
    end = n_blocks - 3 if end is None else end
    if not 0.0 < share < 1.0 or not 0 <= start < end <= n_blocks:
        return {}
    st: dict = {}

    def enter(args, extra):
        h, layout = args["img"], args["layout"]
        video = next((s for s in layout.segments if s[2] == "video"), None)
        if video is None or video[1] - video[0] < 8:
            st.clear()
            return extra["original_block"](args)
        va, vb = int(video[0]), int(video[1])
        n_video = vb - va
        k = max(1, int(round(n_video * (1.0 - share))))
        pick = torch.randperm(n_video, generator=generator)[:k].sort().values.to(h.device) + va
        keep = torch.cat([torch.arange(0, va, device=h.device), pick, torch.arange(vb, h.shape[0], device=h.device)])
        st.update(full=h, keep=keep, segments=_remap_segments(args["mod_segments"], keep, va, vb, pick),
                  rope=args["rope_freqs"].index_select(1, keep) if args["rope_freqs"] is not None else None)
        return inner(dict(args, img=h.index_select(0, keep)), extra)

    def inner(args, extra):
        if not st:
            return extra["original_block"](args)
        return extra["original_block"](dict(args, mod_segments=st["segments"], rope_freqs=st["rope"]))

    def leave(args, extra):
        if not st:
            return extra["original_block"](args)
        h = st["full"].index_copy(0, st["keep"], args["img"])  # the routed tokens rejoin unchanged
        st.clear()
        return extra["original_block"](dict(args, img=h))

    out = {("double_block", start): enter, ("double_block", end): leave}
    for i in range(start + 1, end):
        out[("double_block", i)] = inner
    return out
