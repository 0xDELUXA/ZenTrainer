"""References for MOSS Image: input pictures join its sequence as clean tokens, the way Kontext and
Krea 2 take them, so a LoRA can learn edits that move things (which channel inputs can't).

ComfyUI's MOSS Image reads no `reference_latents` (ComfyUI PR #17296 adds them), so `apply` patches
a model to keep them from the conditioning (ComfyUI's ReferenceLatent node, or with_refs in
training) as `ref_latents`, and run a forward with them. Each reference sits on its own value of
the rope axis MOSS leaves at 0 (r = 1, 2, ...), on the picture's frame, its rows and columns
centered like the picture's: aligned on it, yet told apart. The two layouts are that PR's:
- "index_timestep_zero" (the default; ai-toolkit's MOSS-Image Edit with its kv cache, the PR's
  "ai-toolkit" format): the references are clean (t = 0) and attend only to each other, so a pass
  of their own gives each block's keys and values, which the picture and text attend to beside
  their own (through ComfyUI's attention override, so the blocks stay ComfyUI's).
- "index": the references ride in the sequence after the text, at the step's timestep, and
  attend to everything.

Training and ComfyUI's nodes (nodes.py) patch the model the same way, so a LoRA samples as it
trained."""

from __future__ import annotations

import torch

KEY = "zentrainer_moss_refs"
METHODS = ("index_timestep_zero", "index")
DEFAULT = "index_timestep_zero"
_KV = "zentrainer_moss_ref_kv"


def supports(model) -> bool:
    """A ComfyUI MOSS Image model (BaseModel): these references are for it."""
    dm = getattr(model, "diffusion_model", None)
    return type(model).__name__ == "MOSSImage" and hasattr(dm, "video_embedder") and hasattr(dm, "rope")


def apply(patcher) -> None:
    """Patch a MOSS Image ModelPatcher (in place) to take reference latents."""
    import comfy.patcher_extension

    model = patcher.model
    if not supports(model):
        raise ValueError(f"MOSS references are for MOSS Image, not {type(model).__name__}")
    patcher.add_object_patch("extra_conds", _extra_conds(model))
    patcher.add_object_patch("extra_conds_shapes", _extra_conds_shapes)
    patcher.add_object_patch("memory_usage_factor_conds", ("ref_latents",))
    patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY)
    patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, _wrapper)


def _extra_conds(model):
    base = type(model).extra_conds.__get__(model)

    def extra_conds(**kwargs):
        import comfy.conds

        out = base(**kwargs)
        refs = kwargs.get("reference_latents", None)
        if refs is not None:
            out["ref_latents"] = comfy.conds.CONDList([model.process_latent_in(r) for r in refs])
            how = kwargs.get("reference_latents_method", None)
            if how is not None:
                out["ref_latents_method"] = comfy.conds.CONDConstant(how)
        return out
    return extra_conds


def _extra_conds_shapes(**kwargs):
    import math

    refs = kwargs.get("reference_latents", None)
    if refs is None:
        return {}
    return {"ref_latents": [1, 128, sum(math.prod(r.size()) for r in refs) // 128]}  # MOSS's 128 latent channels


def _wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
    refs = kwargs.pop("ref_latents", None)
    how = kwargs.pop("ref_latents_method", None) or DEFAULT
    if not refs:
        return executor(x, timestep, context, transformer_options, **kwargs)
    if how not in METHODS:
        raise ValueError(f"MOSS references join by {', '.join(METHODS)}, not {how!r}")
    if how == "index_timestep_zero":
        return forward_t0(executor.class_obj, x, timestep, context, refs, transformer_options)
    return forward(executor.class_obj, x, timestep, context, refs, transformer_options)


def rope(dm, h: int, w: int, text_len: int, ref_shapes: list, device, transformer_options={}, f_offset: float = 0.0):
    """MOSS's rope (dm.rope) for the picture, the text, then each reference: (r, f, h, w) ids, the
    picture on r = 0 after the text on f, reference k on r = k at the picture's f. The references
    keep the trained frequencies (Scale ROPE stretches only the picture), as in ComfyUI PR #17296.
    f_offset shifts the picture and text on f (-text_len: the picture at f = 0, where a
    reference pass of their own puts the references)."""
    import comfy.ldm.flux.math

    ro = transformer_options.get("rope_options", {})
    scales = (1.0, 1.0, ro.get("scale_y", 1.0), ro.get("scale_x", 1.0))
    ids = torch.zeros(h * w, 4, device=device, dtype=torch.float32)
    ids[:, 1] = text_len + f_offset + ro.get("shift_t", 0.0)
    ids[:, 2] = ((torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2) * float(scales[2] != 0) + ro.get("shift_y", 0.0)).repeat_interleave(w)
    ids[:, 3] = ((torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2) * float(scales[3] != 0) + ro.get("shift_x", 0.0)).repeat(h)
    text = torch.zeros(text_len, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_len, device=device, dtype=torch.float32) + f_offset
    ids = torch.cat([ids, text])[None]
    pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000 * (s or 1.0) ** (-d / (d - 2)))
                    for i, (d, s) in enumerate(zip(dm.rope_axes_dim, scales))], dim=-3)
    if ref_shapes:
        pe = torch.cat([pe, ref_pe(dm, ref_shapes, device, text_len + f_offset + ro.get("shift_t", 0.0))], dim=1)
    return pe.unsqueeze(1).transpose(1, 2).contiguous()


def ref_pe(dm, ref_shapes: list, device, f: float = 0.0):
    """The references' rope (before rope()'s unsqueeze): reference k on r = k, at f, its rows and
    columns centered, the trained frequencies."""
    import comfy.ldm.flux.math

    parts = []
    for k, (rh, rw) in enumerate(ref_shapes, 1):
        ids = torch.zeros(rh * rw, 4, device=device, dtype=torch.float32)
        ids[:, 0] = k
        ids[:, 1] = f
        ids[:, 2] = (torch.arange(rh, device=device, dtype=torch.float32) - (rh - 1) / 2).repeat_interleave(rw)
        ids[:, 3] = (torch.arange(rw, device=device, dtype=torch.float32) - (rw - 1) / 2).repeat(rh)
        parts.append(ids)
    ids = torch.cat(parts)[None]
    return torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000) for i, d in enumerate(dm.rope_axes_dim)], dim=-3)


def forward(dm, x, timestep, context, refs, transformer_options={}):
    """"index": MOSSImage._forward (comfy/ldm/moss_image/model.py) with the references' tokens
    after the text. Keep in step with it."""
    import comfy.utils
    from comfy.ldm.flux.layers import timestep_embedding
    from comfy.ldm.moss_image.model import modulated_rms_norm

    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    refs = [comfy.utils.repeat_to_batch_size(r.to(device=device, dtype=dtype), B) for r in refs]
    text_len = context.shape[1]
    pe = rope(dm, H, W, text_len, [tuple(r.shape[-2:]) for r in refs], device, transformer_options)

    vec = dm.time_embedding(timestep_embedding(timestep, 256).to(dtype))
    mod = dm.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    head_shift, head_scale = dm.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

    txt = dm.text_embedder(context, pe[:, H * W:H * W + text_len], transformer_options)
    tokens = [dm.video_embedder(r.flatten(2).transpose(1, 2)) for r in refs]
    transformer_options["reference_image_num_tokens"] = [t.shape[1] for t in tokens]
    x = torch.cat([dm.video_embedder(x.flatten(2).transpose(1, 2)), txt] + tokens, dim=1)
    del txt, tokens

    x = _blocks(dm, x, mod, pe, transformer_options, vec)

    out = dm.final_linear_video(modulated_rms_norm(x[:, :H * W], dm.final_norm_video, head_scale, head_shift))
    return out.transpose(1, 2).reshape(B, dm.latent_channels, H, W)


def _attention_hook(prev):
    """ComfyUI's attention override: in the references' own pass, keep each block's keys and
    values for them; in the picture's, attend to them beside its own. Each block's options carry
    its own ("keep", list) or ("use", keys, values), so compiled blocks needn't specialize on
    which block they are. `prev`: the override that was there (the trainer's backend), still called."""

    def override(func, q, k, v, *args, **kwargs):
        kv = (kwargs.get("transformer_options") or {}).get(_KV)
        if kv is not None:  # a main block's (the text embedder's blocks carry none)
            if kv[0] == "keep":
                kv[1].append((k, v))
            else:
                k = torch.cat((k, comfy_batch(kv[1], k.shape[0]).to(k.dtype)), dim=1)
                v = torch.cat((v, comfy_batch(kv[2], v.shape[0]).to(v.dtype)), dim=1)
        if prev is not None:
            return prev(func, q, k, v, *args, **kwargs)
        return func(q, k, v, *args, **kwargs)
    return override


def comfy_batch(t, n: int):
    import comfy.utils

    return t.expand(n, *t.shape[1:]) if t.shape[0] == 1 else comfy.utils.repeat_to_batch_size(t, n)


def _blocks(dm, x, mod, pe, transformer_options, vec=None, own=None):
    """Every main block over x, as MOSSImage._forward runs them; each block its own copy of the
    options (with own(i) in it), so a checkpointed block's recompute (in backward) sees its own."""
    import comfy.model_prefetch

    device, dtype = x.device, x.dtype
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(dm.layers)
    transformer_options["block_type"] = "single"
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(dm.layers), device, transformer_options)
    comfy.model_prefetch.malloc_graph_begin(device)
    for i, block in enumerate(dm.layers):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
        to = {**transformer_options, "block_index": i, **(own(i) if own is not None else {})}
        if ("single_block", i) in blocks_replace:
            def block_wrap(args, block=block):
                return {"img": block(args["img"], args["mod"], args["pe"], args["transformer_options"])}
            x = blocks_replace[("single_block", i)]({"img": x, "vec": vec, "mod": mod, "pe": pe, "transformer_options": to}, {"original_block": block_wrap})["img"]
        else:
            x = block(x, mod, pe, to)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
    comfy.model_prefetch.malloc_graph_end()
    return x


def forward_t0(dm, x, timestep, context, refs, transformer_options={}):
    """"index_timestep_zero": the references' own pass at t = 0 (no gradient: the LoRA shapes their
    keys and values as at inference, and learns through the picture and text), then the picture's,
    attending to them in every block. As ComfyUI PR #17296's preprocess_refs and _forward."""
    import comfy.utils
    from comfy.ldm.flux.layers import timestep_embedding
    from comfy.ldm.moss_image.model import modulated_rms_norm

    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    text_len = context.shape[1]
    kept = [[] for _ in dm.layers]  # each block's (keys, values) for the references
    to = {**transformer_options,
          "optimized_attention_override": _attention_hook(transformer_options.get("optimized_attention_override"))}

    refs = [r.to(device=device, dtype=dtype) for r in refs]
    n = max(r.shape[0] for r in refs)
    with torch.no_grad():
        r = torch.cat([dm.video_embedder(comfy.utils.repeat_to_batch_size(r, n).flatten(2).transpose(1, 2)) for r in refs], dim=1)
        rpe = ref_pe(dm, [tuple(r.shape[-2:]) for r in refs], device).unsqueeze(1).transpose(1, 2).contiguous()
        rvec = dm.time_embedding(timestep_embedding(torch.zeros(1, device=device), 256).to(dtype))
        rmod = dm.shared_block_modulation(rvec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        # no block replacements (skip-layer guidance and the like are for the picture's batch, cond
        # and uncond), as the PR's preprocess_refs calls the blocks themselves
        own = {k: v for k, v in to.items() if k != "patches_replace"}
        _blocks(dm, r, rmod, rpe, own, rvec, lambda i: {_KV: ("keep", kept[i])})
        del r
    kv = [layer[-1] for layer in kept]
    to["reference_image_num_tokens"] = [rr.shape[-2] * rr.shape[-1] for rr in refs]

    pe = rope(dm, H, W, text_len, [], device, transformer_options, f_offset=-text_len)
    vec = dm.time_embedding(timestep_embedding(timestep, 256).to(dtype))
    mod = dm.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    head_shift, head_scale = dm.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    txt = dm.text_embedder(context, pe[:, H * W:], transformer_options)
    x = torch.cat((dm.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
    del txt
    x = _blocks(dm, x, mod, pe, to, vec, lambda i: {_KV: ("use", *kv[i])})
    out = dm.final_linear_video(modulated_rms_norm(x[:, :H * W], dm.final_norm_video, head_scale, head_shift))
    return out.transpose(1, 2).reshape(B, dm.latent_channels, H, W)
