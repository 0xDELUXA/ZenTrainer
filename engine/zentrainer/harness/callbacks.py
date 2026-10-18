"""Things that happen around steps: metrics, checkpoints, samples.

A callback gets the trainer at `on_start`, after every optimizer step, and at
`on_end`. New behaviour (EMA, validation loss, early stopping) is a new class here,
not a change to the loop.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .. import events
from .streams import first_stream

SAMPLE_HEADROOM = 6 * 1024 ** 3  # extra VRAM kept free while an offloaded model samples (see render)

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback:
    def on_start(self, t: Trainer) -> None: ...
    def on_step(self, t: Trainer) -> None: ...
    def on_end(self, t: Trainer, status: str) -> None: ...


class Metrics(Callback):
    """One row per optimizer step. `train_time` counts only optimizer steps; `elapsed` is
    wall-clock for this session; `it_s` is from the step itself, so a sample render or a
    validation pass doesn't show up as a slowdown."""

    def __init__(self):
        self._t0 = 0.0
        self._rate: float | None = None

    def on_start(self, t):
        self._t0 = time.monotonic()

    def on_step(self, t):
        now = time.monotonic()
        dt = t.last_step_seconds or 0.0
        inst = 1.0 / dt if dt > 0 else 0.0
        self._rate = inst if self._rate is None else 0.8 * self._rate + 0.2 * inst  # smoothed
        row = {
            "run_id": t.run_id,
            "step": t.step,
            "total": t.spec.optim.steps,
            "loss": t.last_loss,
            "lr": t.optimizer.mean_lr() if hasattr(t.optimizer, "mean_lr") else t.optimizer.param_groups[0]["lr"],
            "grad_norm": t.last_grad_norm,
            "epoch": t.batches.epoch,
            "elapsed": round(now - self._t0, 3),
            "train_time": round(t.train_seconds, 3),
            "step_time": round(dt, 4),
            "it_s": round(self._rate, 3),
        }
        if torch.cuda.is_available():
            # the step's peak VRAM (PyTorch's allocations: weights resident, activations, LoRA,
            # optimizer): what the offloading headroom has to cover
            row["vram_peak"] = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        t.emit(events.METRICS, row)
        if t.store:
            t.store.append(t.run_id, "metrics", row)


class Checkpoints(Callback):
    """LoRA files ComfyUI can load, plus a resumable training state alongside."""

    def __init__(self, every: int):
        self.every = every

    def on_step(self, t):
        if self.every and t.step % self.every == 0 and t.step < t.spec.optim.steps:
            t.save(f"step{t.step:06d}")

    def on_end(self, t, status):
        if t.step > 0:
            t.save("final" if status == "done" else f"{status}-step{t.step:06d}", final=status == "done")


class _OptionsOnly:
    """Enough of a ModelPatcher for ComfyUI's guidance nodes to add their sampler options to:
    clone() is itself and the setters only touch a copy of the options."""

    def __init__(self, patcher):
        self._patcher = patcher
        self.model_options = patcher.model_options.copy()

    def clone(self):
        return self

    def get_model_object(self, name):
        return self._patcher.get_model_object(name)

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        import comfy.model_patcher

        self.model_options = comfy.model_patcher.set_model_options_post_cfg_function(self.model_options, fn, disable_cfg1_optimization)

    def set_model_sampler_calc_cond_batch_function(self, fn):
        self.model_options = {**self.model_options, "sampler_calc_cond_batch_function": fn}


class Samples(Callback):
    """Render the sample prompts with the in-progress weights, in-process.

    Uses ComfyUI's own sampler on the training patcher (adapters applied), then
    restores the training state. If this proves fragile on some families, the
    fallback is a separate sample job against the last checkpoint.
    """

    def __init__(self, every: int):
        self.every = every
        self._conds = None
        self._alt = None  # base patcher of the alternate sampling model, loaded once

    def on_start(self, t):
        # encode even when unscheduled: a "sample now" command can still ask for them
        with torch.no_grad():
            self._encode(t, t.spec.samples)
        if t.spec.samples.baseline and self._conds and t.step == 0:
            self.render(t)  # the base model's look, before any training

    # ---- baselines: the untrained model's samples depend only on the model and the sample
    # settings, so a run that shares them with an earlier one copies its files instead of
    # rendering them again (output/zentrainer/samples/_baseline/<key>.<ext>)
    def _baseline_key(self, t, i: int) -> str:
        import hashlib

        m, s, it = t.spec.model, t.spec.samples, self.items[i]
        parts = [m.kind, m.ckpt_name, m.unet_name, m.unet_weight_dtype, m.clip_name, m.clip_type, m.vae_name,
                 getattr(m, "audio_vae_name", None), s.model_name, s.model_weight_dtype, t.spec.network.init_from,
                 self.prompts[i], s.negative, it.width, it.height, it.frames, s.fps, it.seed, s.steps, s.cfg,
                 s.sampler, s.scheduler, s.slg_mode, s.slg_layers, s.slg_double_layers, s.slg_scale, s.slg_start,
                 s.slg_end, s.lora_strength if t.spec.network.init_from else None]
        return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:20]

    def _baseline_dir(self) -> Path:
        import folder_paths

        return Path(folder_paths.get_output_directory()) / "zentrainer" / "samples" / "_baseline"

    def _baseline_hit(self, t, i: int, out_dir: str) -> dict | None:
        import shutil

        key = self._baseline_key(t, i)
        for src in self._baseline_dir().glob(f"{key}.*"):
            name = f"step000000_{i:02d}{src.suffix}"
            shutil.copyfile(src, Path(out_dir) / name)
            return {"run_id": t.run_id, "step": 0, "index": i, "prompt": self.prompts[i], "filename": name,
                    "subfolder": f"zentrainer/samples/{t.run_id}", "type": "output", "model": t.spec.samples.model_name or None,
                    "reused": True}
        return None

    def _baseline_keep(self, t, i: int, path: Path) -> None:
        import shutil

        try:
            d = self._baseline_dir()
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, d / f"{self._baseline_key(t, i)}{path.suffix}")
        except OSError:
            pass

    def _sampling_model(self, t):
        """The training patcher itself, or the alternate model with the current LoRA."""
        s = t.spec.samples
        if not s.model_name:
            return t.models.patcher
        import comfy.sd
        import folder_paths

        if self._alt is None:
            if t.spec.model.kind == "split":
                opts = {}
                if s.model_weight_dtype.startswith("fp8_e4m3fn"):
                    opts["dtype"] = torch.float8_e4m3fn
                elif s.model_weight_dtype == "fp8_e5m2":
                    opts["dtype"] = torch.float8_e5m2
                path = folder_paths.get_full_path_or_raise("diffusion_models", s.model_name)
                self._alt = comfy.sd.load_diffusion_model(path, model_options=opts)
            else:
                path = folder_paths.get_full_path_or_raise("checkpoints", s.model_name)
                self._alt = comfy.sd.load_checkpoint_guess_config(path, output_vae=False, output_clip=False)[0]
        patched, _ = comfy.sd.load_lora_for_models(self._alt, None, t.network.state_dict(), s.lora_strength, 0)
        return patched

    def _guided_options(self, model, s) -> dict | None:
        """The sampling model's options with skip-layer guidance added, or None for none.
        Runs ComfyUI's own SkipLayerGuidanceDiT(Simple) on a stand-in that only collects
        options: cloning the training patcher would reload its weights for every sample."""
        if s.slg_mode == "off" or not (s.slg_layers or s.slg_double_layers):
            return None
        from comfy_extras import nodes_slg

        stub = _OptionsOnly(model)
        single, double = ", ".join(map(str, s.slg_layers)), ", ".join(map(str, s.slg_double_layers))
        if s.slg_mode == "uncond":
            nodes_slg.SkipLayerGuidanceDiTSimple.execute(stub, s.slg_start, s.slg_end, double, single)
        else:
            nodes_slg.SkipLayerGuidanceDiT.execute(stub, s.slg_scale, s.slg_start, s.slg_end, double, single)
        return stub.model_options

    def _encode(self, t, s):
        from .data import apply_trigger

        from ..spec import sample_prompts

        trig = t.spec.dataset.trigger
        self.items = sample_prompts(s)
        # samples only substitute [trigger]; they're never rewritten otherwise
        self.prompts = [p.prompt.replace("[trigger]", trig) if trig else p.prompt.replace("[trigger]", "") for p in self.items]
        negative = apply_trigger(s.negative, trig, False)
        self._conds = self._encode_texts(t, self.prompts, negative)

    def _encode_texts(self, t, prompts: list[str], negative: str) -> list:
        """[(cond, negative cond)] for these prompts; cached like captions, so a run with
        nothing new to encode never loads the text encoder."""
        path = _prompt_cache(t, prompts, negative)
        if path is not None and path.is_file():
            try:
                cached = torch.load(path, weights_only=False)
                return [(c, cached["negative"]) for c in cached["prompts"]]
            except Exception:
                pass
        clip = t.models.clip
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
        conds = [(clip.encode_from_tokens_scheduled(clip.tokenize(p)), neg) for p in prompts]
        if path is not None:
            from .data import _to_cpu

            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"prompts": [_to_cpu(c) for c, _ in conds], "negative": _to_cpu(neg)}, path)
        return conds

    def render_manual(self, t, prompt: str, width: int | None = None, height: int | None = None,
                      frames: int | None = None, seed: int | None = None, base: bool = False) -> list[dict]:
        """One sample of any prompt at the current step ("Try a prompt"): rendered like the
        run's samples but kept apart from them (manual rows, their own files)."""
        from ..spec import SamplePrompt
        from .data import apply_trigger

        if self._conds is None:
            self.on_start(t)
        s, trig = t.spec.samples, t.spec.dataset.trigger
        text = prompt.replace("[trigger]", trig) if trig else prompt.replace("[trigger]", "")
        with torch.no_grad():
            conds = self._encode_texts(t, [text], apply_trigger(s.negative, trig, False))
        if hasattr(t, "_drop_text_encoder"):
            t._drop_text_encoder()  # don't keep a big encoder beside training for one prompt
        n = len({f.name.split("_")[1] for f in Path(_out_dir(t)).glob("manual_*")})
        item = SamplePrompt(prompt, width or s.width, height or s.height, frames or s.frames,
                            int(time.time()) % 100000 if seed is None else seed)
        saved = (self.items, self.prompts, self._conds)
        self.items, self.prompts, self._conds = [item], [text], conds
        try:
            rows = []
            if base:  # the same prompt and seed without the LoRA first, to compare against
                with t.network.disabled():
                    rows += self.render(t, manual=n, base=True)
            return rows + self.render(t, manual=n)
        finally:
            self.items, self.prompts, self._conds = saved

    def on_step(self, t):
        if self.every and t.step % self.every == 0:
            self.render(t)

    def render(self, t: Trainer, manual: int | None = None, base: bool = False) -> list[dict]:
        if self._conds is None:
            self.on_start(t)
        if not self._conds:
            t.emit(events.STATUS, {"run_id": t.run_id, "status": "training", "step": t.step,
                                   "message": "sample requested, but the run has no sample prompts"})
            return []
        import comfy.model_management as mm
        import comfy.sample
        import folder_paths
        from PIL import Image

        s = t.spec.samples
        vae = t.models.vae
        comp = vae.spacial_compression_encode()
        out_dir = os.path.join(folder_paths.get_output_directory(), "zentrainer", "samples", t.run_id)
        os.makedirs(out_dir, exist_ok=True)
        rows = []

        if manual is None and t.step == 0 and all(any(self._baseline_dir().glob(f"{self._baseline_key(t, i)}.*")) for i in range(len(self._conds))):
            # every baseline rendered before with these exact settings: copy them, load nothing
            for i in range(len(self._conds)):
                row = self._baseline_hit(t, i, out_dir)
                rows.append(row)
                t.emit(events.SAMPLE, row)
                if t.store:
                    t.store.append(t.run_id, "samples", row)
            return rows

        previewer = _previewer(t)
        count = len(self._conds)

        def progress(i, s_step, s_total, x0=None):
            row = {"run_id": t.run_id, "step": t.step, "phase": "sampling", "index": i, "count": count,
                   "prompt": self.prompts[i], "s_step": s_step, "s_total": s_total}
            if x0 is not None and previewer is not None:
                row["preview"] = _preview_url(previewer, x0)
            t.emit(events.SAMPLE_PROGRESS, row)

        guided = plain_options = model = None
        t.enter_inference()
        # ComfyUI sizes how much of the model to load for sampling by its own estimate, which
        # doesn't count our LoRA wrappers building each layer's full weight as it runs: with the
        # model offloaded, keep the same room free as training does
        reserved = mm.EXTRA_RESERVED_VRAM
        if t.spec.memory.offloading:
            mm.EXTRA_RESERVED_VRAM = reserved + SAMPLE_HEADROOM
        t.network.assist = False  # samples show the LoRA on the real model, not the training adapter
        try:
            with torch.no_grad():
                model = self._sampling_model(t)
                guided = self._guided_options(model, s)
                plain_options = model.model_options
                if guided is not None:
                    model.model_options = guided  # put back in finally
                for i, (pos, neg) in enumerate(self._conds):
                    reused = self._baseline_hit(t, i, out_dir) if t.step == 0 and manual is None else None
                    if reused is not None:
                        rows.append(reused)
                        t.emit(events.SAMPLE, reused)
                        if t.store:
                            t.store.append(t.run_id, "samples", reused)
                        continue
                    progress(i, 0, s.steps)
                    channels = model.model.latent_format.latent_channels
                    it = self.items[i]
                    t_lat = _latent_frames(vae, it.frames)
                    latent = torch.zeros([1, channels] + ([t_lat] if t_lat else []) + [it.height // comp, it.width // comp])
                    latent = comfy.sample.fix_empty_latent_channels(model, latent)
                    noise = comfy.sample.prepare_noise(latent, it.seed)
                    def cb(step, x0, _x, total, _i=i):
                        progress(_i, step + 1, total, x0)

                    # as in training: bypass adapters keep their own (fp32) weights beside the
                    # model's bf16 activations, and only autocast reconciles the two
                    with torch.autocast(t.device.type, dtype=t.train_dtype, enabled=t.train_dtype != torch.float32):
                        out = comfy.sample.sample(model, noise, s.steps, s.cfg, s.sampler, s.scheduler,
                                                  pos, neg, latent, seed=it.seed, disable_pbar=True, callback=cb)
                    img = vae.decode(first_stream(out))
                    img = img.reshape(-1, *img.shape[-3:])  # (frames, H, W, 3)
                    if t_lat:
                        from . import video

                        name = f"manual_{manual:03d}_{'base' if base else f'step{t.step:06d}'}.mp4" if manual is not None else f"step{t.step:06d}_{i:02d}.mp4"
                        video.write(Path(out_dir) / name, img, s.fps, _sound(t, out))
                    else:
                        name = f"manual_{manual:03d}_{'base' if base else f'step{t.step:06d}'}.png" if manual is not None else f"step{t.step:06d}_{i:02d}.png"
                        Image.fromarray((img[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")).save(
                            os.path.join(out_dir, name))
                    if t.step == 0 and manual is None:
                        self._baseline_keep(t, i, Path(out_dir) / name)
                    row = {
                        "run_id": t.run_id, "step": t.step, "index": i, "prompt": self.prompts[i],
                        "filename": name, "subfolder": f"zentrainer/samples/{t.run_id}", "type": "output",
                        "model": s.model_name or None,
                    }
                    if manual is not None:
                        row.update(manual=True, index=manual, width=self.items[i].width, height=self.items[i].height,
                                   seed=self.items[i].seed, base=base)
                    # recorded as each one lands, so the UI swaps that tile now, not after the pass
                    rows.append(row)
                    t.emit(events.SAMPLE, row)
                    if t.store:
                        t.store.append(t.run_id, "samples", row)
        finally:
            t.network.assist = True
            mm.EXTRA_RESERVED_VRAM = reserved
            if guided is not None:
                model.model_options = plain_options
            t.exit_inference()
            mm.soft_empty_cache()
            t.emit(events.SAMPLE_PROGRESS, {"run_id": t.run_id, "step": t.step, "phase": "done", "count": count})

        return rows


class Validation(Callback):
    """Loss on a fixed set: the same images at the same noise levels with the same noise
    every time, so the curve moves only when the LoRA does. Runs before training too."""

    def __init__(self):
        self._set: list[tuple[int, list[float], list[torch.Tensor]]] = []

    def on_start(self, t):
        v = t.spec.validation
        if not v.every:
            return
        import random

        rng = random.Random(v.seed)
        pool = list(range(len(t.dataset.samples)))
        picks = rng.sample(pool, min(v.images, len(pool)))
        gen = torch.Generator().manual_seed(v.seed)
        percents = [1.0 - (k + 0.5) / v.timesteps for k in range(v.timesteps)]  # spread over the schedule
        for i in picks:
            shape = t.streams.packed_shape(tuple(t.dataset.samples[i].latent.shape))
            noises = [torch.randn(shape, generator=gen) for _ in percents]
            self._set.append((i, percents, noises))
        if t.step == 0:
            self.evaluate(t)

    def on_step(self, t):
        if self._set and t.step % t.spec.validation.every == 0:
            self.evaluate(t)

    def on_end(self, t, status):
        if self._set and status == "done" and t.step % t.spec.validation.every:
            self.evaluate(t)

    def evaluate(self, t) -> float | None:
        if not self._set:
            return None
        from . import optim

        ms = t.denoiser.model_sampling
        losses, per_t = [], {}
        optim.set_mode(t.optimizer, training=False)
        try:
            self._run(t, ms, losses, per_t)
        finally:
            optim.set_mode(t.optimizer, training=True)
        return self._report(t, losses, per_t)

    def _run(self, t, ms, losses, per_t) -> None:
        with torch.no_grad():
            for i, percents, noises in self._set:
                x0, weight, shapes = t.batch([i])
                cond = t.fixed_cond(i, x0, shapes)
                for p, noise in zip(percents, noises):
                    sigma = torch.tensor([float(ms.percent_to_sigma(p))], device=t.device)
                    x_t = t.objective.noisy(ms, sigma, noise.to(t.device), x0)
                    with torch.autocast(t.device.type, dtype=t.train_dtype, enabled=t.train_dtype != torch.float32):
                        pred = t.denoiser(x_t, sigma, [cond])
                    loss = float(t.objective.loss(x_t, x0, pred, sigma, weight).mean())
                    losses.append(loss)
                    per_t.setdefault(round(1 - p, 3), []).append(loss)

    def _report(self, t, losses, per_t) -> float:
        val = sum(losses) / len(losses)
        row = {"run_id": t.run_id, "step": t.step, "val_loss": val,
               "by_noise": {str(k): sum(v) / len(v) for k, v in sorted(per_t.items())}}
        t.emit(events.VALIDATION, row)
        if t.store:
            t.store.append(t.run_id, "validation", row)
        return val


def _prompt_cache(t, prompts: list[str], negative: str):
    import hashlib

    enc, root = getattr(t, "encoder_id", None), getattr(t, "cache_dir", None)
    if enc is None or root is None:
        return None
    key = hashlib.sha256(json.dumps([enc, prompts, negative]).encode()).hexdigest()[:20]
    return root / "prompts" / f"{key}.pt"


def _previewer(t):
    """ComfyUI's latent previewer for this model: TAESD when its decoder is installed and
    previews are set to TAESD, else the cheap latent-to-RGB projection."""
    try:
        import latent_preview

        fmt = t.models.patcher.model.latent_format
        previewer = latent_preview.get_previewer(t.device, fmt)
        # ComfyUI launches with previews off by default; ours always want one
        if previewer is None and fmt.latent_rgb_factors is not None:
            previewer = latent_preview.Latent2RGBPreviewer(
                fmt.latent_rgb_factors, fmt.latent_rgb_factors_bias, fmt.latent_rgb_factors_reshape)
        return previewer
    except Exception:
        return None


def _preview_url(previewer, x0) -> str | None:
    import base64
    import io

    try:
        x0 = first_stream(x0)[:1]
        if x0.ndim == 5:  # a video latent: its first frame
            x0 = x0[:, :, 0]
        img = previewer.decode_latent_to_preview(x0)
        img.thumbnail((384, 384))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def write_metadata(spec_dict: dict, step: int) -> dict[str, str]:
    return {"zentrainer.spec": json.dumps(spec_dict), "zentrainer.step": str(step)}


def _latent_frames(vae, frames: int) -> int:
    """Latent frames for a `frames`-long sample from a video VAE, 0 for a still (or an image VAE)."""
    down = getattr(vae, "downscale_ratio", None)
    if frames <= 1 or not (isinstance(down, tuple) and callable(down[0])):
        return 0
    return int(down[0](frames))


def _sound(t, out):
    """A joint audio-video sample's soundtrack ((channels, samples), rate), when the run has the
    model's audio VAE; else None (the sample is saved silent)."""
    avae = getattr(t.models, "audio_vae", None)
    if avae is None or not getattr(out, "is_nested", False):
        return None
    streams = out.unbind()
    if len(streams) < 2:
        return None
    wave = avae.decode(streams[-1]).movedim(-1, 1)[:1]  # (1, channels, samples)
    std = torch.std(wave, dim=[1, 2], keepdim=True) * 5.0  # as ComfyUI's VAEDecodeAudio levels it
    wave = wave / std.clamp(min=1.0)
    rate = getattr(avae, "audio_sample_rate_output", getattr(avae, "audio_sample_rate", 44100))
    return wave[0].float().cpu(), int(rate)


def _out_dir(t) -> str:
    import folder_paths

    d = os.path.join(folder_paths.get_output_directory(), "zentrainer", "samples", t.run_id)
    os.makedirs(d, exist_ok=True)
    return d
