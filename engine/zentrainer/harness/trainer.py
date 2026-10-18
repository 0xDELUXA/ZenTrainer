"""The training loop. Owns the run from model load to final save.

ComfyUI is used as a library: its loaders, ModelPatcher and memory management, weight
adapters, attention backends, model_sampling and samplers. Nothing here goes through
the node graph. The only graph involvement is the ZenTrainSession node that hosts
the session, so a run holds ComfyUI's execution lane (and its GPU) while it trains.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .. import events
from ..spec import TrainSpec, slugify
from . import control as control_mod
from . import data, models, optim
from .callbacks import Callback, Checkpoints, Metrics, Samples, Validation, write_metadata
from . import compat
from .streams import Streams
from .denoiser import Denoiser
from .memory import attention_override, gradient_checkpointing
from . import network as network_mod
from .network import Network
from .objective import Objective, is_flow, sample_percents, sigmas_for

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _oom_advice(spec: TrainSpec, step: int) -> str:
    m, d, o = spec.memory, spec.dataset, spec.optim
    tips = []
    if not m.gradient_checkpointing:
        tips.append("turn on gradient checkpointing")
    elif m.checkpoint_skip:
        tips.append(f"keep fewer blocks un-checkpointed ({m.checkpoint_skip} -> 0)")
    if not m.offloading:
        tips.append("turn on weight offloading")
    if d.resolution > 768:
        tips.append(f"lower resolution ({d.resolution} -> {d.resolution * 3 // 4 // 64 * 64})")
    if d.video_frames > 22:
        tips.append(f"shorter video clips ({d.video_frames} frames) or a lower video resolution ({d.video_resolution})")
    if m.offloading:
        tips.append("more VRAM headroom (Memory -> VRAM headroom, in GB)")
    if o.batch_size > 1:
        tips.append(f"batch size {o.batch_size} -> 1 (use grad accumulation instead)")
    if m.training_dtype == "fp32":
        tips.append("train in bf16")
    where = f"at step {step}" if step else "on the first step"
    return f"Out of VRAM {where}. Try: " + ("; ".join(tips) or "a smaller rank or fewer target layers") + "."


def state_dir(run_id: str) -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "zentrainer" / "states" / run_id


def lora_root(lora_dir: str | None) -> tuple[Path, Path]:
    """(the ComfyUI loras folder LoRA names are relative to, the folder runs save under).
    A chosen folder must be one ComfyUI lists for loras: names stay loadable, and a spec
    can't point writes anywhere else."""
    import folder_paths

    default = Path(folder_paths.get_output_directory()) / "zentrainer" / "loras"
    if not lora_dir:
        return default, default
    want = Path(lora_dir).resolve()
    for p in folder_paths.get_folder_paths("loras"):
        if Path(p).resolve() == want:
            return want, want / "zentrainer"
    raise ValueError(f"output.lora_dir {lora_dir!r} isn't one of ComfyUI's loras folders")


def state_for(checkpoint: dict, run_id: str) -> str | None:
    """The training-state file of a checkpoint row, if it exists. Older checkpoints kept
    it beside the LoRA as <name>.state.pt."""
    for cand in (checkpoint.get("state"), str(Path(checkpoint["path"]).with_suffix(".state.pt"))):
        if cand and Path(cand).is_file():
            return cand
    return None


class Stop(Exception):
    def __init__(self, status: str):
        self.status = status


_compiler_default: bool | None = None


def _comfy_compiler(on: bool) -> None:
    """ComfyUI's memory compiler (comfy_aimdo malloc graphs) records one forward's allocations
    and replays them on the next. That suits repeated inference, not training: autograd keeps
    tensors past each block and bucket shapes change, and the replay fails with "aimdo memory
    compile error". ComfyUI only resets it between nodes, and a whole run is one node, so it's
    off for the whole run (samples included) and back to the launch setting at the end."""
    global _compiler_default
    import comfy.model_prefetch
    from comfy.cli_args import args

    if _compiler_default is None:
        _compiler_default = args.disable_comfy_compiler
    comfy.model_prefetch.cleanup_malloc_graph()
    args.disable_comfy_compiler = _compiler_default if on else True


# VRAM left for the LoRA, optimizer state, activations and ComfyUI itself when the base model
# loads whole; below this the run offloads weights instead (see Trainer._auto_offload)
AUTO_OFFLOAD_HEADROOM = 8 * 1024 ** 3
# video clips are many tokens per step (73 frames at 512 px: ~5k; a 1024 px image on H3: ~1k):
# activations grow with them, so the headroom does too, beyond this many tokens
HEADROOM_TOKENS = 2048
HEADROOM_PER_TOKEN = int(1.0 * 1024 ** 2)  # measured: 73 frames at 512 (~5.3k tokens) peaked ~7 GB of activations


class _Lazy:
    """Stands in for an object until it's actually used (the text encoder, for a cache hit)."""

    def __init__(self, get):
        self._get = get

    def __getattr__(self, name):
        return getattr(self._get(), name)


class Trainer:
    def __init__(self, spec: TrainSpec, run_id: str, *, store=None, emit: Callable = events.emit,
                 control: control_mod.Control | None = None, callbacks: list[Callback] | None = None,
                 resume_from: str | None = None, resume_state: str | None = None):
        self.spec = spec
        self.resume_from = resume_from  # a checkpoint .safetensors
        self.resume_state = resume_state  # its training state (see state_for)
        self.run_id = run_id
        self.store = store
        self.emit = emit
        self.control = control or control_mod.Control()
        self.callbacks = callbacks if callbacks is not None else [
            Metrics(), Checkpoints(spec.save_every), Samples(spec.samples.every), Validation()]
        self.step = 0
        self.train_seconds = 0.0  # time spent in optimizer steps only (no sampling/validation/saves/pauses)
        self.last_step_seconds: float | None = None
        self.last_loss: float | None = None
        self.last_grad_norm: float | None = None
        self._samples = next((c for c in self.callbacks if isinstance(c, Samples)), None)

    # ---- lifecycle -------------------------------------------------------------

    def run(self) -> str:
        """Train to completion (or stop). Returns the final status."""
        import comfy.model_management as mm

        status = "error"
        with torch.inference_mode(False):  # ComfyUI executes nodes under inference_mode
            try:
                self._status("loading")
                self.setup()
                self._status("training")
                with gradient_checkpointing(self.models.patcher.model.diffusion_model, self.spec.memory.gradient_checkpointing,
                                            self.spec.memory.checkpoint_depth, self.spec.memory.offloading,
                                            self.spec.memory.checkpoint_skip) as ckpt:
                    self._input_grad = ckpt.input_grad
                    self.enter_training()
                    try:
                        for cb in self.callbacks:
                            cb.on_start(self)
                        self._drop_text_encoder()
                        # on_start may have loaded the text encoder (sample prompts) and the VAE
                        # (baseline samples), pushing the model out: back in before step one
                        self.enter_training()
                        status = self.loop()
                    except Stop as s:
                        status = s.status
                    except torch.OutOfMemoryError as e:
                        status = "error"
                        raise RuntimeError(_oom_advice(self.spec, self.step)) from e
                    except BaseException as e:
                        status = "interrupted" if type(e).__name__ == "InterruptProcessingException" else "error"
                        raise
                    finally:
                        mm.in_training = False
                        for cb in self.callbacks:
                            try:
                                cb.on_end(self, status)
                            except Exception:
                                logging.exception("[ZenTrainer] callback on_end failed")
                        self.network.eject(self.models.patcher)
                        self.models.patcher.cleanup()
                        _comfy_compiler(True)  # after on_end: its final samples run without it too
            except BaseException as e:
                interrupted = type(e).__name__ == "InterruptProcessingException"
                self._status("interrupted" if interrupted else "error", message=str(e) or type(e).__name__)
                raise
            finally:
                mm.training_fp8_bwd = False
                mm.soft_empty_cache()
        self._status(status, final_step=self.step)
        return status

    def setup(self) -> None:
        import comfy.model_management as mm
        import comfy.utils
        import folder_paths

        s = self.spec
        compat.apply()  # ComfyUI model fixes for training (see compat.py)
        from . import int8_backward

        int8_backward.enable("w8a8" if s.memory.w8a8_backward else "fused" if s.memory.fused_int8_backward else "")
        lora_root(s.output.lora_dir)  # a bad save folder fails now, not at the first save
        torch.manual_seed(s.seed)
        random.seed(s.seed)
        self.generator = torch.Generator().manual_seed(s.seed)

        self.models = models.load(s.model)
        patcher = self.models.patcher
        compat.prune_unused(patcher)
        self._auto_offload(patcher)
        self.device = patcher.load_device
        self.train_dtype = _DTYPES[s.memory.training_dtype]

        encoder_id = repr((s.model.kind, s.model.ckpt_name, s.model.clip_name, s.model.clip_type, s.model.vae_name))
        root = Path(folder_paths.get_input_directory()) / s.dataset.folder
        cache = Path(folder_paths.get_user_directory()) / "zentrainer" / "cache"
        self.encoder_id, self.cache_dir = encoder_id, cache
        if s.dataset.video_audio and self.models.audio_vae is None:
            raise ValueError("training on video sound needs the model's audio VAE: set Model -> Audio VAE")
        from .edit import is_h3

        self.h3 = is_h3(patcher.model)
        if s.dataset.keyframes and not self.h3:
            raise ValueError("keyframes (image-to-video training) are for MiniMax H3's fl2va model")
        self.dataset = data.build(s.dataset, _Lazy(lambda: self.models.clip), self.models.vae, root, cache, encoder_id,
                                  progress=lambda i, n, msg: self._status("loading", message=f"{msg} ({i}/{n})"),
                                  audio_vae=self.models.audio_vae, h3=self.h3)
        self.edit_method = None
        if s.dataset.edit:
            from .edit import method, takes_references

            if not takes_references(patcher.model):
                raise ValueError(f"{type(patcher.model).__name__} doesn't take reference images, so it can't train "
                                 "edits: pick an edit-capable model (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit)")
            pairs = sum(sm.refs is not None or sm.cond is not None for sm in self.dataset.samples)
            if not pairs:
                raise ValueError("edit training, but no image has a <name>_control picture beside it")
            self.edit_method = None if self.h3 else method(patcher.model, s.dataset.edit_method)
            logging.info("[ZenTrainer] edit training: %d of %d images have an input picture (references: %s)",
                         pairs, len(self.dataset.samples), self.edit_method or "the model's own")
        self.batches = data.Batches(self.dataset, s.optim.batch_size, s.seed)

        if s.memory.training_dtype != "fp32":
            patcher.set_model_compute_dtype(self.train_dtype)
        mm.training_fp8_bwd = s.memory.quantized_backward

        init = None
        if self.resume_from:
            init = comfy.utils.load_torch_file(self.resume_from)
        elif s.network.init_from:
            init = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", s.network.init_from))
        # A weight wrapper on a quantized layer turns off its quantized matmul (every forward
        # would dequantize the full weight), so quantized bases train in bypass mode.
        quantized = any(getattr(m, "layout_type", None) is not None for m in patcher.model.modules())
        bypass = s.memory.bypass_mode or quantized
        if quantized and not s.memory.bypass_mode:
            self._status("loading", message="quantized base model: training in bypass mode")
        assist = None
        if s.network.training_adapter:
            left_out: list = []
            assist = network_mod.load_training_adapter(
                patcher, folder_paths.get_full_path_or_raise("loras", s.network.training_adapter), self.device,
                torch.bfloat16, left_out)
            self._partial(s.network.training_adapter, "the training adapter", len(assist), left_out)
            logging.info("[ZenTrainer] training adapter %s on %d layers (off for samples and the saved LoRA)",
                         s.network.training_adapter, len(assist))
        preview = None
        if s.samples.turbo_lora:
            left_out = []
            preview = network_mod.load_training_adapter(
                patcher, folder_paths.get_full_path_or_raise("loras", s.samples.turbo_lora), self.device, torch.bfloat16,
                left_out)
            self._partial(s.samples.turbo_lora, "the turbo LoRA", len(preview), left_out)
            logging.info("[ZenTrainer] samples ride on %s at %.2f", s.samples.turbo_lora, s.samples.turbo_strength)
        self.network = Network(patcher, s.network, self.device, bypass=bypass, init_weights=init, assist=assist,
                               preview=preview, preview_strength=s.samples.turbo_strength)
        self.optimizer, self.scheduler = optim.build(
            self.network.param_groups(s.optim.learning_rate, s.optim.lora_plus), s.optim)
        from .ema import EMA

        self.ema = EMA(self.network.params, s.optim.ema) if s.optim.ema > 0 else None
        self.objective = Objective(s.objective)
        self.denoiser = Denoiser(patcher, attention_override(s.memory.attention))
        self.streams = Streams(patcher.model)
        if self.streams.multi and s.optim.batch_size > 1:
            raise ValueError(f"{type(patcher.model).__name__} trains one image per step: set batch size 1 "
                             f"and use gradient accumulation {s.optim.batch_size} instead")
        self._shift = s.objective.shift
        if self._shift and not is_flow(self.denoiser.model_sampling):
            raise ValueError(f"objective.shift is for flow-matching models; {type(self.denoiser.model_sampling).__name__} isn't one. Set it to 0.")
        self.scaler = torch.amp.GradScaler() if self.train_dtype == torch.float16 else None

        self._conds: dict[tuple[int, tuple], dict] = {}  # (caption, shape) -> prepared cond
        self.progress = comfy.utils.ProgressBar(s.optim.steps)
        if self.resume_from:
            self._restore(Path(self.resume_state or state_for({"path": self.resume_from}, self.run_id) or ""))

    def _restore(self, state_path: Path) -> None:
        """Continue from a checkpoint: adapters were loaded from its LoRA file above;
        this brings back optimizer moments, the LR schedule position, step and RNG."""
        state = torch.load(str(state_path), map_location="cpu", weights_only=False)  # our own file
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state["step"])
        if state.get("live"):  # the checkpoint's LoRA file holds the average; training goes on from these
            with torch.no_grad():
                for k, v in state["live"].items():
                    if k in self.network.params:
                        self.network.params[k].copy_(v.to(self.network.params[k]))
        if self.ema is not None and state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        self.generator.set_state(state["rng"])
        self.batches.epoch = int(state.get("batches_epoch", 0))
        self.train_seconds = float(state.get("train_seconds", 0.0))
        self.batches.rng.seed(self.spec.seed + self.step)  # fresh shuffle order from here
        self.progress.update_absolute(self.step)
        self._status("loading", message=f"resumed from step {self.step}")

    def _drop_text_encoder(self) -> None:
        """Captions are encoded into the dataset cache and the sample prompts at on_start: past
        that the text encoder is never used, so don't hold it in RAM for the whole run (a few
        GB for most models, tens for the big VLM encoders)."""
        import comfy.model_management as mm

        if self.models._clip is None:  # never loaded (everything was cached)
            return
        self.models.release_clip()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
        logging.info("[ZenTrainer] text encoder released: captions and sample prompts are encoded")

    @property
    def headroom(self) -> int:
        """Bytes of VRAM kept free beside the (partly) loaded model: the spec's, or 8 GB plus
        room for the dataset's longest clip."""
        if self.spec.memory.vram_headroom > 0:
            return int(self.spec.memory.vram_headroom * 1024 ** 3)
        tokens = 0
        for smp in getattr(getattr(self, "dataset", None), "samples", [])[:2000]:
            lat = smp.latent
            if lat.ndim == 4:  # (C, T, h, w): a video clip; images keep the flat default
                tokens = max(tokens, lat.shape[1] * lat.shape[2] * lat.shape[3] // 4)
        return AUTO_OFFLOAD_HEADROOM + max(0, tokens - HEADROOM_TOKENS) * HEADROOM_PER_TOKEN

    def _auto_offload(self, patcher) -> None:
        """Offloading off means the whole base model sits in VRAM. When it can't (its weights
        alone leave too little room for the LoRA, optimizer and activations), turn offloading
        on rather than fail at step one: the run is slower, not broken."""
        m = self.spec.memory
        if m.offloading or not torch.cuda.is_available():
            return
        weights = patcher.model_size()
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        if weights > total - self.headroom:
            m.offloading = True
            gb = 1024 ** 3
            logging.info("[ZenTrainer] weight offloading on: the model is %.1f GB on a %.1f GB card", weights / gb, total / gb)
            self._status("loading", message=f"Offloading weights: the model ({weights / gb:.0f} GB) doesn't fit beside training in {total / gb:.0f} GB")

    def enter_training(self) -> None:
        """Load for training: everything else out of VRAM, model in train mode."""
        import comfy.model_management as mm

        patcher = self.models.patcher
        if self.spec.memory.offloading:
            # everything else out, then as much of the model as fits beside training stays in
            # VRAM; only the rest streams. (Asking for "all the memory" here, as before, left a
            # weight budget of nothing: every layer crossed PCIe twice a step.)
            ours = [lm for lm in mm.current_loaded_models if lm.model is patcher]
            mm.free_memory(1e30, self.device, keep_loaded=ours)
            logging.info("[ZenTrainer] keeping %.1f GB of VRAM free for training beside the model", self.headroom / 1024 ** 3)
            mm.load_models_gpu([patcher], memory_required=self.headroom, force_full_load=False)
        else:
            mm.load_models_gpu([patcher], memory_required=1e20, force_full_load=True)
        patcher.model.requires_grad_(False)
        patcher.model.train()
        patcher.pre_run()
        self.network.inject(patcher)
        if hasattr(self, "optimizer"):
            optim.set_mode(self.optimizer, training=True)
        mm.in_training = True
        _comfy_compiler(False)

    def enter_inference(self) -> None:
        import comfy.model_management as mm

        mm.in_training = False
        # the compiler stays off for samples too: models that drive it from their own forward
        # (MiniMax H3's malloc graphs) abort the process on a partly offloaded, LoRA-wrapped
        # model. It's back to the launch setting when the run ends.
        optim.set_mode(self.optimizer, training=False)
        self.models.patcher.model.eval()

    def exit_inference(self) -> None:
        # sampling may have loaded the VAE/text encoder and pushed the model out
        self.enter_training()

    # ---- the loop --------------------------------------------------------------

    def loop(self) -> str:
        total = self.spec.optim.steps
        while self.step < total:
            self._handle_commands()
            t0 = time.perf_counter()
            if getattr(self, "_profile_next", False):
                self._profile_next = False
                self._profiled_step()
            else:
                self._train_step()
            self.last_step_seconds = time.perf_counter() - t0
            self.train_seconds += self.last_step_seconds
            self.step += 1
            self.progress.update_absolute(self.step)  # also raises on ComfyUI interrupt
            for cb in self.callbacks:
                cb.on_step(self)
        return "done"

    def _profiled_step(self) -> None:
        """One training step under torch.profiler ("profile" command): a table of where the GPU
        time went (by op, and by op + input shapes) and a Chrome trace, in the run's folder."""
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            self._train_step()
            torch.cuda.synchronize()
        out = self.store.dir(self.run_id) if self.store else Path(".")
        ka = prof.key_averages()
        table = ka.table(sort_by="cuda_time_total", row_limit=60)
        shapes = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=40)
        (out / f"profile-step{self.step:06d}.txt").write_text(f"{table}\n\nby input shape\n{shapes}\n")
        prof.export_chrome_trace(str(out / f"profile-step{self.step:06d}.json"))
        logging.info("[ZenTrainer] profiled step %d: %s", self.step, out / f"profile-step{self.step:06d}.txt")

    def _train_step(self) -> None:
        s = self.spec
        ms = self.denoiser.model_sampling
        acc = s.optim.grad_accumulation_steps
        total_loss = 0.0
        # where the step's time goes (synchronised, so each phase owns its GPU work): data,
        # forward, backward, optimizer. Shown on the run page; a sync per phase costs ~nothing
        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        ph = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        mark = time.perf_counter()

        def lap(name):
            nonlocal mark
            sync()
            now = time.perf_counter()
            ph[name] += now - mark
            mark = now

        for _ in range(acc):
            idx = next(self.batches)
            x0, weight, shapes = self.batch(idx)
            conds = [self._cond(i, x0, shapes) for i in idx]

            noise = torch.randn(x0.shape, generator=self.generator).to(self.device)
            sigma = sigmas_for(ms, sample_percents(s.objective, len(idx), self.generator), self._shift).to(self.device)
            x_t = self.objective.noisy(ms, sigma, noise, x0)
            lap("data")

            # the input needs grad only for ComfyUI's per-layer offload checkpoint (see memory.py);
            # otherwise it stays grad-free so the backward ends at the first trained layer
            with torch.autocast(self.device.type, dtype=self.train_dtype, enabled=self.train_dtype != torch.float32):
                x0_pred = self.denoiser(x_t.requires_grad_(getattr(self, "_input_grad", True)), sigma, conds,
                                        self._tread(idx))
            loss = self.objective.loss(x_t.detach(), x0, x0_pred, sigma, weight).mean()
            lap("forward")
            scaled = loss / acc
            (self.scaler.scale(scaled) if self.scaler else scaled).backward()
            total_loss += loss.item()
            lap("backward")

        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        params = self.network.trainable()
        for p in params:  # adapters may run in a lower precision than their grads
            if p.grad is not None and p.grad.dtype != p.dtype:
                p.grad = p.grad.to(p.dtype)
        clip = s.optim.max_grad_norm
        norm = torch.nn.utils.clip_grad_norm_(params, clip if clip > 0 else float("inf"))
        self.last_grad_norm = float(norm)
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.step)
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        lap("optimizer")
        self.last_phases = {k: round(v, 4) for k, v in ph.items()}
        self.last_loss = total_loss / acc

    def _tread(self, idx: list[int]) -> dict | None:
        """TREAD's block replacements for this step: only on a video clip, only for models whose
        blocks take the replace hook with a sequence layout (MiniMax H3)."""
        share = self.spec.memory.tread
        if share <= 0 or any(self.dataset.samples[i].latent.ndim != 4 or self.dataset.samples[i].latent.shape[1] < 2 for i in idx):
            return None
        dm = self.models.patcher.model.diffusion_model
        if "minimax" not in type(dm).__module__.lower() or not hasattr(dm, "blocks"):
            return None
        from .tread import patches

        p = patches(len(dm.blocks), share, self.generator)
        return {"patches_replace": {"dit": p}} if p else None

    def batch(self, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor | None, list | None]:
        """The model-ready x0 for these samples, its loss weights (masks; for a multi-stream
        model also which part is the picture) and, for a multi-stream model, the stream shapes
        its conds need. See streams.py."""
        model = self.models.patcher.model
        latent = torch.stack([self.dataset.samples[i].latent for i in idx]).to(self.device, torch.float32)
        weight = self._mask_weight(idx, latent) if self.spec.dataset.masked_loss else None
        audio = [self.dataset.samples[i].audio for i in idx]
        packed = self.streams.pack(latent, weight, audio if any(a is not None for a in audio) else None)
        if packed is None:
            return model.process_latent_in(latent), weight, None
        model.latent_shapes = packed.shapes  # how the sampler leaves it: process_latent_in reads it
        return model.process_latent_in(packed.x0), packed.weight, packed.shapes

    def _mask_weight(self, idx: list[int], x0: torch.Tensor) -> torch.Tensor | None:
        """(B, h, w) loss weights: mask where present, 1 elsewhere; the unmasked area
        keeps `unmasked_weight` (0 = ignore it entirely)."""
        masks = [self.dataset.samples[i].mask for i in idx]
        if all(m is None for m in masks):
            return None
        floor = self.spec.dataset.unmasked_weight
        hw = x0.shape[-2:]
        w = torch.stack([m if m is not None else torch.ones(hw) for m in masks]).to(x0.device)
        return floor + (1.0 - floor) * w

    def fixed_cond(self, i: int, x0: torch.Tensor, shapes: list | None = None) -> dict:
        """A sample's own caption, prepared; never dropped out (validation uses this)."""
        sample = self.dataset.samples[i]
        return self._prepared(i, sample.caption, x0, shapes)

    def _prepared(self, i: int, cap: int, x0: torch.Tensor, shapes: list | None) -> dict:
        """A caption's processed conds for sample i's shape; an edit pair's carry its own input
        picture, so those are kept per sample."""
        sample = self.dataset.samples[i]
        key = (cap, tuple(sample.latent.shape), i if sample.refs or sample.cond is not None else None)
        if key not in self._conds:
            cond = self.dataset.conditionings[cap]
            if sample.cond is not None:  # grounded: the encoder saw this image's inputs
                empty = cap == self.dataset.empty_caption and sample.cond_empty is not None
                cond = sample.cond_empty if empty else sample.cond
            if sample.refs:
                from .edit import with_refs

                cond = with_refs(cond, sample.refs, self.edit_method)
            with torch.no_grad():
                self._conds[key] = self.denoiser.prepare(cond, x0[:1], self.device, shapes)
        return self._conds[key]

    def _cond(self, i: int, x0: torch.Tensor, shapes: list | None = None) -> dict:
        sample = self.dataset.samples[i]
        cap = random.choice(sample.variants) if len(sample.variants) > 1 else sample.caption
        if self.spec.dataset.caption_dropout and random.random() < self.spec.dataset.caption_dropout:
            cap = self.dataset.empty_caption  # an edit pair keeps its input picture
        return self._prepared(i, cap, x0, shapes)

    def _handle_commands(self) -> None:
        import comfy.model_management as mm

        self._apply(self.control.drain())
        if self.control.paused:
            self._status("paused")
            while not self.control.wait_resumed(0.5):
                mm.throw_exception_if_processing_interrupted()
            self._apply(self.control.drain())  # whatever arrived while paused (set_lr, stop, ...)
            self._status("training")

    def _partial(self, name: str, what: str, kept: int, left_out: list) -> None:
        """Say so when a frozen adapter fits only part of this model (it was made for another build
        of it, e.g. the full H3 against a pruned one): what it does then is only partly there."""
        if not left_out:
            return
        share = len(left_out) / (kept + len(left_out))
        msg = (f"{name}: {len(left_out)} of {kept + len(left_out)} layers don't fit this model (made for another "
               f"build of it?), so {what} is only partly applied")
        logging.warning("[ZenTrainer] %s", msg)
        if share >= 0.05:
            self._status("loading", message=msg)

    def _apply(self, commands) -> None:
        tries: list[dict] = []
        for cmd in commands:
            if cmd.name == "stop":
                raise Stop("stopped")
            if cmd.name == "save":
                self.save(f"manual-step{self.step:06d}")
            elif cmd.name == "sample" and self._samples:
                if cmd.args.get("prompt"):
                    tries.append(cmd.args)  # "Try a prompt": batched below
                else:
                    self._samples.render(self)
            elif cmd.name == "profile":
                self._profile_next = True
            elif cmd.name == "set_lr":
                optim.set_lr(self.optimizer, self.scheduler, float(cmd.args["lr"]))
        if tries and self._samples:
            # manual tries in one session; ones queued while it renders join it, so the model and
            # encoder aren't swapped in and out per try. Other commands that arrive meanwhile
            # wait for the next step boundary.
            later = []

            def more():
                got = []
                for c in self.control.drain():
                    (got.append(c.args) if c.name == "sample" and c.args.get("prompt") else later.append(c))
                return got

            self._samples.render_manual(self, tries, more)
            if later:
                self._apply(later)

    # ---- outputs ---------------------------------------------------------------

    def save(self, tag: str, final: bool = False) -> str:
        import folder_paths
        import safetensors.torch

        optim.set_mode(self.optimizer, training=False)  # schedule-free: save the eval weights
        try:
            if self.ema is None:
                return self._save(tag, final, folder_paths, safetensors.torch)
            with self.ema.applied() as live:  # the LoRA file is the average; the state keeps the live weights
                return self._save(tag, final, folder_paths, safetensors.torch, live)
        finally:
            optim.set_mode(self.optimizer, training=True)

    def _save(self, tag, final, folder_paths, sft, live: dict | None = None) -> str:
        root, base = lora_root(self.spec.output.lora_dir)
        name = f"{slugify(self.spec.name)}_{self.run_id}" + ("" if final else f"_{tag}")
        folder = base / slugify(self.spec.name) if final else base / "checkpoints" / self.run_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.safetensors"
        sft.save_file(self.network.state_dict(), str(path),
                                    metadata=write_metadata(self.spec.to_dict(), self.step))
        # resumable state (optimizer, schedule, step, RNG) goes outside the LoRA folders:
        # those are registered with ComfyUI, which would list a .pt there as a "LoRA"
        keep = self.spec.output.keep_states
        state = state_dir(self.run_id) / f"{name}.pt"
        if keep != "none":
            state.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": self.step, "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "rng": self.generator.get_state(), "batches_epoch": self.batches.epoch,
                "train_seconds": self.train_seconds,
                **({"live": {k: v.cpu() for k, v in live.items()}, "ema": self.ema.state_dict()} if live is not None else {}),
            }, str(state))
            if keep == "latest":  # resume only ever needs the newest
                for old in state.parent.glob("*.pt"):
                    if old != state:
                        old.unlink(missing_ok=True)
        row = {"run_id": self.run_id, "step": self.step, "path": str(path), "final": final,
               "lora_name": os.path.relpath(path, root), "state": str(state) if keep != "none" else None}
        self.emit(events.CHECKPOINT, row)
        if self.store:
            self.store.append(self.run_id, "checkpoints", row)
        return str(path)

    def _status(self, status: str, message: str | None = None, **fields: Any) -> None:
        self.emit(events.STATUS, {"run_id": self.run_id, "status": status, "message": message, "step": self.step})
        self.emit(events.ZENKIT_JOB, {
            "id": f"zentrainer:{self.run_id}", "name": f"Train {self.spec.name}",
            "status": {"loading": "start", "training": "progress", "paused": "progress", "done": "done"}.get(
                status, "error"),
            "current": self.step, "total": self.spec.optim.steps, "message": message or status,
            "source": "ZenTrainer",
        })
        if self.store:
            self.store.update(self.run_id, status=status, message=message, **fields)
