"""Run specs — the declarative contract between the UI and the harness.

A spec says *what* to train; the harness decides *how*. Pure Python, no ComfyUI
imports, so it can be validated anywhere.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

ModelKind = Literal["checkpoint", "split"]


@dataclass
class ModelSpec:
    # "checkpoint": one file with MODEL+CLIP+VAE (SD1.5/SDXL style).
    # "split": separate diffusion model / text encoder / VAE (Flux, Qwen-Image, Z-Image, ...).
    kind: ModelKind = "checkpoint"
    ckpt_name: str | None = None
    unet_name: str | None = None
    unet_weight_dtype: str = "default"
    clip_name: str | None = None
    clip_type: str | None = None
    vae_name: str | None = None
    # Joint audio-video models (MiniMax H3): their audio VAE, needed to train on a video's sound
    # (dataset.video_audio) and to put sound in video samples.
    audio_vae_name: str | None = None


@dataclass
class DatasetSpec:
    # A subfolder of ComfyUI's input dir: images + same-name caption files. A
    # kohya-style "N_name" subfolder repeats its images N times.
    folder: str = ""
    resolution: int = 1024
    caption_ext: str = ".txt"
    # Aspect-ratio buckets: images are resized to the closest bucket of ~resolution²
    # pixels, sides multiples of bucket_step. Off = center-crop squares.
    buckets: bool = True
    bucket_step: int = 64
    # Edit training: each image's <name>_control picture is the input the model edits from
    # (passed as reference latents), the caption is the instruction. Models that take reference
    # images (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit...).
    edit: bool = False
    # how the input picture joins the sequence: "" = the model's own way ("index" for models
    # without one, like Krea 2), or "index", "offset", "index_timestep_zero"
    edit_method: str = ""
    # wrap each instruction: "{}" is the caption (NK2E's generalist was trained on "Make the
    # following edit to the reference image, ...: {}", so specialists on top of it want the same)
    edit_template: str = ""
    # the text encoder sees the input pictures too (Qwen3-VL based: Krea 2, Qwen Image), as
    # krea2edit's identity edits do: conditioning becomes per image, encoded when the dataset caches
    edit_grounded: bool = False
    # MiniMax H3 image-to-video: each clip conditioned on its own first frame ("first") or first
    # and last ("first_last"), as the fl2va model is at inference; the text encoder sees them too
    keyframes: str = ""
    caption_dropout: float = 0.0
    # Caption sets: <img>.txt is "default", <img>.<set>.txt the others (tags, nl, bbox, ...).
    # Train on one set, or mix several: each step picks one of the image's sets at random.
    caption_set: str = "default"
    caption_mix: list[str] = field(default_factory=list)
    # Trigger word (as in ai-toolkit): "[trigger]" in captions and sample prompts becomes
    # this, and captions that don't mention it get it prepended ("ohwx, a photo of ...").
    trigger: str = ""
    trigger_prepend: bool = True
    # Masked training: images with a mask only learn where it's white. The rest of the
    # image keeps `unmasked_weight` (0 = ignored, 0.1 = learned lightly).
    masked_loss: bool = False
    unmasked_weight: float = 0.0
    # Videos (.mp4 .webm .mov .mkv ...) train as clips: `video_frames` long at most (snapped to a
    # length the model's VAE encodes exactly), resampled to `video_fps` (0 = the video's own),
    # picked by `video_mode` (uniform: spread over the video; head: the start; chunks:
    # consecutive) up to `video_clips` per video, each at about `video_resolution`² pixels (a
    # clip costs frames x pixels, so videos usually train smaller than images). An image model
    # trains a video's frames as images.
    video_frames: int = 49
    video_fps: float = 0.0
    video_mode: str = "uniform"
    video_clips: int = 1
    video_resolution: int = 512
    # Train on each clip's soundtrack too (joint audio-video models with model.audio_vae_name):
    # the audio stream joins the loss instead of being left out, so cuts can learn the beat.
    video_audio: bool = False
    # Keep clips inside one shot: hard cuts are detected and no clip spans one. Off for edits
    # where the cutting is the point (music videos, PMVs).
    video_avoid_cuts: bool = False


@dataclass
class NetworkSpec:
    algorithm: str = "LoRA"  # LoRA | LoHa | LoKr | OFT (comfy.weight_adapter)
    rank: int = 16
    alpha: float | None = None  # None = rank (scale 1.0)
    # fp32: AdamW's steps on a LoRA are smaller than bf16 can represent at the weights' size,
    # so in bf16 most of them round away (a toy LoRA learned 6% vs 51% in fp32, same steps).
    # bf16 only with an optimizer that rounds stochastically (Automagic).
    lora_dtype: str = "fp32"
    # Which layers get adapters: "attn" (attention projections), "attn_mlp"
    # (attention + feed-forward linears), "all" (every adaptable layer, like
    # ComfyUI's native trainer), or a list of regexes over module names.
    targets: str | list[str] = "attn_mlp"
    train_bias_norm: bool = False
    init_from: str | None = None  # a LoRA in the loras folder to continue
    # A LoRA (loras folder) applied while training only, never in samples or the saved file:
    # e.g. a de-distillation adapter for a guidance-distilled model (MiniMax H3), so training
    # sits on an undistilled base and the result runs on the real one.
    training_adapter: str | None = None
    # Only these transformer blocks get adapters, e.g. "20-49" (MiniMax H3's identity blocks,
    # Fizgig's default) or "10-": the backward then stops at the first one, so steps are
    # quicker, and the model's early (layout, motion) blocks stay untouched. "" = every block.
    blocks: str = ""


@dataclass
class OptimSpec:
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "constant"  # constant | linear | cosine
    warmup_steps: int = 0
    max_grad_norm: float = 1.0  # 0 = off
    steps: int = 1000
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    # LoRA+: the up (B) matrices learn this many times faster than the down (A) ones, which
    # trains a LoRA faster for the same stability (paper: 16; 4 is a safe start). 0 or 1 = off.
    lora_plus: float = 0.0
    # Weight averaging: checkpoints, samples and the saved LoRA use a running average of the
    # adapter (0.98 = roughly the last 50 steps). 0 = off.
    ema: float = 0.0


@dataclass
class ObjectiveSpec:
    loss: str = "mse"  # mse | l1 | huber | smooth_l1
    # "native": each model's own prediction space (eps for SD-style, velocity for
    # flow models). "x0": loss on the denoised image, ComfyUI's native trainer default.
    target: str = "native"
    timesteps: str = "uniform"  # uniform | logit_normal
    logit_mean: float = 0.0
    logit_std: float = 1.0
    # share of steps drawn uniformly instead (logit-normal only): keeps a little training at
    # the extremes it rarely visits (malcolmrey: 0.01)
    uniform_mix: float = 0.0
    # Training timestep shift for flow-matching models (0 = the model's own sampling shift).
    # The sampling shift favours heavy noise (MOSS's 5 puts ~70% of steps above σ≈0.7):
    # composition and colour. A lower one (1-3) spends more steps on detail, e.g. faces.
    shift: float = 0.0


@dataclass
class MemorySpec:
    training_dtype: str = "bf16"  # bf16 | fp16 | fp32
    gradient_checkpointing: bool = True
    checkpoint_depth: int = 1
    # leave the last N blocks un-checkpointed: they keep their activations (more VRAM) and
    # skip the recompute. Their backward runs right after their forward, so it's the
    # cheapest place to trade memory for speed.
    checkpoint_skip: int = 0
    offloading: bool = False
    # TREAD token routing on video steps (MiniMax H3): this share of a clip's tokens skips the
    # middle blocks (2 to n-3) and rejoins unchanged. 0.5 roughly halves those blocks' work;
    # stills, samples and validation always run whole. 0 = off.
    tread: float = 0.0
    # int8 ConvRot bases (MiniMax H3, Krea 2 int8): the backward computes each frozen layer's
    # input gradient with a fused Triton kernel instead of rebuilding its bf16 weight (Fizgig's,
    # by @mabseyuk). Saves a transient weight per layer; faster at large token counts.
    fused_int8_backward: bool = False
    # int8 ConvRot bases: the backward's gradient quantized to int8 as well, so the frozen layers'
    # input gradients run on int8 tensor cores (Triton kernels: 2x the bf16 backward on H3's
    # clip-size layers, RTX 5090). Some gradient precision for the speed (OneTrainer's W8A8).
    w8a8_backward: bool = False
    # With offloading: VRAM kept free for activations, the LoRA and its optimizer (GB);
    # the rest holds as much of the model as fits. 0 = auto: 8 GB, more for long video clips.
    vram_headroom: float = 0.0
    quantized_backward: bool = False
    bypass_mode: bool = False
    # Attention backend while training. Flash and sage have no backward pass, so the
    # default pins differentiable SDPA regardless of ComfyUI's launch flags.
    attention: str = "pytorch"


@dataclass
class SamplePrompt:
    """One sample: its prompt and, optionally, its own size/length/seed (else the run's)."""
    prompt: str
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    seed: int | None = None
    # edit runs: the picture to edit (a path in the dataset folder, or under ComfyUI's input dir)
    control: str | None = None


@dataclass
class SampleSpec:
    every: int = 0  # 0 = off
    # each a prompt, or {"prompt", "width"?, "height"?, "frames"?, "seed"?} for its own size
    prompts: list = field(default_factory=list)
    negative: str = ""
    width: int = 1024
    height: int = 1024
    # Video models: frames per sample (1 = a still). Snapped to a length the VAE decodes,
    # saved as an .mp4 at `fps`.
    frames: int = 1
    fps: float = 24.0
    steps: int = 20
    cfg: float = 5.0
    sampler: str = "euler"
    scheduler: str = "normal"
    seed: int = 42
    # Render the prompts once before training: the base model's look, to compare against.
    baseline: bool = True
    # Sample with a different model than the one being trained (e.g. train Krea raw,
    # preview on Krea turbo): a diffusion_models file for split models, a checkpoints
    # file for checkpoint models. Same family, so the training text encoder/VAE are used.
    model_name: str | None = None
    model_weight_dtype: str = "default"
    lora_strength: float = 1.0
    # A turbo LoRA under every sample (not training, not the saved LoRA): few-step previews,
    # e.g. MiniMax H3's minimax_h3_turbo_v4_step600_ema at 0.75 with 6 steps, CFG 1.
    turbo_lora: str | None = None
    turbo_strength: float = 0.75
    # Skip-layer guidance, as ComfyUI's SkipLayerGuidanceDiT nodes do it. "uncond" skips the
    # layers in the negative pass only (no extra cost; MOSS Image's recommended setting);
    # "standard" adds a pass without them and pushes away from it by slg_scale. Layers are
    # the main (single-stream) block indices; slg_double_layers is for double-stream models.
    slg_mode: str = "off"  # off | uncond | standard
    slg_layers: list[int] = field(default_factory=list)
    slg_double_layers: list[int] = field(default_factory=list)
    slg_scale: float = 3.0
    slg_start: float = 0.01
    slg_end: float = 1.0


@dataclass
class ValidationSpec:
    """A smooth "is it learning" curve: loss on fixed images at fixed noise levels with
    fixed noise, so the only thing that changes between evaluations is the LoRA."""
    every: int = 100  # 0 = off
    images: int = 4
    timesteps: int = 4
    seed: int = 1234


@dataclass
class OutputSpec:
    # One of ComfyUI's "loras" folders (e.g. your model library's, on a roomier disk);
    # None = output/zentrainer/loras inside this install. LoRAs go in a zentrainer/
    # subfolder of it, so ComfyUI lists them as zentrainer/<run>/<file>.
    lora_dir: str | None = None
    # Resume states (optimizer, schedule, RNG) beside each checkpoint, often bigger than
    # the LoRA: "latest" keeps only the newest (resume from the last save), "all" one per
    # checkpoint, "none" skips them (the run can't be resumed).
    keep_states: str = "latest"


@dataclass
class TrainSpec:
    name: str = "untitled"
    seed: int = 0
    save_every: int = 250
    model: ModelSpec = field(default_factory=ModelSpec)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    network: NetworkSpec = field(default_factory=NetworkSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    samples: SampleSpec = field(default_factory=SampleSpec)
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainSpec:
        def build(dc, raw):
            raw = raw or {}
            known = {f.name for f in fields(dc)}
            unknown = set(raw) - known
            if unknown:
                raise SpecError(f"unknown {dc.__name__} field(s): {', '.join(sorted(unknown))}")
            return dc(**raw)

        raw = dict(data)
        nested = {f.name: f.default_factory for f in fields(cls) if callable(f.default_factory)}
        parts = {k: build(dc, raw.pop(k, None)) for k, dc in nested.items()}
        spec = build(cls, raw)
        for k, v in parts.items():
            setattr(spec, k, v)
        spec.validate()
        return spec

    def validate(self) -> None:
        m = self.model
        if m.kind not in ("checkpoint", "split"):
            raise SpecError(f"model.kind must be 'checkpoint' or 'split', got {m.kind!r}")
        if m.kind == "checkpoint" and not m.ckpt_name:
            raise SpecError("model.ckpt_name is required for a checkpoint model")
        if m.kind == "split":
            missing = [k for k in ("unet_name", "clip_name", "clip_type", "vae_name") if not getattr(m, k)]
            if missing:
                raise SpecError(f"split model needs: {', '.join('model.' + k for k in missing)}")
        if not self.dataset.folder:
            raise SpecError("dataset.folder is required")
        _choice("network.algorithm", self.network.algorithm, ("LoRA", "LoHa", "LoKr", "OFT"))
        _choice("optim.scheduler", self.optim.scheduler, ("constant", "linear", "cosine"))
        _choice("objective.loss", self.objective.loss, ("mse", "l1", "huber", "smooth_l1"))
        _choice("objective.target", self.objective.target, ("native", "x0"))
        _choice("objective.timesteps", self.objective.timesteps, ("uniform", "logit_normal"))
        _choice("memory.training_dtype", self.memory.training_dtype, ("bf16", "fp16", "fp32"))
        _choice("samples.slg_mode", self.samples.slg_mode, ("off", "uncond", "standard"))
        _choice("output.keep_states", self.output.keep_states, ("latest", "all", "none"))
        if self.objective.shift < 0:
            raise SpecError("objective.shift must be >= 0 (0 = the model's own)")
        if not 0.0 <= self.samples.slg_start <= self.samples.slg_end <= 1.0:
            raise SpecError("samples.slg_start/slg_end must satisfy 0 <= start <= end <= 1")
        if isinstance(self.network.targets, str):
            _choice("network.targets", self.network.targets, ("attn", "attn_mlp", "all"))
        if self.optim.steps < 1:
            raise SpecError("optim.steps must be >= 1")
        for p in self.samples.prompts:
            if isinstance(p, dict):
                if not isinstance(p.get("prompt"), str):
                    raise SpecError("each sample needs a prompt")
                for k in ("width", "height", "frames", "seed"):
                    if p.get(k) is not None and (not isinstance(p[k], int) or p[k] < (0 if k == "seed" else 1)):
                        raise SpecError(f"sample {k} must be a positive whole number")
                if any(p.get(k) and p[k] % 16 for k in ("width", "height")):
                    raise SpecError("sample width/height must be multiples of 16")
            elif not isinstance(p, str):
                raise SpecError("samples.prompts holds prompts (text) or {prompt, width, height, frames, seed}")
        if self.network.blocks and not re.fullmatch(r"\s*\d*\s*-?\s*\d*\s*", self.network.blocks):
            raise SpecError("network.blocks is a range like 20-49, 20- or 10")
        if not 0.0 <= self.memory.tread < 1.0:
            raise SpecError("memory.tread is a share of the video tokens, from 0 (off) to below 1")
        if not 0.0 <= self.objective.uniform_mix <= 1.0:
            raise SpecError("objective.uniform_mix is a share between 0 and 1")
        if self.dataset.edit_method not in ("", "index", "offset", "index_timestep_zero", "negative_index", "uxo"):
            raise SpecError("dataset.edit_method is one of index, offset, index_timestep_zero (or empty for the model's own)")
        if self.dataset.keyframes not in ("", "first", "first_last"):
            raise SpecError("dataset.keyframes is first, first_last or empty")
        if self.dataset.edit_template and "{}" not in self.dataset.edit_template:
            raise SpecError("dataset.edit_template needs {} where the instruction goes")
        if not 0.0 <= self.optim.ema < 1.0:
            raise SpecError("optim.ema is a decay between 0 (off) and below 1, e.g. 0.98")
        if self.optim.lora_plus < 0:
            raise SpecError("optim.lora_plus must be >= 0 (0 = off)")
        if self.samples.frames < 1 or self.samples.fps <= 0:
            raise SpecError("samples.frames must be >= 1 and samples.fps > 0")
        if self.save_every < 0 or self.samples.every < 0:
            raise SpecError("save_every and samples.every must be >= 0 (0 = off)")
        if self.validation.every < 0 or self.validation.images < 1 or self.validation.timesteps < 1:
            raise SpecError("validation needs every >= 0, images >= 1 and timesteps >= 1")
        if not 0.0 <= self.dataset.unmasked_weight <= 1.0:
            raise SpecError("dataset.unmasked_weight must be between 0 and 1")
        d = self.dataset
        if d.video_mode not in ("uniform", "head", "chunks"):
            raise SpecError("dataset.video_mode is uniform, head or chunks")
        if d.video_frames < 1 or d.video_clips < 1 or d.video_resolution < 64 or d.video_fps < 0:
            raise SpecError("dataset video settings: frames and clips at least 1, resolution at least 64, fps 0 or more")


class SpecError(ValueError):
    pass


def _choice(name: str, value: str, options: tuple[str, ...]) -> None:
    if value not in options:
        raise SpecError(f"{name} must be one of {', '.join(options)}; got {value!r}")


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or "run"


def sample_prompts(s: SampleSpec) -> list[SamplePrompt]:
    """The samples with every field filled from the run's defaults."""
    out = []
    for i, p in enumerate(s.prompts):
        d = p if isinstance(p, dict) else {"prompt": p}
        out.append(SamplePrompt(d["prompt"], d.get("width") or s.width, d.get("height") or s.height,
                                d.get("frames") or s.frames, s.seed + i if d.get("seed") is None else d["seed"],
                                d.get("control") or None))
    return out


# what a trained run's saved LoRA, optimizer state and dataset cache depend on: everything else
# (memory, samples, validation, LR and schedule, length, timesteps) may change at a resume
LOCKED_AT_RESUME = ("model", "dataset", "network", "optim.optimizer", "optim.batch_size", "optim.lora_plus", "seed")


def locked_changes(old: dict, new: dict) -> list[str]:
    def at(d: dict, path: str):
        for k in path.split("."):
            d = d.get(k) if isinstance(d, dict) else None
        return d

    out = []
    for path in LOCKED_AT_RESUME:
        a, b = at(old, path), at(new, path)
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{path}.{k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        elif a != b:
            out.append(path)
    return out
