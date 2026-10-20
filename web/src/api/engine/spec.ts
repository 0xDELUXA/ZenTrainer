// Run specs: the TrainSpec the New run page builds, its defaults, presets. Mirrors engine/zentrainer/spec.py; keep them in step.

export interface TrainSpec {
  name: string
  seed: number
  save_every: number
  model: {
    kind: 'checkpoint' | 'split'
    ckpt_name?: string | null
    unet_name?: string | null
    unet_weight_dtype?: string
    clip_name?: string | null
    clip_type?: string | null
    vae_name?: string | null
    /** joint audio-video models: the audio VAE (sound in training and samples) */
    audio_vae_name?: string | null
  }
  dataset: {
    folder: string
    resolution: number
    /** lower resolutions first: [{resolution: 768, until: 0.6}] trains at 768 for 60% of the run */
    stages?: { resolution: number; until: number }[]
    /** share of the run each resolution change blends over */
    stage_blend?: number
    caption_ext: string
    buckets: boolean
    bucket_step: number
    caption_dropout: number
    masked_loss: boolean
    edit?: boolean
    /** control sets (named layers of input pictures) the run edits from, in order; any turns edit on */
    control_sets?: string[]
    edit_method?: string
    edit_template?: string
    edit_grounded?: boolean
    keyframes?: string
    clip_stills?: boolean
    unmasked_weight: number
    /** videos: longest clip (snapped to the VAE), fps (0 = source), which windows, how many, size */
    video_frames: number
    video_fps: number
    video_mode: 'uniform' | 'head' | 'chunks'
    video_clips: number
    video_resolution: number
    /** keep clips inside one shot (hard cuts detected) */
    video_avoid_cuts?: boolean
    /** train on each clip's soundtrack (joint audio-video models) */
    video_audio?: boolean
    caption_set: string
    caption_mix: string[]
    trigger: string
    trigger_prepend: boolean
  }
  network: {
    algorithm: string
    rank: number
    alpha: number | null
    lora_dtype: string
    targets: 'attn' | 'attn_mlp' | 'all' | string[]
    train_bias_norm: boolean
    init_from: string | null
    /** a LoRA applied while training only (not in samples or the saved file): a de-distillation adapter */
    training_adapter: string | null
    /** a LoRA the trained one sits on everywhere (training, samples, use); saved beside, not in */
    base_lora?: string | null
    base_strength?: number
    /** only these blocks get adapters, e.g. "20-49" ("" = all): the backward stops at the first */
    blocks?: string
  }
  optim: {
    optimizer: string
    learning_rate: number
    weight_decay: number
    scheduler: 'constant' | 'linear' | 'cosine'
    warmup_steps: number
    max_grad_norm: number
    steps: number
    batch_size: number
    grad_accumulation_steps: number
    /** LoRA+: up matrices learn this many times faster (0 = off) */
    lora_plus?: number
    ema?: number
    /** Automagic3's lowest LR, as a fraction of the start LR; 0 = none */
    lr_floor?: number
  }
  objective: {
    loss: 'mse' | 'l1' | 'huber' | 'smooth_l1'
    target: 'native' | 'x0'
    timesteps: 'uniform' | 'logit_normal'
    logit_mean: number
    logit_std: number
    /** logit-normal: share of steps drawn uniformly instead */
    uniform_mix?: number
    /** flow models: training timestep shift (0 = the model's own sampling shift) */
    shift: number
  }
  memory: {
    training_dtype: 'bf16' | 'fp16' | 'fp32'
    gradient_checkpointing: boolean
    checkpoint_depth: number
    /** the last N blocks keep their activations: faster, more VRAM */
    checkpoint_skip: number
    /** TREAD: share of a video clip's tokens routed around the middle blocks (0 = off) */
    tread?: number
    /** fused Triton backward for int8 ConvRot layers */
    fused_int8_backward?: boolean
    w8a8_backward?: boolean
    compile_blocks?: boolean
    /** int8 bases: frozen layers run the int8 kernel forward (engine default on) */
    int8_forward?: boolean
    tread_steps?: number
    offloading: boolean
    quantized_backward: boolean
    bypass_mode: boolean
    attention: string
  }
  samples: {
    every: number
    /** a prompt, or a sample with its own size/length/seed */
    prompts: (string | SamplePrompt)[]
    negative: string
    width: number
    height: number
    /** video models: frames per sample (1 = a still) */
    frames?: number
    fps?: number
    steps: number
    cfg: number
    sampler: string
    scheduler: string
    seed: number
    baseline: boolean
    model_name: string | null
    model_weight_dtype: string
    lora_strength: number
    /** a turbo LoRA under every sample (few-step previews), and its strength */
    turbo_lora?: string | null
    turbo_strength?: number
    /** skip-layer guidance: off, skip the layers in the negative pass, or an extra pass without them */
    slg_mode: 'off' | 'uncond' | 'standard'
    slg_layers: number[]
    slg_double_layers: number[]
    slg_scale: number
    slg_start: number
    slg_end: number
  }
  validation: { every: number; images: number; timesteps: number; seed: number }
  /** where checkpoints go and which resume states are kept */
  output: { lora_dir: string | null; keep_states: 'latest' | 'all' | 'none' }
}

export interface Preset<T = unknown> {
  id: string
  name: string
  description: string
  data: T
  builtin: boolean
}

export type PresetKind = 'run' | 'prompts'

/** a sample with its own settings (unset ones follow the run's) */
export interface SamplePrompt {
  prompt: string
  width?: number
  height?: number
  frames?: number
  seed?: number
  /** edit runs: the picture to edit (dataset-relative) */
  control?: string
}

export const promptText = (p: string | SamplePrompt) => (typeof p === 'string' ? p : p.prompt)

export function defaultSpec(): TrainSpec {
  return {
    name: 'my-lora',
    seed: 0,
    save_every: 250,
    model: { kind: 'checkpoint', ckpt_name: null, unet_weight_dtype: 'default' },
    dataset: {
      folder: '',
      resolution: 1024,
      caption_ext: '.txt',
      buckets: true,
      bucket_step: 64,
      caption_dropout: 0,
      masked_loss: false,
      unmasked_weight: 0,
      video_frames: 49,
      video_fps: 0,
      video_mode: 'uniform',
      video_clips: 1,
      video_resolution: 512,
      caption_set: 'default',
      caption_mix: [],
      trigger: '',
      trigger_prepend: true,
    },
    network: {
      algorithm: 'LoRA',
      rank: 16,
      alpha: null,
      lora_dtype: 'fp32',
      targets: 'attn_mlp',
      train_bias_norm: false,
      init_from: null,
      training_adapter: null,
    },
    optim: {
      optimizer: 'AdamW',
      learning_rate: 1e-4,
      weight_decay: 0,
      scheduler: 'constant',
      warmup_steps: 0,
      max_grad_norm: 1,
      steps: 1000,
      batch_size: 1,
      grad_accumulation_steps: 1,
    },
    objective: { loss: 'mse', target: 'native', timesteps: 'uniform', logit_mean: 0, logit_std: 1, shift: 0 },
    memory: {
      training_dtype: 'bf16',
      gradient_checkpointing: true,
      checkpoint_depth: 1,
      checkpoint_skip: 0,
      offloading: false,
      quantized_backward: false,
      bypass_mode: false,
      attention: 'pytorch',
    },
    samples: {
      every: 250, // with each save (save_every's default): each checkpoint gets samples, and a likeness
      prompts: [],
      negative: '',
      width: 1024,
      height: 1024,
      steps: 20,
      cfg: 5,
      sampler: 'euler',
      scheduler: 'normal',
      seed: 42,
      baseline: true,
      model_name: null,
      model_weight_dtype: 'default',
      lora_strength: 1,
      slg_mode: 'off',
      slg_layers: [],
      slg_double_layers: [],
      slg_scale: 3,
      slg_start: 0.01,
      slg_end: 1,
    },
    validation: { every: 100, images: 4, timesteps: 4, seed: 1234 },
    output: { lora_dir: null, keep_states: 'latest' },
  }
}
