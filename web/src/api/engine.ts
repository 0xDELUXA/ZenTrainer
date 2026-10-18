// Client for the ZenTrainer engine routes (engine/zentrainer/routes.py).
// Types mirror engine/zentrainer/spec.py; keep them in step.

import { getClientId } from './comfy'

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
    caption_ext: string
    buckets: boolean
    bucket_step: number
    caption_dropout: number
    masked_loss: boolean
    edit?: boolean
    edit_method?: string
    edit_template?: string
    edit_grounded?: boolean
    keyframes?: string
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

export type RunStatus =
  | 'queued'
  | 'loading'
  | 'training'
  | 'paused'
  | 'done'
  | 'stopped'
  | 'error'
  | 'interrupted'
  | 'rejected'

export const ACTIVE: RunStatus[] = ['queued', 'loading', 'training', 'paused']

export interface RunMeta {
  run_id: string
  name: string
  prompt_id: string | null
  status: RunStatus
  created: number
  updated: number
  final_step: number | null
  message?: string
  resume_from?: string | null
}

export interface MetricRow {
  run_id: string
  step: number
  total: number
  loss: number | null
  lr: number | null
  grad_norm?: number | null
  epoch?: number
  elapsed: number
  train_time?: number
  step_time?: number
  it_s: number
  /** the step's peak VRAM (bytes, PyTorch allocations) */
  vram_peak?: number
  /** seconds the step spent in each phase */
  phases?: { data: number; forward: number; backward: number; optimizer: number }
}

export interface CheckpointRow {
  run_id: string
  step: number
  path: string
  final: boolean
  lora_name: string
  /** bytes on disk; null if the file is gone */
  size?: number | null
}

export interface SampleRow {
  run_id: string
  step: number
  index: number
  prompt: string
  filename: string
  subfolder: string
  type: string
  model?: string | null
  /** a "Try a prompt" render, apart from the run's samples (index counts them) */
  manual?: boolean
  /** a manual render without the LoRA, to compare with the one at the same index */
  base?: boolean
  seed?: number
  width?: number
  height?: number
  frames?: number
}

export interface RunDetail {
  run: RunMeta
  spec: TrainSpec
  metrics: MetricRow[]
  checkpoints: CheckpointRow[]
  samples: SampleRow[]
  validation: ValidationRow[]
  live: boolean
}

export interface DatasetInfo {
  folder: string
  images: number
  /** how many of `images` are videos */
  videos?: number
  masked?: number
  /** images with a <name>_control input picture (edit pairs) */
  pairs?: number
  /** total size of the media files */
  bytes?: number
  /** newest media file's mtime (s) */
  modified?: number
  steps_per_epoch: number
  captioned: number
  sets: string[]
  thumbs: { subfolder: string; filename: string; path?: string }[]
}

export interface DatasetItem {
  path: string
  subfolder: string
  filename: string
  caption: string
  repeats: number
  width: number | null
  height: number | null
  bytes: number
  mask: string | null
  /** share of the image the mask trains at full weight (0-1) */
  mask_coverage?: number
  /** the mask's shape differs from the image's: stretched over it, it won't line up */
  mask_mismatch?: boolean
  captions: Record<string, string>
  /** edit pairs: the input picture(s), dataset-relative */
  controls?: string[]
  kind?: 'video'
  fps?: number | null
  frames?: number | null
  duration?: number | null
}

export type CaptionMode = 'missing' | 'all' | 'selected'
export interface CaptionJob {
  job_id: string
  dataset: string
  set_name: string
  clip_name: string
  mode?: CaptionMode
  subject?: string
  preview?: number
  status: 'queued' | 'running' | 'done' | 'error' | 'stopped' | 'undone'
  done: number
  total: number
  file: string
  text: string
  message: string
  /** paths it wrote (from the job endpoints; not in live events) */
  written?: string[]
  /** how many existing captions it replaced (undo puts them back) */
  replaced?: number
  results?: { path: string; text: string }[]
  /** the job's log (job endpoint only) */
  log?: { t: number; level: 'info' | 'warn' | 'error'; msg: string }[]
  /** path -> the caption it wrote (job endpoint only) */
  captions?: Record<string, string>
  prompt?: string
  created?: number
  finished?: number | null
}
export interface CaptionRequest {
  clip_name: string
  clip_type: string
  set: string
  style?: string
  prompt?: string
  mode: CaptionMode
  paths?: string[]
  subject?: string
  preview?: number
  replace_default?: boolean
}
export interface CaptionStyle {
  id: string
  label: string
  prompt: string
}

export interface MemSample {
  t: number
  ram_total?: number
  ram_used?: number
  proc_rss?: number
  vram_total?: number
  vram_used?: number
  torch_allocated?: number
  torch_reserved?: number
  aimdo_used?: number
}

export interface AppInput {
  id: string
  type: 'prompt' | 'text' | 'number' | 'slider' | 'seed' | 'toggle' | 'choice' | 'model' | 'family' | 'model-file' | 'loras' | 'lora' | 'resolution' | 'size'
  /** the newer name for type */
  component?: string
  label?: string
  bind?: string | string[]
  default?: unknown
  min?: number
  max?: number
  step?: number
  options?: string[]
  options_from?: string
  sweep?: boolean
  group?: string
  placeholder?: string
  hint?: string
  /** help behind an ⓘ */
  tip?: string
  /** "advanced" groups it under the main controls */
  section?: 'main' | 'advanced'
  /** bound only where the selected workflow has its node */
  optional?: boolean
}
export interface AppInfo {
  id: string
  name: string
  description: string
  icon?: string
  category: string
  inputs: AppInput[]
  outputs: { id: string; from: string; view: 'gallery' | 'grid' }[]
  builtin: boolean
  layouts: string[]
  /** the shared view that draws it: "generate", or "classic" (the older app page) */
  layout: string
  /** has a cover image at /apps/<id>/cover */
  cover: boolean
  /** model families with a workflow, for an app that picks one per model */
  families: string[]
}
export interface BatchCell {
  coords: Record<string, unknown>
  /** the values a workflow-app run actually used (random seeds filled in) */
  values?: Record<string, unknown>
  prompt_id: string
  status: 'queued' | 'done' | 'error'
  images: { filename: string; subfolder: string; type: string }[]
  error?: string | null
}
/** a workflow app: Text to Image, a ComfyUI template in App mode, or the user's own workflow */
export interface WorkflowApp {
  id: string
  name: string
  source: 'builtin' | 'template' | 'workflow'
  media: string
  description: string
  icon?: string
  thumbnail?: string
  /** runs partner API nodes (spends Comfy credits) */
  api?: boolean
  date?: string | null
}
export interface WorkflowAppInput {
  key: string
  node_id: string
  input: string
  class_type: string
  /** Text to Image's inputs found by role */
  role: 'prompt' | 'negative' | 'enhance' | 'width' | 'height' | 'aspect_ratio' | 'megapixels' | 'seed' | 'steps' | 'cfg' | null
  label: string
  title: string
  value: unknown
  spec: {
    type?: string
    choices?: (string | number)[]
    min?: number
    max?: number
    step?: number
    round?: number
    default?: unknown
    multiline?: boolean
    image_upload?: boolean
    control_after_generate?: boolean
  }
}
export interface WorkflowAppDetail {
  id: string
  family?: string
  template?: string
  needs_family?: boolean
  inputs: WorkflowAppInput[]
  outputs: string[]
}
/** one image an app made, with the values that made it */
export interface GalleryItem {
  batch_id: string
  prompt_id: string
  app: string
  created: number
  image: { filename: string; subfolder: string; type: string }
  values: Record<string, unknown>
}
/** a model ComfyUI holds, and how much of it (per block) is in VRAM */
export interface LoadedModel {
  /** the file it was loaded from (several joined with " + ") */
  file?: string | null
  kind?: 'diffusion model' | 'text encoder' | 'vae'
  name: string
  dynamic?: boolean
  bytes: number
  vram: number
  groups?: { name: string; bytes: number; vram: number }[]
  error?: string
}
export interface StorageReport {
  disks: { labels: string[]; path: string; total: number; used: number; free: number }[]
  runs: { run_id: string; name: string; status: RunStatus; created: number; checkpoints: number; checkpoint_count: number; states: number; samples: number }[]
  datasets: { folder: string; count: number; images: number; cache: number }[]
  apps: number
}
export interface Batch {
  batch_id: string
  app: string
  inputs: Record<string, unknown>
  sweep: Record<string, unknown[]>
  family?: string | null
  created: number
  cells: BatchCell[]
  done: number
  total: number
  status: 'queued' | 'running' | 'done'
}

export interface ComfyVersion {
  path: string
  version?: string
  commit?: string
  describe?: string
  dirty?: boolean
  pin?: { ref: string; describe?: string }
  matches_pin?: boolean
  upstream?: { commit: string; describe: string; behind: number }
  upstream_error?: string
}

export interface QueueState {
  paused: boolean
  current: RunMeta | null
  items: RunMeta[]
}

export interface OptimizerInfo {
  id: string
  label: string
  available: boolean
  pip: string | null
  lr_hint: string
  note: string
}

export interface Preset<T = unknown> {
  id: string
  name: string
  description: string
  data: T
  builtin: boolean
}
export type PresetKind = 'run' | 'prompts'

export interface ValidationRow {
  run_id: string
  step: number
  val_loss: number
  by_noise: Record<string, number>
}

export interface Health {
  version: string
  /** the engine's code on disk is newer than what's running: restart to load it */
  stale?: boolean
  capabilities: {
    optimizers?: OptimizerInfo[]
    train_attention?: string[]
    infer_attention?: string[]
    device?: string
    device_name?: string
    vram_total?: number
    vram_free?: number
  }
}

export type Command = 'pause' | 'resume' | 'stop' | 'save' | 'sample' | 'set_lr'

export class EngineError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: unknown,
  ) {
    super(message)
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/zentrainer/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new EngineError(body?.error ?? res.statusText, res.status, body)
  return body as T
}

const runPath = (id: string) => `/runs/${encodeURIComponent(id)}`

// ---- model families + downloads (engine/zentrainer/families.py, downloads.py) -------------

export type ModelFolder = 'diffusion_models' | 'text_encoders' | 'vae' | 'checkpoints'
export interface ModelFile {
  name: string
  folder: ModelFolder
  url: string
  size: number | null
  distilled: boolean
  from_template: boolean
  /** the name ComfyUI's loaders list it by; null when not installed */
  installed: string | null
}
export interface SampleDefaults {
  model: string
  steps: number
  cfg: number
  sampler: string
  scheduler: string
}
export interface ModelFamily {
  id: string
  name: string
  kind: 'split' | 'checkpoint'
  clip_type: string | null
  repos: string[]
  templates: string[]
  files: Partial<Record<ModelFolder, ModelFile[]>>
  sample_defaults: SampleDefaults | Record<string, never>
  repo_listed: boolean
  /** templates: from ComfyUI's workflow templates; repo: a Comfy-Org or maker's ComfyUI repo with no template yet */
  source: 'templates' | 'repo'
  /** hand-picked settings for this model, when we have them */
  profile: ModelProfile | null
  /** ISO date: first template, or repo creation */
  added: string | null
  /** the repo asks you to accept a license (and set HF_TOKEN) before downloading */
  gated: boolean
  recommended: { train: Partial<Record<ModelFolder, string | null>>; sample: SampleDefaults | null }
  missing: { folder: ModelFolder; name: string; size: number | null }[]
}
/** engine/zentrainer/adapters.py listing() */
export interface TrainingAdapter {
  id: string
  label: string
  /** local file name */
  name: string
  /** the name ComfyUI lists it by when installed (what the spec stores), else null */
  installed: string | null
  source: 'curated' | 'custom' | 'local'
  /** suits the family asked about */
  fits: boolean
  /** worth offering when the family is picked */
  suggest: boolean
  families: string[]
  note: string
  repo: string
  file: string
  size: number | null
  /** the repo page */
  url: string
}
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
/** engine/zentrainer/segment.py: one SAM3 request from the mask editor */
export interface SegmentJob {
  job_id: string
  status: 'queued' | 'running' | 'done' | 'error'
  message: string
}
export interface ModelProfile {
  id: string
  families: string[]
  /** recipe: the model makers' published settings; verified: we've trained with it and checked */
  status: 'recipe' | 'verified'
  summary: string
  sources: { label: string; url: string }[]
  train: Record<string, unknown>
  samples: Partial<TrainSpec['samples']>
  notes: string[]
}
export interface ModelDownload {
  id: string
  family: string
  folder: ModelFolder | 'loras'
  name: string
  target: string
  total: number | null
  done: number
  status: 'queued' | 'downloading' | 'done' | 'error' | 'cancelled'
  error: string | null
  speed: number
  created: number
}

export const engine = {
  families: () => call<{ families: ModelFamily[] }>('/families').then((r) => r.families),
  family: (id: string, repo = false, refresh = false) =>
    call<ModelFamily>(`/families/${encodeURIComponent(id)}?repo=${repo ? 1 : 0}${refresh ? '&refresh=1' : ''}`),
  downloads: () =>
    call<{ downloads: ModelDownload[]; targets: Partial<Record<ModelFolder, string>>; hf_token: boolean }>('/downloads'),
  download: (family: string, folder: ModelFolder, name: string) =>
    call<ModelDownload>('/downloads', { method: 'POST', body: JSON.stringify({ family, folder, name }) }),
  cancelDownload: (id: string) => call<{ cancelled: boolean }>(`/downloads/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  clearDownloads: () => call<{ ok: true }>('/downloads/clear', { method: 'POST' }),
  segment: (name: string, body: { path: string; text?: string; positive?: { x: number; y: number }[]; negative?: { x: number; y: number }[]; threshold?: number }) =>
    call<{ job: SegmentJob }>(`${dsPath(name)}/segment`, { method: 'POST', body: JSON.stringify(body) }),
  segmentJob: (id: string) => call<{ job: SegmentJob }>(`/segment/${encodeURIComponent(id)}`).then((r) => r.job),
  saveMask: (name: string, path: string, png: string) =>
    call<{ mask: string | null }>(`${dsPath(name)}/mask`, { method: 'PUT', body: JSON.stringify({ path, png }) }),
  splitVideos: (name: string, body: { seconds: number; count: number; longer_than: number }) =>
    call<{ clips: number; moved: string[] }>(`${dsPath(name)}/split`, { method: 'POST', body: JSON.stringify(body) }),
  mergeLoras: (body: { members: { name: string; strength: number }[]; out_name: string; max_rank?: number; energy?: number }) =>
    call<{ lora_name: string; mean_rank?: number; kept_energy?: number; bytes: number }>('/loras/merge', { method: 'POST', body: JSON.stringify(body) }),
  adapters: (family?: string | null) =>
    call<{ adapters: TrainingAdapter[] }>(`/adapters${family ? `?family=${encodeURIComponent(family)}` : ''}`).then((r) => r.adapters),
  addAdapter: (link: string, family?: string | null, file?: string) =>
    call<{ adapter?: TrainingAdapter; choices?: { file: string; size: number | null }[]; repo?: string }>('/adapters', {
      method: 'POST',
      body: JSON.stringify({ link, family, file }),
    }),
  downloadAdapter: (id: string) => call<ModelDownload>('/adapters/download', { method: 'POST', body: JSON.stringify({ id }) }),
  removeAdapter: (id: string) => call<{ removed: boolean }>(`/adapters?id=${encodeURIComponent(id)}`, { method: 'DELETE' }),
  health: () => call<Health>('/health'),
  datasets: () => call<{ datasets: DatasetInfo[] }>('/datasets').then((r) => r.datasets),
  runs: () => call<{ runs: RunMeta[] }>('/runs').then((r) => r.runs),
  run: (id: string) => call<RunDetail>(runPath(id)),
  /** position "next" starts it now if idle (else right after the current run); "end" queues it last */
  /** change a run still waiting in the queue, or one that ended before training (it's requeued) */
  editRun: (id: string, spec: TrainSpec) =>
    call<{ run: RunMeta; requeued: boolean }>(`/runs/${encodeURIComponent(id)}/spec`, {
      method: 'PUT',
      body: JSON.stringify({ spec, client_id: getClientId() }),
    }),
  start: (spec: TrainSpec, position: 'next' | 'end' = 'next') =>
    call<{ run: RunMeta }>('/runs', {
      method: 'POST',
      body: JSON.stringify({ spec, position, client_id: getClientId() }),
    }).then((r) => r.run),
  queue: () => call<QueueState>('/queue'),
  editQueue: (body: { action: 'pause' | 'resume' } | { action: 'order'; order: string[] } | { action: 'top' | 'remove'; run_id: string }) =>
    call('/queue', { method: 'POST', body: JSON.stringify(body) }),
  presets: <T = unknown>(kind: PresetKind) => call<{ presets: Preset<T>[] }>(`/presets/${kind}`).then((r) => r.presets),
  savePreset: (kind: PresetKind, name: string, data: unknown, description = '') =>
    call<{ preset: Preset }>(`/presets/${kind}`, { method: 'POST', body: JSON.stringify({ name, data, description }) }),
  deletePreset: (kind: PresetKind, id: string) => call(`/presets/${kind}/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  system: (since = 0) => call<{ interval: number; samples: MemSample[] }>(`/system?since=${since}`),
  apps: () => call<{ apps: AppInfo[]; errors: string[] }>('/apps'),
  runApp: (id: string, inputs: Record<string, unknown>, sweep: Record<string, unknown[]>) =>
    call<{ batch: Batch }>(`/apps/${encodeURIComponent(id)}/run`, {
      method: 'POST',
      body: JSON.stringify({ inputs, sweep, client_id: getClientId() }),
    }).then((r) => r.batch),
  batch: (id: string) => call<{ batch: Batch }>(`/apps/batches/${encodeURIComponent(id)}`).then((r) => r.batch),
  storage: () => call<StorageReport>('/storage'),
  loraPlaces: () =>
    call<{ loras: { path: string; free: number; total: number; default: boolean; inside_install: boolean }[] }>('/storage/places').then((r) => r.loras),
  disks: () => call<Pick<StorageReport, 'disks'>>('/storage?disks=1').then((r) => r.disks),
  cleanStorage: (body: { kind: 'states'; run_id: string } | { kind: 'states_all' } | { kind: 'cache'; folder: string }) =>
    call<{ freed: number; runs?: number }>('/storage/clean', { method: 'POST', body: JSON.stringify(body) }),
  loadedModels: () => call<{ models: LoadedModel[] }>('/system/models').then((r) => r.models),
  wapps: () => call<{ apps: WorkflowApp[] }>('/wapps').then((r) => r.apps),
  wapp: (id: string, family?: string | null) =>
    call<WorkflowAppDetail>(`/wapps/describe?id=${encodeURIComponent(id)}${family ? `&family=${encodeURIComponent(family)}` : ''}`),
  gallery: (app?: string, limit = 500) =>
    call<{ items: GalleryItem[]; total: number }>(`/gallery?limit=${limit}${app ? `&app=${encodeURIComponent(app)}` : ''}`),
  deleteGalleryItem: (batchId: string, promptId: string) =>
    call<{ removed: number }>(`/gallery/${encodeURIComponent(batchId)}/${encodeURIComponent(promptId)}`, { method: 'DELETE' }),
  appDefaults: (id: string, family?: string | null) =>
    call<{ values: Record<string, unknown>; inputs: string[] }>(`/apps/${encodeURIComponent(id)}/defaults${family ? `?family=${encodeURIComponent(family)}` : ''}`),
  appBatches: (id: string, limit = 12) =>
    call<{ batches: Batch[] }>(`/apps/${encodeURIComponent(id)}/batches?limit=${limit}`).then((r) => r.batches),
  comfyui: (check = false) => call<ComfyVersion>(`/comfyui${check ? '?check=1' : ''}`),
  restart: (force = false) => call<{ ok: true }>('/restart', { method: 'POST', body: JSON.stringify({ force }) }),
  control: (id: string, command: Command, args?: Record<string, unknown>) =>
    call<{ ok: true }>(`${runPath(id)}/control`, { method: 'POST', body: JSON.stringify({ command, args }) }),
  resume: (id: string, checkpoint?: string) =>
    call<{ run: RunMeta }>(`${runPath(id)}/resume`, {
      method: 'POST',
      body: JSON.stringify({ checkpoint, client_id: getClientId() }),
    }).then((r) => r.run),

  dataset: (name: string) => call<{ folder: string; items: DatasetItem[]; sets: string[] }>(dsPath(name)),
  createDataset: (name: string) => call<{ ok: true }>('/datasets', { method: 'POST', body: JSON.stringify({ name }) }),
  setCaption: (name: string, path: string, caption: string, set = 'default') =>
    call<{ ok: true }>(`${dsPath(name)}/caption`, { method: 'PUT', body: JSON.stringify({ path, caption, set }) }),
  bulkCaptions: (name: string, op: 'prefix' | 'suffix' | 'replace', text: string, find = '', set = 'default') =>
    call<{ changed: number }>(`${dsPath(name)}/bulk`, { method: 'POST', body: JSON.stringify({ op, text, find, set }) }),
  captionInfo: (dataset?: string) =>
    call<{ styles: CaptionStyle[]; models: string[]; jobs: CaptionJob[] }>(`/caption${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ''}`),
  startCaption: (name: string, body: CaptionRequest) =>
    call<{ job: CaptionJob }>(`${dsPath(name)}/caption`, { method: 'POST', body: JSON.stringify({ ...body, client_id: getClientId() }) }),
  captionJobs: () => call<{ jobs: CaptionJob[] }>('/caption?all=1').then((r) => r.jobs),
  captionActive: () => call<{ jobs: CaptionJob[] }>('/caption?active=1').then((r) => r.jobs),
  captionJob: (id: string) => call<{ job: CaptionJob }>(`/caption/${encodeURIComponent(id)}`).then((r) => r.job),
  captionAction: (id: string, action: 'cancel' | 'undo') =>
    call<{ job: CaptionJob; restored?: number }>(`/caption/${encodeURIComponent(id)}/${action}`, { method: 'POST' }),
  deleteItem: (name: string, path: string) =>
    call<{ ok: true }>(`${dsPath(name)}/item?path=${encodeURIComponent(path)}`, { method: 'DELETE' }),
  async upload(name: string, files: File[]): Promise<{ saved: string[]; errors: string[] }> {
    const form = new FormData()
    for (const f of files) form.append('file', f, f.name)
    const res = await fetch(`/zentrainer/api${dsPath(name)}/upload`, { method: 'POST', body: form })
    if (!res.ok) throw new EngineError(res.statusText, res.status, null)
    return res.json()
  },
}

const dsPath = (name: string) => `/datasets/${encodeURIComponent(name)}`

/** ComfyUI's /view URL for an image in its input/output folders. */
export function viewUrl(f: { filename: string; subfolder: string; type?: string }): string {
  const q = new URLSearchParams({ filename: f.filename, subfolder: f.subfolder, type: f.type ?? 'output' })
  return `/view?${q}`
}

/** Websocket event names; mirror engine/zentrainer/events.py. */
export const EVENTS = {
  metrics: 'zentrainer.metrics',
  checkpoint: 'zentrainer.checkpoint',
  status: 'zentrainer.status',
  sample: 'zentrainer.sample',
  sampleProgress: 'zentrainer.sample_progress',
  validation: 'zentrainer.validation',
  caption: 'zentrainer.caption',
  download: 'zentrainer.download',
} as const

export interface SampleProgress {
  run_id: string
  phase: 'requested' | 'sampling' | 'done'
  step?: number
  index?: number
  count?: number
  prompt?: string
  s_step?: number
  s_total?: number
  preview?: string
  /** a "Try a prompt" render (previewed in the Manual tab), its try number, and whether it's the no-LoRA one */
  manual?: boolean
  manual_index?: number
  base?: boolean
}

/** Cached server-side thumbnail of a dataset image or mask. */
export function thumbUrl(dataset: string, path: string, size = 384): string {
  return `/zentrainer/api/datasets/${encodeURIComponent(dataset)}/thumb?${new URLSearchParams({ path, size: String(size) })}`
}

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
      every: 0,
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
