// Model families, downloads, training adapters, noise schedules, LoRA merging (engine server/routes/models.py, loras.py).

import { call } from './client'
import type { TrainSpec } from './spec'

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

/** engine/zentrainer/models/noise.py profile(): σ from full noise (first) to clean (last), evenly spaced */
export interface NoiseProfile {
  known: boolean
  model?: string
  flow?: boolean
  shift?: number | null
  sigmas?: number[]
}

/** training: on while training only; base: on always (training, samples, use) */
export type AdapterKind = 'training' | 'base'

/** engine/zentrainer/models/adapters.py listing() */
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

export const modelsApi = {
  families: () => call<{ families: ModelFamily[] }>('/families').then((r) => r.families),
  family: (id: string, repo = false, refresh = false) =>
    call<ModelFamily>(`/families/${encodeURIComponent(id)}?repo=${repo ? 1 : 0}${refresh ? '&refresh=1' : ''}`),
  downloads: () =>
    call<{ downloads: ModelDownload[]; targets: Partial<Record<ModelFolder, string>>; hf_token: boolean }>('/downloads'),
  download: (family: string, folder: ModelFolder, name: string) =>
    call<ModelDownload>('/downloads', { method: 'POST', body: JSON.stringify({ family, folder, name }) }),
  cancelDownload: (id: string) => call<{ cancelled: boolean }>(`/downloads/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  clearDownloads: () => call<{ ok: true }>('/downloads/clear', { method: 'POST' }),
  mergeLoras: (body: { members: { name: string; strength: number }[]; out_name: string; max_rank?: number; energy?: number }) =>
    call<{ lora_name: string; mean_rank?: number; kept_energy?: number; bytes: number }>('/loras/merge', { method: 'POST', body: JSON.stringify(body) }),
  /** a model's own noise schedule from its file's header (engine/zentrainer/models/noise.py) */
  noise: (folder: 'diffusion_models' | 'checkpoints', name: string) =>
    call<NoiseProfile>(`/noise?${new URLSearchParams({ folder, name })}`),
  adapters: (family?: string | null, kind: AdapterKind = 'training') =>
    call<{ adapters: TrainingAdapter[] }>(
      `/adapters?kind=${kind}${family ? `&family=${encodeURIComponent(family)}` : ''}`,
    ).then((r) => r.adapters),
  addAdapter: (link: string, family?: string | null, file?: string, kind: AdapterKind = 'training') =>
    call<{ adapter?: TrainingAdapter; choices?: { file: string; size: number | null }[]; repo?: string }>('/adapters', {
      method: 'POST',
      body: JSON.stringify({ link, family, file, kind }),
    }),
  downloadAdapter: (id: string) => call<ModelDownload>('/adapters/download', { method: 'POST', body: JSON.stringify({ id }) }),
  removeAdapter: (id: string) => call<{ removed: boolean }>(`/adapters?id=${encodeURIComponent(id)}`, { method: 'DELETE' }),
  loraPlaces: () =>
    call<{ loras: { path: string; free: number; total: number; default: boolean; inside_install: boolean }[] }>('/storage/places').then((r) => r.loras),
}
