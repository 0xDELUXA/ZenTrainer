// Datasets, captions and auto-captioning, masks (engine server/routes/datasets.py).

import { getClientId } from '../comfy'
import { call, EngineError } from './client'
import type { ModelDownload } from './models'

/** a dataset's health now and over time (engine dataset_stats.snapshot) */
export interface DatasetStats {
  current: { metrics: Record<string, number | null | Record<string, number>>; scores: Record<string, number>; flags?: Record<string, string[]> }
  history: { at: number; scores: Record<string, number>; metrics: Record<string, unknown> }[]
}

/** a dataset's own settings (engine datasets.get_settings) */
export interface DatasetSettings {
  /** face report on open, faces in the stats, sample likeness in runs */
  faces: boolean
  /** what auto-captions call the main subject (a trigger word, a name) */
  caption_subject?: string
}

/** a dataset's faces (engine faces.dataset_faces) */
export interface DatasetFaces {
  available: boolean
  hint?: string
  /** faces: in the trained (white) part of the mask; masked: faces the mask leaves out */
  items?: Record<string, { faces: number; masked?: number; share: number; likeness: number | null; outlier: boolean }>
  with_face?: number
  total?: number
  pack?: string
  /** median of each face against the others' average; one person ≳ 0.5, strangers ≲ 0.2 */
  identity?: number | null
  one_identity?: boolean
}

export interface DatasetInfo {
  folder: string
  images: number
  /** how many of `images` are videos */
  videos?: number
  masked?: number
  /** the last stats snapshot's 0-1 scores (captions, consistency, resolution, uniqueness, faces|balance) */
  health?: { scores: Record<string, number>; set: string } | null
  /** pictures with an input in any control set (edit pairs) */
  pairs?: number
  /** control sets (edit training's inputs) and how many pictures have an input in each */
  control_sets?: Record<string, number>
  /** total size of the media files */
  bytes?: number
  /** newest media file's mtime (s) */
  modified?: number
  steps_per_epoch: number
  captioned: number
  sets: string[]
  thumbs: { subfolder: string; filename: string; path?: string }[]
}

/** a curated captioning LoRA (engine captioner.TE_LORA_CATALOG) */
export interface CaptionLora {
  id: string
  label: string
  repo: string
  file: string
  size: number
  /** regex on the captioning model's file name: the base it was extracted from */
  base: string
  note: string
  /** its loras/ name when downloaded */
  installed: string | null
}

/** what a captioning model takes in (engine captioner.model_senses) */
export interface ModelSenses {
  image: boolean
  /** "clip": a whole timestamped clip (Gemma 4); "stills": a few frames spread over it */
  video: 'clip' | 'stills'
  audio: boolean
  frames: number
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
  /** caption sets it's muted in: kept, but not trained on in those sets */
  muted?: string[]
  /** edit training's input picture(s) per control set, dataset-relative */
  control_sets?: Record<string, string[]>
  /** each input's [width, height], by its path */
  control_sizes?: Record<string, [number | null, number | null]>
  kind?: 'video'
  fps?: number | null
  frames?: number | null
  duration?: number | null
}

export type CaptionMode = 'missing' | 'all' | 'selected'

export interface CaptionJob {
  job_id: string
  /** a prompt rewrite: the prompts (no dataset; the rewrites are its results) */
  rewrite?: string[]
  /** how its answers are kept ("fibo_json": structured captions) */
  format?: CaptionFormat
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
  results?: { path: string; text: string; prompt?: string }[]
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
  /** with a prompt: how its answer is kept (a style brings its own) */
  format?: CaptionFormat
  mode: CaptionMode
  paths?: string[]
  subject?: string
  preview?: number
  replace_default?: boolean
}

/** "text": the answer as written; "fibo_json": a Fibo 2.0 structured caption, checked and compact */
export type CaptionFormat = 'text' | 'fibo_json'

export interface CaptionStyle {
  id: string
  label: string
  prompt: string
  format?: CaptionFormat
  max_length?: number
}

/** engine/zentrainer/datasets/segment.py: one SAM3 request from the mask editor */
export interface SegmentJob {
  job_id: string
  status: 'queued' | 'running' | 'done' | 'error'
  message: string
}

export const dsPath = (name: string) => `/datasets/${encodeURIComponent(name)}`

/** Cached server-side thumbnail of a dataset image or mask. */
export function thumbUrl(dataset: string, path: string, size = 384): string {
  return `/zentrainer/api/datasets/${encodeURIComponent(dataset)}/thumb?${new URLSearchParams({ path, size: String(size) })}`
}

export const datasetsApi = {
  segment: (name: string, body: { path: string; text?: string; positive?: { x: number; y: number }[]; negative?: { x: number; y: number }[]; threshold?: number }) =>
    call<{ job: SegmentJob }>(`${dsPath(name)}/segment`, { method: 'POST', body: JSON.stringify(body) }),
  segmentJob: (id: string) => call<{ job: SegmentJob }>(`/segment/${encodeURIComponent(id)}`).then((r) => r.job),
  saveMask: (name: string, path: string, png: string) =>
    call<{ mask: string | null }>(`${dsPath(name)}/mask`, { method: 'PUT', body: JSON.stringify({ path, png }) }),
  /** one mask onto many images of its size; cutout keeps each image's own mask and takes the new one's black out of it */
  applyMasks: (name: string, paths: string[], png: string, mode: 'cutout' | 'replace') =>
    call<{ batch: string | null; results: { path: string; mask?: string; skipped?: string }[] }>(`${dsPath(name)}/masks/apply`, {
      method: 'POST',
      body: JSON.stringify({ paths, png, mode }),
    }),
  undoMasks: (name: string, batch: string) =>
    call<{ restored: number }>(`${dsPath(name)}/masks/undo`, { method: 'POST', body: JSON.stringify({ batch }) }),
  splitVideos: (name: string, body: { seconds: number; count: number; longer_than: number }) =>
    call<{ clips: number; moved: string[] }>(`${dsPath(name)}/split`, { method: 'POST', body: JSON.stringify(body) }),
  datasetStats: (name: string, set = 'default') =>
    call<DatasetStats>(`/datasets/${encodeURIComponent(name)}/stats?set=${encodeURIComponent(set)}`),
  datasetMuted: (name: string, set: string, paths: string[], muted: boolean) =>
    call<{ muted: Record<string, string[]> }>(`/datasets/${encodeURIComponent(name)}/muted`, {
      method: 'PUT',
      body: JSON.stringify({ set, paths, muted }),
    }),
  datasetFaces: (name: string) => call<DatasetFaces>(`/datasets/${encodeURIComponent(name)}/faces`),
  addCaptionLora: (link: string, file?: string, id?: string) =>
    call<{ download?: ModelDownload; name?: string; choices?: { file: string; size: number | null }[]; repo?: string }>('/caption/loras', {
      method: 'POST',
      body: JSON.stringify({ link, file, id }),
    }),
  datasets: () => call<{ datasets: DatasetInfo[] }>('/datasets').then((r) => r.datasets),
  dataset: (name: string) =>
    call<{ folder: string; items: DatasetItem[]; sets: string[]; control_sets: string[]; settings?: DatasetSettings }>(dsPath(name)),
  datasetSettings: (name: string, changes: Partial<DatasetSettings>) =>
    call<DatasetSettings>(`${dsPath(name)}/settings`, { method: 'PUT', body: JSON.stringify(changes) }),
  createDataset: (name: string) => call<{ ok: true }>('/datasets', { method: 'POST', body: JSON.stringify({ name }) }),
  setCaption: (name: string, path: string, caption: string, set = 'default') =>
    call<{ ok: true }>(`${dsPath(name)}/caption`, { method: 'PUT', body: JSON.stringify({ path, caption, set }) }),
  bulkCaptions: (name: string, op: 'prefix' | 'suffix' | 'replace', text: string, find = '', set = 'default') =>
    call<{ changed: number }>(`${dsPath(name)}/bulk`, { method: 'POST', body: JSON.stringify({ op, text, find, set }) }),
  captionInfo: (dataset?: string) =>
    call<{ styles: CaptionStyle[]; notes?: { video: string; sound: string; subject: string }; models: string[]; hearing?: string[]; loras?: string[]; lora_catalog?: CaptionLora[]; senses?: Record<string, ModelSenses>; jobs: CaptionJob[] }>(`/caption${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ''}`),
  startCaption: (name: string, body: CaptionRequest) =>
    call<{ job: CaptionJob }>(`${dsPath(name)}/caption`, { method: 'POST', body: JSON.stringify({ ...body, client_id: getClientId() }) }),
  captionJobs: () => call<{ jobs: CaptionJob[] }>('/caption?all=1').then((r) => r.jobs),
  captionActive: () => call<{ jobs: CaptionJob[] }>('/caption?active=1').then((r) => r.jobs),
  captionJob: (id: string) => call<{ job: CaptionJob }>(`/caption/${encodeURIComponent(id)}`).then((r) => r.job),
  /** rewrite prompts with a language model, as Fibo 2.0 structured captions by default: a job; its results are the rewrites */
  rewritePrompts: (body: { clip_name: string; prompts: string[]; format?: CaptionFormat; prompt?: string }) =>
    call<{ job: CaptionJob }>('/caption/rewrite', { method: 'POST', body: JSON.stringify({ ...body, client_id: getClientId() }) }).then((r) => r.job),
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
  // control sets: edit training's input pictures, a named layer per set
  createControlSet: (name: string, set: string) => call<{ ok: true }>(ctlPath(name, set), { method: 'POST' }),
  /** files named like the pictures they're the input of (`<name>.png`, `<name>.2.png` for a second) */
  async uploadControls(name: string, set: string, files: File[]): Promise<{ saved: string[]; errors: string[] }> {
    const form = new FormData()
    for (const f of files) form.append('file', f, f.name)
    const res = await fetch(`/zentrainer/api${ctlPath(name, set)}`, { method: 'POST', body: form })
    if (!res.ok) throw new EngineError(res.statusText, res.status, null)
    return res.json()
  },
  deleteControl: (name: string, set: string, path: string) =>
    call<{ ok: true }>(`${ctlPath(name, set)}?path=${encodeURIComponent(path)}`, { method: 'DELETE' }),
  deleteControlSet: (name: string, set: string) => call<{ deleted: number }>(ctlPath(name, set), { method: 'DELETE' }),
  renameControlSet: (name: string, set: string, to: string) =>
    call<{ moved: number }>(`${ctlPath(name, set)}/rename`, { method: 'POST', body: JSON.stringify({ to }) }),
}

const ctlPath = (name: string, set: string) => `${dsPath(name)}/controls/${encodeURIComponent(set)}`
