// Apps, their batches and the gallery; App-mode workflows (engine server/routes/apps.py).

import { getClientId } from '../comfy'
import { call } from './client'

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

export const appsApi = {
  apps: () => call<{ apps: AppInfo[]; errors: string[] }>('/apps'),
  runApp: (id: string, inputs: Record<string, unknown>, sweep: Record<string, unknown[]>) =>
    call<{ batch: Batch }>(`/apps/${encodeURIComponent(id)}/run`, {
      method: 'POST',
      body: JSON.stringify({ inputs, sweep, client_id: getClientId() }),
    }).then((r) => r.batch),
  batch: (id: string) => call<{ batch: Batch }>(`/apps/batches/${encodeURIComponent(id)}`).then((r) => r.batch),
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
}
