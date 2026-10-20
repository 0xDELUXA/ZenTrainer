// Model downloads, kept live from the engine's websocket events. Shared by the model
// picker (per-file progress) and the sidebar (anything downloading).

import { computed, reactive, ref } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { EVENTS, engine, type ModelDownload, type ModelFolder } from '@/api/engine'

export const downloads = reactive<Record<string, ModelDownload>>({})
export const targets = ref<Partial<Record<ModelFolder, string>>>({})
/** the server has a Hugging Face token (HF_TOKEN), which gated repos need */
export const hfToken = ref(false)
/** bumped whenever a download finishes, so install state can be re-read */
export const finished = ref(0)

export const activeDownloads = computed(() =>
  Object.values(downloads).filter((d) => d.status === 'queued' || d.status === 'downloading'),
)
export const downloadFor = (folder: string, name: string) =>
  Object.values(downloads)
    .filter((d) => d.folder === folder && d.name === name)
    .sort((a, b) => b.created - a.created)[0]

function put(d: ModelDownload) {
  const was = downloads[d.id]?.status
  downloads[d.id] = d
  if (d.status === 'done' && was !== 'done') finished.value++
}

let started = false
export async function startDownloadSync(): Promise<void> {
  if (started) return
  started = true
  onComfyMessage((m) => {
    if (m.type === EVENTS.download) put(m.data as unknown as ModelDownload)
  })
  try {
    const r = await engine.downloads()
    targets.value = r.targets
    hfToken.value = r.hf_token
    for (const d of r.downloads) put(d)
  } catch {
    // engine unreachable; events will fill it in
  }
}

export const fmtBytes = (b?: number | null) =>
  b == null ? '' : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${Math.round(b / 1e6)} MB` : `${Math.round(b / 1e3)} KB`

const DAY = 864e5
/** added to ComfyUI (first template, or repo creation) within the last month */
export const isNew = (added?: string | null) => !!added && Date.now() - Date.parse(added) < 30 * DAY
/** "Oct 2026" */
export const fmtAdded = (added?: string | null) =>
  added ? new Date(added).toLocaleDateString(undefined, { month: 'short', year: 'numeric' }) : ''
