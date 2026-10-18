// Thin client for the stock ComfyUI API: node schemas (for enum choices) and the
// websocket. Framework-free on purpose, so this layer survives a UI rewrite.
//
// Why not @comfyorg/sdk: it targets the v2 API (Comfy Cloud, or a local server behind
// comfy-api-proxy) and has no object_info or local model listing. It's the right
// tool later for running inference apps on Cloud; the local engine talks to ComfyUI.

export interface ComfyMessage {
  type: string
  data: Record<string, unknown>
}

type Listener = (msg: ComfyMessage) => void

const clientId = crypto.randomUUID()
const listeners = new Set<Listener>()
let socket: WebSocket | null = null

export function getClientId(): string {
  return clientId
}

type PreviewListener = (promptId: string | null, url: string) => void
const previewListeners = new Set<PreviewListener>()
let lastPreview: string | null = null

/** Live sampling previews: ComfyUI's binary frames, as an object URL per frame (the previous
 *  one is revoked) tagged with the prompt it belongs to when ComfyUI says. */
export function onComfyPreview(fn: PreviewListener): () => void {
  previewListeners.add(fn)
  connect()
  return () => previewListeners.delete(fn)
}

function onBinary(buf: ArrayBuffer): void {
  if (!previewListeners.size || buf.byteLength < 8) return
  const view = new DataView(buf)
  const type = view.getUint32(0)
  let promptId: string | null = null
  let mime = 'image/jpeg'
  let bytes: ArrayBuffer
  if (type === 4) {
    // [type][metadata length][metadata JSON][image]
    const len = view.getUint32(4)
    try {
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, len))) as { prompt_id?: string; image_type?: string }
      promptId = meta.prompt_id ?? null
      mime = meta.image_type ?? mime
    } catch {
      return
    }
    bytes = buf.slice(8 + len)
  } else if (type === 1) {
    // [type][image format: 1 jpeg, 2 png][image]
    mime = view.getUint32(4) === 2 ? 'image/png' : 'image/jpeg'
    bytes = buf.slice(8)
  } else return
  if (lastPreview) URL.revokeObjectURL(lastPreview)
  lastPreview = URL.createObjectURL(new Blob([bytes], { type: mime }))
  for (const fn of previewListeners) fn(promptId, lastPreview)
}

/** Subscribe to every JSON message on ComfyUI's /ws. Reconnects on drop. */
export function onComfyMessage(fn: Listener): () => void {
  listeners.add(fn)
  connect()
  return () => listeners.delete(fn)
}

function connect(): void {
  if (socket) return
  const url = new URL(`/ws?clientId=${clientId}`, location.href)
  url.protocol = url.protocol.replace('http', 'ws')
  socket = new WebSocket(url)
  socket.binaryType = 'arraybuffer'
  socket.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return onBinary(ev.data as ArrayBuffer)
    const msg = JSON.parse(ev.data) as ComfyMessage
    for (const fn of listeners) fn(msg)
  }
  socket.onclose = () => {
    socket = null
    if (listeners.size || previewListeners.size) setTimeout(connect, 1000)
  }
}

/** The choices ComfyUI reports for one node input (models, dataset folders, enums). */
export async function inputChoices(nodeClass: string, input: string): Promise<string[]> {
  const res = await fetch(`/object_info/${encodeURIComponent(nodeClass)}`)
  if (!res.ok) return []
  const info = (await res.json())[nodeClass]
  const spec = info?.input?.required?.[input] ?? info?.input?.optional?.[input]
  if (!spec) return []
  // V1 combo: [["a","b"], {...}]   V3 combo: ["COMBO", {options: ["a","b"]}]
  if (Array.isArray(spec[0])) return spec[0] as string[]
  if (spec[0] === 'COMBO') return (spec[1]?.options as string[]) ?? []
  return []
}

/** Upload an image to ComfyUI's input folder; returns the name a LoadImage input takes. */
export async function uploadImage(file: File): Promise<string> {
  const body = new FormData()
  body.append('image', file)
  const res = await fetch('/upload/image', { method: 'POST', body })
  if (!res.ok) throw new Error(`upload failed: ${res.status}`)
  const r = (await res.json()) as { name: string; subfolder?: string }
  return r.subfolder ? `${r.subfolder}/${r.name}` : r.name
}
