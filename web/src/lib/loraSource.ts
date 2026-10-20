// ZenTrainer's LoRA library, shaped as the LoraSource zenkit-ui's LoraPicker and LoraBrowser
// read from (engine routes under /zentrainer/api/lora*). Registered once at start.

import { setLoraSource, type LoraInfo, type LoraItem } from '@nynxz/zenkit-ui'

const API = '/zentrainer/api'
const getJson = async <T>(url: string, fallback: T): Promise<T> => {
  try {
    const r = await fetch(url)
    return r.ok ? ((await r.json()) as T) : fallback
  } catch {
    return fallback
  }
}

export function registerLoraSource(): void {
  setLoraSource({
    list: async () => (await getJson<{ loras: LoraItem[] }>(`${API}/loras`, { loras: [] })).loras,
    favorites: async () => (await getJson<{ loras: string[] }>(`${API}/lora/favorites`, { loras: [] })).loras,
    setFavorite: async (name, pinned) => {
      try {
        const r = await fetch(`${API}/lora/favorites`, { method: 'POST', body: JSON.stringify({ name, pinned }) })
        return ((await r.json()) as { loras: string[] }).loras ?? []
      } catch {
        return []
      }
    },
    previewUrl: (name, size) => `${API}/lora/preview?name=${encodeURIComponent(name)}${size ? `&size=${size}` : ''}`,
    mediaUrl: (name) => `${API}/lora/preview?raw=1&name=${encodeURIComponent(name)}`,
    info: (name) => getJson<LoraInfo | null>(`${API}/lora/info?name=${encodeURIComponent(name)}`, null),
  })
}
