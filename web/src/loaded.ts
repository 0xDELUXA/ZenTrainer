// The models ComfyUI holds right now (engine/zentrainer/models/residency.py), polled while anything
// shows them: the sidebar, a run's page, the log drawer.

import { computed, ref } from 'vue'
import { engine, type LoadedModel } from './api/engine'

export const loadedModels = ref<LoadedModel[]>([])
let users = 0
let timer: ReturnType<typeof setTimeout> | undefined

async function poll() {
  try {
    loadedModels.value = (await engine.loadedModels()).filter((m) => m.bytes > 0)
  } catch {
    // engine away: keep the last list
  }
  if (users > 0) timer = setTimeout(poll, 3000)
}

/** start polling; returns the stop for onUnmounted */
export function watchLoaded(): () => void {
  if (users++ === 0) poll()
  return () => {
    if (--users === 0) clearTimeout(timer)
  }
}

export const loadedVram = computed(() => loadedModels.value.reduce((n, m) => n + m.vram, 0))
