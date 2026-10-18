// The app's own housekeeping status for the sidebar: the engine's code changed on disk since
// it started (restart to load it), or ComfyUI has an update upstream. Checked at launch, the
// engine every minute; ComfyUI once (its check fetches from git).

import { ref } from 'vue'
import { engine } from './api/engine'

export const engineStale = ref(false)
/** set when this ComfyUI is older than ZenTrainer needs (a node-pack install into an old one) */
export const comfyOld = ref<string | null>(null)
export const comfyBehind = ref(0)
export const restarting = ref(false)

let started = false
export function startStatusSync(): void {
  if (started) return
  started = true
  const tick = async () => {
    const h = await engine.health().catch(() => null)
    if (h) {
      engineStale.value = !!h.stale
      comfyOld.value = h.comfy_old ?? null
    }
  }
  tick()
  setInterval(tick, 60_000)
  setTimeout(async () => {
    const v = await engine.comfyui(true).catch(() => null)
    comfyBehind.value = v?.upstream?.behind ?? 0
  }, 15_000)
}

/** Restart ComfyUI, then reload the page once it answers again. */
export async function restartEngine(): Promise<void> {
  restarting.value = true
  try {
    await engine.restart()
  } catch (e) {
    restarting.value = false
    throw e
  }
  await new Promise((r) => setTimeout(r, 3000))
  for (let i = 0; i < 120; i++) {
    if (await engine.health().then(() => true).catch(() => false)) return location.reload()
    await new Promise((r) => setTimeout(r, 1000))
  }
  restarting.value = false
}
