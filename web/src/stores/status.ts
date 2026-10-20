// The app's own housekeeping status for the sidebar: the engine's code changed on disk since
// it started (restart to load it), the web app's source is newer than its build (rebuild it: a
// pull doesn't bring web/dist), a rebuild landed since this tab loaded (reload), or ComfyUI has
// an update upstream. Checked at launch, the engine every minute; ComfyUI once (it fetches from git).

import { ref } from 'vue'
import { engine } from '@/api/engine'

export const engineStale = ref(false)
/** set when this ComfyUI is older than ZenTrainer needs (a node-pack install into an old one) */
export const comfyOld = ref<string | null>(null)
export const comfyBehind = ref(0)
export const restarting = ref(false)
export const webStale = ref(false)
export const webBuildable = ref(false)
/** web/dist was rebuilt after this tab loaded it: a reload shows the new page */
export const webNewer = ref(false)
let webLoaded: number | null | undefined

let started = false
export function startStatusSync(): void {
  if (started) return
  started = true
  const tick = async () => {
    const h = await engine.health().catch(() => null)
    if (h) {
      engineStale.value = !!h.stale
      comfyOld.value = h.comfy_old ?? null
      webStale.value = !!h.web_stale
      webBuildable.value = !!h.web_buildable
      if (webLoaded === undefined) webLoaded = h.web_built
      webNewer.value = h.web_built != null && h.web_built !== webLoaded
    }
  }
  tick()
  setInterval(tick, 60_000)
  setTimeout(async () => {
    const v = await engine.comfyui(true).catch(() => null)
    comfyBehind.value = v?.upstream?.behind ?? 0
  }, 15_000)
}

/** Load what's waiting: rebuild the web app if its source moved on, restart ComfyUI if the
 *  engine's code did, then reload the page (once ComfyUI answers again). */
export async function restartEngine(): Promise<void> {
  restarting.value = true
  try {
    if (webStale.value) await engine.buildWeb()
    if (!engineStale.value) return location.reload()
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
