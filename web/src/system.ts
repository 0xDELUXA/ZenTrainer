// Memory history from the engine's 1 Hz sampler, polled incrementally. Shared by the
// sidebar meters and the System page.

import { computed, ref } from 'vue'
import { engine, type MemSample, type StorageReport } from './api/engine'

export const samples = ref<MemSample[]>([])
export const memLatest = computed(() => samples.value.at(-1))
const MAX = 900

let timer: ReturnType<typeof setInterval> | undefined
async function poll() {
  try {
    const since = samples.value.at(-1)?.t ?? 0
    const { samples: fresh } = await engine.system(since)
    if (fresh.length) samples.value = [...samples.value, ...fresh].slice(-MAX)
  } catch {
    // engine unreachable (restarting?); keep what we have
  }
}

/** free space where ZenTrainer writes; checked once a minute */
export const disks = ref<StorageReport['disks']>([])
let diskTimer: ReturnType<typeof setInterval> | undefined
async function pollDisks() {
  disks.value = await engine.disks().catch(() => disks.value)
}

export function startSystemSync(): void {
  if (timer) return
  poll()
  timer = setInterval(poll, 2000)
  pollDisks()
  diskTimer = setInterval(pollDisks, 60_000)
}

export const gb = (b?: number) => (b == null ? '–' : (b / 1024 ** 3).toFixed(1))
