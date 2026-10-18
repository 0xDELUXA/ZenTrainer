<script setup lang="ts">
// Hold Shift over any picture or video (sample tiles, dataset items, cards) for a large copy of
// it in the middle of the screen, videos playing with sound. An element can name what to show
// with data-peek="url" (a still that stands for a clip, a thumb for a full-size file).
import { onBeforeUnmount, onMounted, ref } from 'vue'

const src = ref<string | null>(null)
const video = ref(false)
const time = ref(0)
let shift = false
let x = 0
let y = 0

const VIDEO = /\.(mp4|webm|mov|mkv)(\?|$)/i

function under(): { url: string; video: boolean; time: number } | null {
  for (const el of document.elementsFromPoint(x, y)) {
    if (el.closest('.peek')) continue
    const tagged = (el as HTMLElement).closest<HTMLElement>('[data-peek]')
    if (tagged?.dataset.peek) return { url: tagged.dataset.peek, video: VIDEO.test(tagged.dataset.peek), time: 0 }
    if (el instanceof HTMLVideoElement && el.currentSrc) return { url: el.currentSrc, video: true, time: el.currentTime }
    if (el instanceof HTMLImageElement && el.currentSrc && el.naturalWidth > 1) return { url: el.currentSrc, video: false, time: 0 }
  }
  return null
}

function update() {
  const hit = shift ? under() : null
  if (hit?.url === src.value) return
  src.value = hit?.url ?? null
  video.value = !!hit?.video
  time.value = hit?.time ?? 0
}

function onMove(e: PointerEvent) {
  x = e.clientX
  y = e.clientY
  shift = e.shiftKey
  if (shift || src.value) update()
}
function onKey(e: KeyboardEvent) {
  if (e.key !== 'Shift') return
  const t = e.target as HTMLElement | null
  if (e.type === 'keydown' && t?.closest('input, textarea, [contenteditable="true"]')) return // typing capitals
  shift = e.type === 'keydown'
  update()
}
function off() {
  shift = false
  update()
}
function start(e: Event) {
  const v = e.target as HTMLVideoElement
  if (time.value) v.currentTime = time.value // carry on from where the tile was
}

onMounted(() => {
  window.addEventListener('pointermove', onMove, { passive: true })
  window.addEventListener('keydown', onKey)
  window.addEventListener('keyup', onKey)
  window.addEventListener('blur', off)
})
onBeforeUnmount(() => {
  window.removeEventListener('pointermove', onMove)
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('keyup', onKey)
  window.removeEventListener('blur', off)
})
</script>

<template>
  <Transition name="peek">
    <div v-if="src" class="peek" aria-hidden="true">
      <video v-if="video" :key="src" :src="src" autoplay loop playsinline @loadedmetadata="start" />
      <img v-else :src="src" alt="" />
    </div>
  </Transition>
</template>

<style scoped>
.peek {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  pointer-events: none;
  background: color-mix(in srgb, var(--zen-bg) 55%, transparent);
  backdrop-filter: blur(3px);
}
.peek video,
.peek img {
  max-width: 86vw;
  max-height: 86vh;
  border-radius: var(--r-lg, 12px);
  border: 1px solid var(--zen-border);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
  background: #000;
  display: block;
}
.peek-enter-active,
.peek-leave-active {
  transition: opacity var(--zen-dur-fast) ease;
}
.peek-enter-from,
.peek-leave-to {
  opacity: 0;
}
</style>
