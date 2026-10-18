<script setup lang="ts">
// The mask editor for one dataset image: find the subject with SAM3 (a word, and/or clicks:
// left keeps, right excludes), touch it up with a brush, then grow/shrink and feather it, and
// save it as the image's mask (<name>-masklabel.png; white trains at full weight).
// Grow/shrink and feather apply on top of the raw mask, so they can be changed freely.
// Keys: [ / ] brush size, B brush, E eraser, P points, ←/→ previous/next image, Ctrl+S save.
import { ZenButton, ZenInput, ZenSlider } from '@nynxz/zenkit-ui'
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { EngineError, engine, type DatasetItem } from '@/api/engine'

const props = defineProps<{ name: string }>()
const route = useRoute()
const router = useRouter()

const items = ref<DatasetItem[]>([])
const path = computed(() => (typeof route.query.path === 'string' ? route.query.path : ''))
const index = computed(() => items.value.findIndex((i) => i.path === path.value))
const item = computed(() => items.value[index.value] ?? null)

// original files through ComfyUI's /view (datasets live in its input folder)
const fileUrl = (rel: string) => {
  const parts = `${props.name}/${rel}`.split('/')
  const filename = parts.pop()!
  return `/view?${new URLSearchParams({ filename, subfolder: parts.join('/'), type: 'input' })}&t=${Date.now()}`
}

// ---- canvases ------------------------------------------------------------------------------
const stage = ref<HTMLDivElement | null>(null)
const view = ref<HTMLCanvasElement | null>(null)
const img = new Image()
const work = document.createElement('canvas') // the raw mask (white = keep), image resolution
const W = ref(0)
const H = ref(0)
const loaded = ref(false)
const dirty = ref(false)
const hasMask = ref(false)

const tool = ref<'points' | 'brush' | 'erase'>('points')
const brush = ref(40)
const grow = ref(0) // px, negative shrinks
const feather = ref(0) // px
const show = ref<'overlay' | 'mask'>('overlay')
const text = ref('')
const threshold = ref(0.5)
const points = ref<{ x: number; y: number; keep: boolean }[]>([])
const busy = ref(false)
const status = ref('')
const error = ref<string | null>(null)

function wctx() {
  return work.getContext('2d', { willReadFrequently: true })!
}

async function load() {
  loaded.value = false
  dirty.value = false
  points.value = []
  error.value = null
  status.value = ''
  if (!item.value) return
  await new Promise<void>((res, rej) => {
    img.onload = () => res()
    img.onerror = () => rej(new Error('image failed to load'))
    img.src = fileUrl(item.value!.path)
  }).catch((e) => (error.value = String(e)))
  W.value = img.naturalWidth
  H.value = img.naturalHeight
  work.width = W.value
  work.height = H.value
  const c = wctx()
  c.fillStyle = '#000'
  c.fillRect(0, 0, W.value, H.value)
  hasMask.value = !!item.value.mask
  if (item.value.mask) {
    const m = new Image()
    await new Promise<void>((res) => {
      m.onload = () => res()
      m.onerror = () => res()
      m.src = fileUrl(item.value!.mask!)
    })
    if (m.naturalWidth) c.drawImage(m, 0, 0, W.value, H.value)
  }
  loaded.value = true
  await nextTick()
  render()
}

// the final mask: raw, grown or shrunk, then feathered (white = keep)
function finalMask(): HTMLCanvasElement {
  const out = document.createElement('canvas')
  out.width = W.value
  out.height = H.value
  const o = out.getContext('2d', { willReadFrequently: true })!
  if (grow.value) {
    const r = Math.abs(grow.value)
    o.filter = `blur(${r / 2}px)`
    o.drawImage(work, 0, 0)
    o.filter = 'none'
    const d = o.getImageData(0, 0, W.value, H.value)
    const cut = grow.value > 0 ? 8 : 247 // blur then a low cut grows, a high cut shrinks
    for (let i = 0; i < d.data.length; i += 4) {
      const v = d.data[i]! > cut ? 255 : 0
      d.data[i] = d.data[i + 1] = d.data[i + 2] = v
      d.data[i + 3] = 255
    }
    o.putImageData(d, 0, 0)
  } else o.drawImage(work, 0, 0)
  if (feather.value) {
    const tmp = document.createElement('canvas')
    tmp.width = W.value
    tmp.height = H.value
    const t = tmp.getContext('2d')!
    t.filter = `blur(${feather.value}px)`
    t.drawImage(out, 0, 0)
    o.clearRect(0, 0, W.value, H.value)
    o.drawImage(tmp, 0, 0)
  }
  return out
}

// view geometry: the image fitted in the stage
const fit = ref({ x: 0, y: 0, s: 1 })
function render() {
  const v = view.value
  const st = stage.value
  if (!v || !st || !loaded.value) return
  const dpr = window.devicePixelRatio || 1
  const cw = st.clientWidth
  const ch = st.clientHeight
  v.width = cw * dpr
  v.height = ch * dpr
  v.style.width = `${cw}px`
  v.style.height = `${ch}px`
  const s = Math.min(cw / W.value, ch / H.value)
  fit.value = { x: (cw - W.value * s) / 2, y: (ch - H.value * s) / 2, s }
  const c = v.getContext('2d')!
  c.setTransform(dpr, 0, 0, dpr, 0, 0)
  c.clearRect(0, 0, cw, ch)
  const { x, y } = fit.value
  const mask = finalMask()
  if (show.value === 'mask') {
    c.drawImage(mask, x, y, W.value * s, H.value * s)
  } else {
    c.drawImage(img, x, y, W.value * s, H.value * s)
    // tint what's excluded (black in the mask), so the kept subject reads clean
    const tint = document.createElement('canvas')
    tint.width = W.value
    tint.height = H.value
    const t = tint.getContext('2d', { willReadFrequently: true })!
    t.drawImage(mask, 0, 0)
    const d = t.getImageData(0, 0, W.value, H.value)
    for (let i = 0; i < d.data.length; i += 4) {
      const keep = d.data[i]! / 255
      d.data[i] = 20
      d.data[i + 1] = 14
      d.data[i + 2] = 40
      d.data[i + 3] = Math.round((1 - keep) * 175)
    }
    t.putImageData(d, 0, 0)
    c.drawImage(tint, x, y, W.value * s, H.value * s)
  }
  for (const p of points.value) {
    c.beginPath()
    c.arc(x + p.x * s, y + p.y * s, 7, 0, Math.PI * 2)
    c.fillStyle = p.keep ? '#22c55e' : '#ef4444'
    c.fill()
    c.lineWidth = 2
    c.strokeStyle = '#fff'
    c.stroke()
  }
}
watch([grow, feather, show], render)

// ---- input ---------------------------------------------------------------------------------
const toImage = (e: PointerEvent) => {
  const r = view.value!.getBoundingClientRect()
  return { x: (e.clientX - r.left - fit.value.x) / fit.value.s, y: (e.clientY - r.top - fit.value.y) / fit.value.s }
}
let painting = false
let last: { x: number; y: number } | null = null
function stroke(p: { x: number; y: number }) {
  const c = wctx()
  c.strokeStyle = c.fillStyle = tool.value === 'erase' ? '#000' : '#fff'
  c.lineCap = 'round'
  c.lineWidth = brush.value
  c.beginPath()
  if (last) {
    c.moveTo(last.x, last.y)
    c.lineTo(p.x, p.y)
    c.stroke()
  } else {
    c.arc(p.x, p.y, brush.value / 2, 0, Math.PI * 2)
    c.fill()
  }
  last = p
  dirty.value = true
  render()
}
function down(e: PointerEvent) {
  if (!loaded.value) return
  const p = toImage(e)
  if (p.x < 0 || p.y < 0 || p.x > W.value || p.y > H.value) return
  if (tool.value === 'points') {
    points.value = [...points.value, { x: Math.round(p.x), y: Math.round(p.y), keep: e.button !== 2 && !e.altKey }]
    render()
    detect()
    return
  }
  painting = true
  last = null
  ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  stroke(p)
}
function move(e: PointerEvent) {
  cursor.value = { x: e.offsetX, y: e.offsetY }
  if (painting) stroke(toImage(e))
}
function up() {
  painting = false
  last = null
}
const cursor = ref<{ x: number; y: number } | null>(null)

// ---- SAM3 ----------------------------------------------------------------------------------
let seq = 0
async function detect() {
  if (!item.value || (!text.value.trim() && !points.value.some((p) => p.keep))) return
  const mine = ++seq
  busy.value = true
  error.value = null
  status.value = 'Finding…'
  try {
    let { job } = await engine.segment(props.name, {
      path: item.value.path,
      text: text.value.trim(),
      positive: points.value.filter((p) => p.keep).map(({ x, y }) => ({ x, y })),
      negative: points.value.filter((p) => !p.keep).map(({ x, y }) => ({ x, y })),
      threshold: threshold.value,
    })
    while (job.status === 'queued' || job.status === 'running') {
      status.value = job.message || 'Finding…'
      await new Promise((r) => setTimeout(r, 1000))
      if (mine !== seq) return
      job = await engine.segmentJob(job.job_id)
    }
    if (mine !== seq) return
    if (job.status === 'error') throw new Error(job.message)
    const m = new Image()
    await new Promise<void>((res, rej) => {
      m.onload = () => res()
      m.onerror = () => rej(new Error('mask failed to load'))
      m.src = `/zentrainer/api/segment/${job.job_id}.png`
    })
    const c = wctx()
    c.drawImage(m, 0, 0, W.value, H.value)
    dirty.value = true
    status.value = 'Found. Click to add (left) or exclude (right), or touch up with the brush.'
    render()
  } catch (e) {
    if (mine === seq) error.value = e instanceof EngineError ? e.message : (e as Error).message
    status.value = ''
  } finally {
    if (mine === seq) busy.value = false
  }
}

// ---- edits ---------------------------------------------------------------------------------
function fill(white: boolean) {
  const c = wctx()
  c.fillStyle = white ? '#fff' : '#000'
  c.fillRect(0, 0, W.value, H.value)
  dirty.value = true
  render()
}
function invert() {
  const c = wctx()
  const d = c.getImageData(0, 0, W.value, H.value)
  for (let i = 0; i < d.data.length; i += 4) d.data[i] = d.data[i + 1] = d.data[i + 2] = 255 - d.data[i]!
  c.putImageData(d, 0, 0)
  dirty.value = true
  render()
}
const coverage = computed(() => {
  void dirty.value
  if (!loaded.value) return null
  const d = wctx().getImageData(0, 0, W.value, H.value).data
  let s = 0
  for (let i = 0; i < d.length; i += 4 * 16) s += d[i]!
  return s / (255 * (d.length / (4 * 16)))
})

async function save() {
  if (!item.value) return
  busy.value = true
  error.value = null
  try {
    await engine.saveMask(props.name, item.value.path, finalMask().toDataURL('image/png'))
    dirty.value = false
    hasMask.value = true
    grow.value = 0
    feather.value = 0
    status.value = 'Saved'
    await refresh()
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
async function remove() {
  if (!item.value || !confirm('Delete this image’s mask? It trains on the whole image again.')) return
  await engine.saveMask(props.name, item.value.path, '')
  await refresh()
  await load()
}
async function refresh() {
  items.value = (await engine.dataset(props.name)).items.filter((i) => i.kind !== 'video')
}
function go(by: number) {
  const next = items.value[index.value + by]
  if (!next) return
  if (dirty.value && !confirm('Leave without saving this mask?')) return
  router.replace({ query: { path: next.path } })
}

function onKey(e: KeyboardEvent) {
  if ((e.target as HTMLElement).closest('input, textarea')) return
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault()
    save()
  } else if (e.key === '[') brush.value = Math.max(4, brush.value - 8)
  else if (e.key === ']') brush.value = Math.min(400, brush.value + 8)
  else if (e.key === 'b') tool.value = 'brush'
  else if (e.key === 'e') tool.value = 'erase'
  else if (e.key === 'p') tool.value = 'points'
  else if (e.key === 'ArrowLeft') go(-1)
  else if (e.key === 'ArrowRight') go(1)
}
const ro = new ResizeObserver(() => render())
onMounted(async () => {
  await refresh()
  if (!path.value && items.value[0]) router.replace({ query: { path: items.value[0].path } })
  else load()
  addEventListener('keydown', onKey)
  if (stage.value) ro.observe(stage.value)
})
onUnmounted(() => {
  removeEventListener('keydown', onKey)
  ro.disconnect()
})
watch(path, load)
</script>

<template>
  <div class="page mask-page">
    <div class="page-head">
      <div class="titles">
        <h1>
          <RouterLink :to="`/datasets/${encodeURIComponent(name)}`" class="crumb">{{ name }}</RouterLink>
          Mask
        </h1>
        <p class="lede mono">{{ item?.path }} <span v-if="items.length" class="muted">· {{ index + 1 }} / {{ items.length }}</span></p>
      </div>
      <ZenButton icon="mdi mdi-chevron-left" :disabled="index <= 0" @click="go(-1)">Previous</ZenButton>
      <ZenButton icon="mdi mdi-chevron-right" :disabled="index >= items.length - 1" @click="go(1)">Next</ZenButton>
    </div>

    <div class="editor">
      <div ref="stage" class="stage card" :class="tool" @contextmenu.prevent>
        <canvas ref="view" @pointerdown="down" @pointermove="move" @pointerup="up" @pointerleave="(cursor = null), up()" />
        <span
          v-if="cursor && tool !== 'points'"
          class="brush-ring"
          :style="{ left: `${cursor.x}px`, top: `${cursor.y}px`, width: `${brush * fit.s}px`, height: `${brush * fit.s}px` }"
        />
        <div v-if="!loaded && !error" class="loading muted small"><i class="mdi mdi-loading mdi-spin" /> Loading…</div>
      </div>

      <aside class="side">
        <section class="card grp">
          <div class="eyebrow">Find it (SAM3)</div>
          <div class="row">
            <ZenInput v-model="text" size="sm" placeholder="what to keep, e.g. person" @keydown.enter="detect" />
            <ZenButton size="sm" icon="mdi mdi-magnify" :disabled="busy || !text.trim()" @click="detect">Find</ZenButton>
          </div>
          <p class="muted small">Or pick <b>Points</b> and click the subject: left-click keeps, right-click excludes.</p>
          <label class="f small">
            <span class="muted">Confidence {{ threshold.toFixed(2) }}</span>
            <ZenSlider v-model="threshold" :min="0.1" :max="0.9" :step="0.05" />
          </label>
          <div v-if="points.length" class="row small">
            <span class="muted">{{ points.filter((p) => p.keep).length }} keep · {{ points.filter((p) => !p.keep).length }} exclude</span>
            <button class="link" @click="(points = []), render()">Clear points</button>
          </div>
        </section>

        <section class="card grp">
          <div class="eyebrow">Tools</div>
          <div class="seg">
            <button :class="{ on: tool === 'points' }" title="P" @click="tool = 'points'"><i class="mdi mdi-gesture-tap" /> Points</button>
            <button :class="{ on: tool === 'brush' }" title="B" @click="tool = 'brush'"><i class="mdi mdi-brush" /> Brush</button>
            <button :class="{ on: tool === 'erase' }" title="E" @click="tool = 'erase'"><i class="mdi mdi-eraser" /> Erase</button>
          </div>
          <label v-if="tool !== 'points'" class="f small">
            <span class="muted">Brush size {{ brush }} px · [ ]</span>
            <ZenSlider v-model="brush" :min="4" :max="400" :step="2" />
          </label>
          <div class="row">
            <button class="mini" @click="invert"><i class="mdi mdi-invert-colors" /> Invert</button>
            <button class="mini" @click="fill(true)">Keep all</button>
            <button class="mini" @click="fill(false)">Clear</button>
          </div>
        </section>

        <section class="card grp">
          <div class="eyebrow">Edges</div>
          <label class="f small">
            <span class="muted">{{ grow >= 0 ? 'Grow' : 'Shrink' }} {{ Math.abs(grow) }} px</span>
            <ZenSlider v-model="grow" :min="-60" :max="60" :step="1" />
          </label>
          <label class="f small">
            <span class="muted">Feather {{ feather }} px</span>
            <ZenSlider v-model="feather" :min="0" :max="60" :step="1" />
          </label>
          <div class="seg small">
            <button :class="{ on: show === 'overlay' }" @click="show = 'overlay'">Overlay</button>
            <button :class="{ on: show === 'mask' }" @click="show = 'mask'">Mask only</button>
          </div>
        </section>

        <section class="card grp">
          <div class="row small">
            <span class="muted">{{ coverage != null ? `${Math.round(coverage * 100)}% trains at full weight` : '' }}</span>
          </div>
          <div v-if="status" class="small muted">{{ status }}</div>
          <div v-if="error" class="small err">{{ error }}</div>
          <div class="row">
            <ZenButton variant="primary" icon="mdi mdi-content-save-outline" :disabled="busy || !loaded" @click="save">{{ dirty || grow || feather ? 'Save mask' : 'Saved' }}</ZenButton>
            <ZenButton v-if="hasMask" icon="mdi mdi-trash-can-outline" :disabled="busy" @click="remove">Delete</ZenButton>
          </div>
        </section>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.mask-page {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
.editor {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 14px;
}
.stage {
  position: relative;
  padding: 0;
  overflow: hidden;
  min-height: 420px;
  background: repeating-conic-gradient(color-mix(in srgb, var(--zen-text) 6%, transparent) 0% 25%, transparent 0% 50%) 50% / 24px 24px;
}
.stage canvas {
  position: absolute;
  inset: 0;
  touch-action: none;
}
.stage.points canvas {
  cursor: crosshair;
}
.stage.brush canvas,
.stage.erase canvas {
  cursor: none;
}
.brush-ring {
  position: absolute;
  transform: translate(-50%, -50%);
  border-radius: 50%;
  border: 1.5px solid #fff;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.5);
  pointer-events: none;
}
.loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
}
.side {
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
  scrollbar-width: thin;
}
.grp {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
}
.row {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}
.row :deep(.zen-input) {
  flex: 1;
}
.f {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.seg {
  display: flex;
  gap: 4px;
}
.seg button {
  flex: 1;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 5px 8px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.seg button.on {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
.mini {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 8px;
  border-radius: var(--r-sm);
  cursor: pointer;
}
.mini:hover {
  color: var(--zen-text);
  border-color: var(--zen-accent);
}
.link {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  color: var(--zen-accent);
  cursor: pointer;
}
.small {
  font-size: 12px;
}
.muted {
  color: var(--zen-muted);
}
.err {
  color: var(--zen-danger);
}
@media (max-width: 900px) {
  .editor {
    grid-template-columns: 1fr;
  }
}
</style>
