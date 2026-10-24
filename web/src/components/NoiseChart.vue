<script setup lang="ts">
// Where steps land on a flow model's noise schedule, read left to right as an image comes out of
// the noise: full noise (left), where training shapes composition, through form, to nearly clean
// (right), where it shapes detail. Each zone names its share of the steps. A curve is how the
// steps spread (higher = more of them there); `levels` are a scheduler's exact steps, drawn as
// ticks; `weights` a loss-weight curve; `points` [σ, loss] of trained samples, as dots with their
// average.
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

export interface NoiseSeries {
  label: string
  /** shares of the steps in equal stretches of σ, 1 (first) down to 0 */
  hist?: number[]
  /** exact levels instead: [σ, share] */
  levels?: [number, number][]
  kind: 'area' | 'line'
}

const props = withDefaults(defineProps<{
  series: NoiseSeries[]
  /** loss weights at evenly spaced σ, 1 (first) down to 0 */
  weights?: number[] | null
  /** [σ, loss] of trained samples */
  points?: [number, number][]
  /** the zones' shares of the steps [composition, form, detail], and the comparison's */
  zones?: number[] | null
  zonesCompare?: number[] | null
  height?: number
}>(), { weights: null, points: () => [], zones: null, zonesCompare: null, height: 140 })

const W = 600
const H = computed(() => props.height)
const TOP = 34 // room for the zone labels
const BOTTOM = 4
const plotH = computed(() => H.value - TOP - BOTTOM)
const x = (sigma: number) => (1 - Math.min(1, Math.max(0, sigma))) * W
const yOf = (v: number, max: number) => TOP + plotH.value * (1 - Math.min(1, v / (max || 1)))
const base = computed(() => TOP + plotH.value)

// curves share one scale, so the run and the comparison read against each other
const curveMax = computed(() => Math.max(1e-9, ...props.series.flatMap((s) => s.hist ?? [])) * 1.05)
function path(hist: number[], close: boolean): string {
  const n = hist.length
  const pts = hist.map((v, i) => `${((i + 0.5) / n) * W},${yOf(v, curveMax.value)}`)
  return close ? `M0,${base.value}L${pts.join('L')}L${W},${base.value}Z` : `M${pts.join('L')}`
}
const weightMax = computed(() => Math.max(1e-9, ...(props.weights ?? [])) * 1.1)
const weightPath = computed(() => {
  const w = props.weights
  if (!w?.length) return ''
  return `M${w.map((v, i) => `${(i / (w.length - 1)) * W},${yOf(v, weightMax.value)}`).join('L')}`
})

// trained samples: the loss axis runs to the 98th percentile, so a few outliers don't flatten it
const BINS = 24
const lossMax = computed(() => {
  const ls = props.points.map((p) => p[1]).sort((a, b) => a - b)
  return ls.length ? ls[Math.floor((ls.length - 1) * 0.98)]! * 1.1 : 1
})
const dots = computed(() => {
  const pts = props.points
  const every = Math.max(1, Math.ceil(pts.length / 1500)) // at most ~1500 dots
  return pts.filter((_, i) => i % every === 0).map(([s, l]) => ({ cx: x(s), cy: yOf(l, lossMax.value) }))
})
// the dots on one canvas, not an SVG circle each: as circles, every hover move (and every step)
// re-rendered and diffed all of them, which lagged the page on slower machines during long runs.
// Redrawn only when they or the plot's size change.
const canvas = ref<HTMLCanvasElement | null>(null)
function drawDots() {
  const c = canvas.value
  if (!c) return
  const w = c.clientWidth
  const h = c.clientHeight
  const dpr = window.devicePixelRatio || 1
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr)
    c.height = Math.round(h * dpr)
  }
  const g = c.getContext('2d')
  if (!g) return
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  g.clearRect(0, 0, w, h)
  g.fillStyle = getComputedStyle(c).color
  const sx = w / W
  const sy = h / H.value
  const r = 1.6
  g.beginPath()
  for (const d of dots.value) {
    g.moveTo(d.cx * sx + r, d.cy * sy)
    g.arc(d.cx * sx, d.cy * sy, r, 0, Math.PI * 2)
  }
  g.fill()
}
let sized: ResizeObserver | undefined
onMounted(() => {
  drawDots()
  if (canvas.value) (sized = new ResizeObserver(drawDots)).observe(canvas.value)
})
onBeforeUnmount(() => sized?.disconnect())
watch(dots, drawDots)
const lossByBin = computed(() => {
  const sum = new Array<number>(BINS).fill(0)
  const cnt = new Array<number>(BINS).fill(0)
  for (const [s, l] of props.points) {
    const b = Math.min(BINS - 1, Math.max(0, Math.floor((1 - s) * BINS)))
    sum[b]! += l
    cnt[b]! += 1
  }
  return sum.map((v, i) => (cnt[i]! >= 3 ? v / cnt[i]! : null))
})
const lossPath = computed(() => {
  let d = ''
  lossByBin.value.forEach((v, i) => {
    if (v == null) return
    d += `${d && lossByBin.value[i - 1] != null ? 'L' : 'M'}${((i + 0.5) / BINS) * W},${yOf(v, lossMax.value)}`
  })
  return d
})

const ZONES = [
  { name: 'Composition', hint: 'layout, pose, colour' },
  { name: 'Form', hint: 'shapes, features' },
  { name: 'Detail', hint: 'faces, texture' },
]
const pct = (v: number) => `${Math.round(v * 100)}%`

// hover: how noisy there, and what's there
const hover = ref<number | null>(null)
function onMove(ev: PointerEvent) {
  const box = (ev.currentTarget as HTMLElement).getBoundingClientRect()
  hover.value = Math.min(1, Math.max(0, 1 - (ev.clientX - box.left) / box.width))
}
const readout = computed(() => {
  const s = hover.value
  if (s == null) return null
  const parts: string[] = []
  if (props.weights?.length) {
    const w = props.weights
    parts.push(`loss counts ${w[Math.round((1 - s) * (w.length - 1))]!.toFixed(2)}×`)
  }
  const l = lossByBin.value[Math.min(BINS - 1, Math.floor((1 - s) * BINS))]
  if (l != null) parts.push(`average loss ${l.toFixed(3)}`)
  return { s, text: `${Math.round(s * 100)}% noise${parts.length ? ` · ${parts.join(' · ')}` : ''}` }
})
</script>

<template>
  <div class="noise-chart">
    <div class="plot" :style="{ height: `${H}px` }" @pointermove="onMove" @pointerleave="hover = null">
      <!-- three layers, as one picture: the zones and histograms, the dots, then the lines on top -->
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none">
        <rect v-for="i in 3" :key="`z${i}`" :x="((i - 1) * W) / 3" :y="0" :width="W / 3" :height="H" :class="`zone z${i - 1}`" />
        <template v-for="(s, i) in series" :key="i">
          <template v-if="s.hist">
            <path v-if="s.kind === 'area'" :d="path(s.hist, true)" class="area" />
            <path :d="path(s.hist, false)" :class="s.kind === 'area' ? 'area-edge' : 'compare'" vector-effect="non-scaling-stroke" />
          </template>
          <!-- exact levels (a scheduler's steps): ticks along the bottom, the curve shows how they spread -->
          <line v-for="([sig], j) in s.levels ?? []" :key="`t${j}`" :x1="x(sig)" :x2="x(sig)" :y1="base" :y2="base - 12" class="tick" vector-effect="non-scaling-stroke" />
        </template>
      </svg>
      <canvas ref="canvas" class="dots" aria-hidden="true" />
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none">
        <path v-if="lossPath" :d="lossPath" class="loss" vector-effect="non-scaling-stroke" />
        <path v-if="weightPath" :d="weightPath" class="weight" vector-effect="non-scaling-stroke" />
        <line v-if="readout" :x1="x(readout.s)" :x2="x(readout.s)" :y1="TOP" :y2="base" class="cursor" vector-effect="non-scaling-stroke" />
      </svg>
      <!-- the hover readout floats inside the plot, beside the cursor: nothing around it moves -->
      <div v-if="readout" class="readout mono" :class="{ flip: readout.s < 0.5 }" :style="{ left: `${(1 - readout.s) * 100}%` }">{{ readout.text }}</div>
      <div class="zone-labels">
        <div v-for="(z, i) in ZONES" :key="z.name" class="zl" :title="z.hint">
          <span class="zn">{{ z.name }}<b v-if="zones">{{ pct(zones[i]!) }}</b></span>
          <span class="zh">{{ zonesCompare ? `model's own ${pct(zonesCompare[i]!)}` : z.hint }}</span>
        </div>
      </div>
    </div>
    <div class="axis small muted"><span>← full noise</span><span>clean →</span></div>
    <div class="foot small">
      <span v-for="(s, i) in series" :key="i" class="key"><i :class="s.kind === 'area' ? 'k-area' : 'k-compare'" />{{ s.label }}</span>
      <span v-if="points.length" class="key"><i class="k-loss" />Average loss (dots: each sample)</span>
      <span v-if="weights?.length" class="key"><i class="k-weight" />How much the loss counts</span>
    </div>
  </div>
</template>

<style scoped>
.noise-chart {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.plot {
  position: relative;
  border-radius: var(--r-md);
  overflow: hidden;
  cursor: crosshair;
}
svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.zone.z0 {
  fill: color-mix(in srgb, var(--zen-accent) 5%, var(--zen-surface-2));
}
.zone.z1 {
  fill: color-mix(in srgb, var(--zen-accent) 9%, var(--zen-surface-2));
}
.zone.z2 {
  fill: color-mix(in srgb, var(--zen-accent) 14%, var(--zen-surface-2));
}
.zone-labels {
  position: absolute;
  inset: 0 0 auto 0;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  pointer-events: none;
}
.zl {
  display: flex;
  flex-direction: column;
  padding: 4px 8px;
  line-height: 1.25;
  min-width: 0;
}
.zn {
  font-size: 11.5px;
  font-weight: 600;
  color: var(--zen-text);
  white-space: nowrap;
}
.zn b {
  margin-left: 6px;
  color: var(--zen-accent);
}
.zh {
  font-size: 10.5px;
  color: var(--zen-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.area {
  fill: color-mix(in srgb, var(--zen-accent) 35%, transparent);
}
.area-edge {
  fill: none;
  stroke: var(--zen-accent);
  stroke-width: 1.5;
}
.compare {
  fill: none;
  stroke: var(--zen-muted);
  stroke-width: 1.5;
  stroke-dasharray: 4 3;
}
.tick {
  stroke: var(--zen-text);
  stroke-width: 2;
}
.dots {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  color: color-mix(in srgb, var(--zen-text) 35%, transparent); /* the dots' fill (drawDots reads it) */
}
.loss {
  fill: none;
  stroke: var(--zen-text);
  stroke-width: 2;
}
.weight {
  fill: none;
  stroke: var(--zen-warn);
  stroke-width: 1.5;
}
.cursor {
  stroke: var(--zen-muted);
  stroke-width: 1;
}
.readout {
  position: absolute;
  bottom: 18px;
  margin-left: 8px;
  padding: 2px 6px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-surface) 92%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  color: var(--zen-text);
  font-size: 10.5px;
  white-space: nowrap;
  pointer-events: none;
}
/* past the middle it sits left of the cursor, so it never runs off the edge */
.readout.flip {
  transform: translateX(-100%);
  margin-left: -8px;
}
.axis {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 10.5px;
}
.foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 12px;
}
.key {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--zen-muted);
}
.key i {
  display: inline-block;
  width: 14px;
  height: 0;
  border-top: 2px solid;
}
.k-area {
  border-color: var(--zen-accent) !important;
}
.k-compare {
  border-top-style: dashed !important;
  border-color: var(--zen-muted) !important;
}
.k-loss {
  border-color: var(--zen-text) !important;
}
.k-weight {
  border-color: var(--zen-warn) !important;
}
</style>
